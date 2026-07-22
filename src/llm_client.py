from __future__ import annotations

import configparser
import os
import random
import time
from time import perf_counter
from typing import Any, Callable, Dict, Generator, TypeVar
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

T = TypeVar("T")


class LLMClient:
    def __init__(self, config_path: str = "config.ini"):
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        api_config = config["API"]
        self.model = api_config.get("model")
        self.enable_thinking = api_config.getboolean("enable_thinking", fallback=False)
        self.base_url = os.getenv("BASE_URL", "")
        self.max_retries = max(1, int(os.getenv("LLM_MAX_RETRIES", "6")))
        self.retry_base_seconds = max(
            0.0, float(os.getenv("LLM_RETRY_BASE_SECONDS", "2"))
        )
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            float(os.getenv("LLM_RETRY_MAX_SECONDS", "30")),
        )
        timeout_seconds = max(1.0, float(os.getenv("LLM_TIMEOUT_SECONDS", "180")))

        self.client = OpenAI(
            api_key = os.getenv("API_KEY"),
            base_url = self.base_url,
            timeout=timeout_seconds,
            # Retry accounting must remain visible to experiment checkpoints.
            max_retries=0,
        )
        self.last_model_timing: Dict[str, float | None] = {"first_char_seconds": None}
        self.last_request_attempts = 0
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
        }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.6,
        max_tokens: int | None = None,
        response_schema: Dict[str, Any] | None = None,
    ) -> str:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._normalized_temperature(temperature),
        }
        extra_body = self._provider_extra_body()
        if extra_body:
            request["extra_body"] = extra_body
        if response_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        response = self._call_with_retry(
            lambda: self.client.chat.completions.create(**request),
            operation="chat",
        )
        self._record_usage(response)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned an empty response")
        return content.strip()

    def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._normalized_temperature(temperature),
            "stream": True,
        }
        extra_body = self._provider_extra_body()
        if extra_body:
            request["extra_body"] = extra_body
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        request_start = perf_counter()
        self.last_model_timing = {"first_char_seconds": None}
        # Opening a stream can be retried. Once content is yielded, restarting
        # would duplicate output and is intentionally left to the caller.
        completion = self._call_with_retry(
            lambda: self.client.chat.completions.create(**request),
            operation="chat_stream_open",
        )

        for chunk in completion:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                if self.last_model_timing["first_char_seconds"] is None:
                    elapsed = round(perf_counter() - request_start, 3)
                    self.last_model_timing["first_char_seconds"] = elapsed
                    print(f"[model timing] input_to_first_char_seconds={elapsed}")
                yield content

    def _call_with_retry(self, call: Callable[[], T], operation: str) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self.last_request_attempts = attempt
            try:
                return call()
            except Exception as exc:
                last_error = exc
                retryable = self._is_retryable(exc)
                if not retryable or attempt >= self.max_retries:
                    print(
                        f"[LLM {operation} error] attempt={attempt}/{self.max_retries} "
                        f"retryable={retryable} error={type(exc).__name__}: {exc}"
                    )
                    raise

                retry_after = self._retry_after_seconds(exc)
                if retry_after is None:
                    exponential = self.retry_base_seconds * (2 ** (attempt - 1))
                    wait_seconds = min(self.retry_max_seconds, exponential)
                    wait_seconds *= random.uniform(0.8, 1.2)
                else:
                    wait_seconds = min(self.retry_max_seconds, retry_after)
                print(
                    f"[LLM {operation} retry] attempt={attempt}/{self.max_retries} "
                    f"wait={wait_seconds:.1f}s error={type(exc).__name__}: {exc}"
                )
                time.sleep(wait_seconds)

        assert last_error is not None
        raise last_error

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code is not None:
            try:
                status = int(status_code)
            except (TypeError, ValueError):
                status = 0
            return status in {408, 409, 425, 429} or status >= 500

        name = type(exc).__name__.lower()
        transient_markers = (
            "connection", "timeout", "ratelimit", "internalserver",
            "serviceunavailable",
        )
        return isinstance(exc, (ConnectionError, TimeoutError)) or any(
            marker in name for marker in transient_markers
        )

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    def _provider_extra_body(self) -> Dict[str, Any]:
        host = self.base_url.lower()
        if "dashscope" in host:
            return {"enable_thinking": self.enable_thinking}
        if "moonshot" in host or "kimi" in host:
            return {
                "thinking": {
                    "type": "enabled" if self.enable_thinking else "disabled"
                }
            }
        return {}

    def _normalized_temperature(self, requested: float) -> float:
        if self.model == "kimi-k2.6" and (
            "moonshot" in self.base_url.lower() or "kimi" in self.base_url.lower()
        ):
            return 0.6
        return requested

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["calls"] += 1
        print(f"[token usage] prompt={prompt_tokens}, completion={completion_tokens}")

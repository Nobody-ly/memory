from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    attempts: int
    reasoning_content: str = ""


class ChatBackend(Protocol):
    model: str

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.9,
        response_schema: dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> ChatResult: ...


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is None:
        return True
    return int(status_code) in {408, 409, 429} or int(status_code) >= 500


class OpenAICompatibleChatBackend:
    """Small retrying client isolated from the runtime agent configuration."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 180.0,
        max_attempts: int = 6,
        enable_thinking: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")

        self.model = model
        self.base_url = base_url
        self.max_attempts = max_attempts
        self.enable_thinking = enable_thinking
        self.is_kimi_k2 = model.startswith(("kimi-k2.5", "kimi-k2.6"))
        self.is_dashscope_qwen = (
            "dashscope.aliyuncs.com" in base_url.lower()
            and model.lower().startswith("qwen")
        )
        self.is_dashscope_deepseek = (
            "dashscope.aliyuncs.com" in base_url.lower()
            and model.lower().startswith("deepseek")
        )
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
            "network_attempts": 0,
            "network_retries": 0,
        }
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for real PersonaEmp model calls. "
                "Install the project dependencies before running generation."
            ) from exc
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    @classmethod
    def from_env(
        cls,
        prefix: str = "PERSONAEMP_GENERATOR",
    ) -> OpenAICompatibleChatBackend:
        def env(name: str, fallback: str = "") -> str:
            return os.getenv(f"{prefix}_{name}", fallback).strip()

        return cls(
            api_key=env("API_KEY", os.getenv("API_KEY", "")),
            base_url=env("BASE_URL", os.getenv("BASE_URL", "")),
            model=env("MODEL", "qwen3-30b-a3b-instruct-2507"),
            timeout_seconds=float(env("TIMEOUT_SECONDS", "180")),
            max_attempts=int(env("MAX_ATTEMPTS", "6")),
            enable_thinking=env("ENABLE_THINKING", "false").lower()
            in {"1", "true", "yes", "on"},
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.9,
        response_schema: dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        last_error: Exception | None = None
        if not hasattr(self, "token_usage"):
            self.token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
                "network_attempts": 0,
                "network_retries": 0,
            }

        thinking_enabled = (
            self.enable_thinking
            if enable_thinking is None
            else bool(enable_thinking)
        )
        thinking_prompt_schema = bool(
            response_schema is not None
            and getattr(self, "is_dashscope_qwen", False)
            and thinking_enabled
        )
        for attempt in range(1, self.max_attempts + 1):
            self.token_usage["network_attempts"] += 1
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": (
                        [{"role": "system", "content": system_prompt}]
                        if system_prompt.strip()
                        else []
                    )
                    + [{
                        "role": "user",
                        "content": (
                            user_prompt
                            + "\n\nReturn one JSON object matching this exact schema:\n"
                            + json.dumps(
                                response_schema["schema"], ensure_ascii=False
                            )
                            if thinking_prompt_schema
                            else user_prompt
                        ),
                    }],
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                }
                if self.is_kimi_k2:
                    request["temperature"] = (
                        1.0 if thinking_enabled else 0.6
                    )
                    request["extra_body"] = {
                        "thinking": {
                            "type": (
                                "enabled"
                                if thinking_enabled
                                else "disabled"
                            )
                        }
                    }
                else:
                    request["temperature"] = temperature
                    if self.is_dashscope_qwen or getattr(
                        self, "is_dashscope_deepseek", False
                    ):
                        request["extra_body"] = {
                            "enable_thinking": thinking_enabled
                        }
                    elif thinking_enabled:
                        request["extra_body"] = {"enable_thinking": True}
                if response_schema is not None:
                    if thinking_prompt_schema:
                        pass
                    elif self._uses_required_tool_schema():
                        schema_name = str(response_schema.get("name") or "").strip()
                        schema = response_schema.get("schema")
                        if not schema_name or not isinstance(schema, dict):
                            raise ValueError(
                                "Kimi structured output requires schema name and schema"
                            )
                        request["tools"] = [
                            {
                                "type": "function",
                                "function": {
                                    "name": schema_name,
                                    "description": (
                                        "Return the validated structured result."
                                    ),
                                    "parameters": schema,
                                },
                            }
                        ]
                        request["tool_choice"] = {
                            "type": "function",
                            "function": {"name": schema_name},
                        }
                    else:
                        request["response_format"] = {
                            "type": "json_schema",
                            "json_schema": response_schema,
                        }

                response = self.client.chat.completions.create(**request)
                message = response.choices[0].message
                if (
                    response_schema is not None
                    and self._uses_required_tool_schema()
                    and not thinking_prompt_schema
                ):
                    tool_calls = list(message.tool_calls or [])
                    if (
                        len(tool_calls) != 1
                        or tool_calls[0].function.name
                        != response_schema["name"]
                    ):
                        raise RuntimeError(
                            "Kimi did not return the required structured tool call"
                        )
                    content = str(
                        tool_calls[0].function.arguments or ""
                    ).strip()
                else:
                    content = (message.content or "").strip()
                if not content:
                    raise RuntimeError("model returned an empty response")
                reasoning_content = str(
                    getattr(message, "reasoning_content", "") or ""
                ).strip()

                usage = getattr(response, "usage", None)
                result = ChatResult(
                    content=content,
                    model=self.model,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(
                        getattr(usage, "completion_tokens", 0) or 0
                    ),
                    latency_seconds=round(time.perf_counter() - started, 4),
                    attempts=attempt,
                    reasoning_content=reasoning_content,
                )
                self.token_usage["prompt_tokens"] += result.prompt_tokens
                self.token_usage["completion_tokens"] += result.completion_tokens
                self.token_usage["calls"] += 1
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts or not _is_retryable(exc):
                    break
                self.token_usage["network_retries"] += 1
                retry_after = _retry_after_seconds(exc)
                wait_seconds = (
                    retry_after
                    if retry_after is not None
                    else min(1.5 * (2 ** (attempt - 1)), 30.0) + random.random()
                )
                time.sleep(wait_seconds)

        assert last_error is not None
        raise RuntimeError(
            f"model call failed after {attempt} attempts: {last_error}"
        ) from last_error

    def available_models(self) -> list[str]:
        """Return model IDs visible to the configured credential."""
        response = self.client.models.list()
        return sorted(
            str(item.id)
            for item in getattr(response, "data", [])
            if getattr(item, "id", None)
        )

    def _uses_required_tool_schema(self) -> bool:
        """Use forced tool arguments where provider JSON Schema is not strict."""
        return self.is_kimi_k2 or getattr(self, "is_dashscope_qwen", False)

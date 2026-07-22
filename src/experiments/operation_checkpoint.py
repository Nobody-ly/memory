"""Atomic resumable operation and result checkpointing for experiments."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List


class CheckpointSignatureError(RuntimeError):
    pass


class OperationCheckpoint:
    def __init__(self, path: Path, run_signature: str):
        self.path = path
        self.run_signature = run_signature
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": 1,
                "run_signature": self.run_signature,
                "operations": {},
                "results": {},
                "failures": {},
            }
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("run_signature") != self.run_signature:
            raise CheckpointSignatureError(
                "checkpoint does not match this run; use another output directory or --fresh"
            )
        return data

    def execute(
        self,
        key: str,
        operation: Callable[[], Any],
        validator: Callable[[Any], Any],
        max_attempts: int,
        usage_supplier: Callable[[], Dict[str, int]] | None = None,
    ) -> Any:
        cached = self.data["operations"].get(key)
        if cached and cached.get("status") == "complete":
            return cached["value"]

        started = perf_counter()
        usage_before = usage_supplier() if usage_supplier else {}
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                value = validator(operation())
                self.data["operations"][key] = {
                    "status": "complete",
                    "value": value,
                    "attempts": attempt,
                    "elapsed_seconds": round(perf_counter() - started, 3),
                    "token_usage": _usage_delta(
                        usage_before, usage_supplier() if usage_supplier else {}
                    ),
                    "completed_at_utc": _now(),
                }
                self.data["failures"].pop(key, None)
                self.save()
                return value
            except Exception as exc:
                last_error = exc
                self.data["failures"][key] = {
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                    "updated_at_utc": _now(),
                }
                self.save()
                print(
                    f"[operation retry] key={key} attempt={attempt}/{max_attempts} "
                    f"error={type(exc).__name__}: {exc}"
                )
        assert last_error is not None
        raise last_error

    def store_result(self, result_id: str, result: Dict[str, Any]) -> None:
        self.data["results"][result_id] = result
        self.save()

    def store_excluded_result(self, result_id: str, failure: Dict[str, Any]) -> None:
        self.data["failures"][f"sample:{result_id}"] = failure
        self.save()

    def result_values(self) -> List[Dict[str, Any]]:
        return list(self.data["results"].values())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in ("prompt_tokens", "completion_tokens", "calls")
    }

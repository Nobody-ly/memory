"""Auditable adapter for AlpsBench's published two-stage annotation code."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ._vendor import alpsbench_wildchat_two_stage_pipeline as upstream
from .client import ChatBackend
from .generation import prompt_hash
from .wildchat_reconstruction import PAPER_MEMORY_MODEL


ALPSBENCH_SOURCE_COMMIT = "0fd801c0eac123510013e14aa0d032c8392de9dc"
ALPSBENCH_SOURCE_PATH = "src/utils/wildchat_two_stage_pipeline.py"
ALPSBENCH_ORIGINAL_SOURCE_SHA256 = (
    "d4c23a3340f55c14d19d121e28b8086884da8a02ea52984063b12051d19f2c06"
)
PROTOCOL = "alpsbench_two_stage_0fd801c_deepseek_v3_2"


def _session(source: dict[str, Any]) -> dict[str, Any]:
    timestamp = source.get("source_timestamp")
    return {
        "session_id": source["session_id"],
        "started_at": timestamp,
        "ended_at": timestamp,
        "turns": [
            {
                "utterance_index": index,
                "timestamp": timestamp,
                "role": turn["role"],
                "text": turn["text"],
            }
            for index, turn in enumerate(source["turns"])
        ],
    }


def _conversation_sha256(source: dict[str, Any]) -> str:
    value = json.dumps(source["turns"], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AlpsBenchTwoStageExtractor:
    """Run the official two-stage prompts with the paper-specified model."""

    def __init__(self, backend: ChatBackend) -> None:
        if backend.model != PAPER_MEMORY_MODEL:
            raise ValueError(f"expected {PAPER_MEMORY_MODEL}; got {backend.model}")
        client = getattr(backend, "client", None)
        if client is None:
            raise TypeError("AlpsBench adapter requires an OpenAI-compatible client")
        self.backend = backend
        self.client = client

    def run_identity(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "model": self.backend.model,
            "upstream_repository": "ThisIsCosine/AlpsBench",
            "upstream_commit": ALPSBENCH_SOURCE_COMMIT,
            "upstream_path": ALPSBENCH_SOURCE_PATH,
            "upstream_original_source_sha256": ALPSBENCH_ORIGINAL_SOURCE_SHA256,
            "intent_system_prompt_sha256": prompt_hash(
                upstream.INTENT_SYSTEM_PROMPT
            ),
            "intent_user_template_sha256": prompt_hash(
                upstream.INTENT_FEW_SHOT_TEMPLATE
            ),
            "memory_stage1_system_prompt_sha256": prompt_hash(
                upstream.MEMORY_STAGE1_SYSTEM_PROMPT
            ),
            "memory_stage1_user_template_sha256": prompt_hash(
                upstream.MEMORY_STAGE1_FEW_SHOT_TEMPLATE
            ),
            "memory_stage2_system_prompt_sha256": prompt_hash(
                upstream.MEMORY_STAGE2_SYSTEM_PROMPT
            ),
            "response_format": "json_object",
            "intent_temperature": 0.1,
            "memory_stage1_temperature": 0.2,
            "memory_stage2_temperature": 0.0,
        }

    def provenance(self, source: dict[str, Any]) -> dict[str, Any]:
        return {
            **self.run_identity(),
            "conversation_sha256": _conversation_sha256(source),
        }

    def cache_key(self, source: dict[str, Any]) -> str:
        identity = {
            "source_key": source["source_key"],
            **self.provenance(source),
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def extract(self, source: dict[str, Any]) -> dict[str, Any]:
        upstream.client = self.client
        session = _session(source)
        result = upstream.run_two_stage(session, model=self.backend.model)
        return {
            "user_id": "unknown_user",
            "line_index": source["source_row"],
            "sessions": [session],
            "dialogue": source["turns"],
            "intents_ranked": result["intents_ranked"],
            "memory_items": result["memory_items"],
            "memory_stage1_candidates": result["memory_stage1_candidates"],
            "reconstruction_metadata": {
                **self.provenance(source),
                "source_key": source["source_key"],
                "manual_verification": False,
            },
        }

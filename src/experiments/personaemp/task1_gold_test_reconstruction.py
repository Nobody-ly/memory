"""Build a paper-scale PersonaEmp test pool from public Task 1 gold Memory.

The public AlpsBench Task 1 dev/validation records already contain the Memory
annotations. This entry point reuses those annotations, reconstructs only the
missing intent metadata with the published AlpsBench intent prompt, then runs
the official PersonaEmp downstream construction code. It deliberately emits
test-only Random/OOD artifacts and never materialises train data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
from typing import Any, Iterable

from .alpsbench_two_stage import (
    ALPSBENCH_ORIGINAL_SOURCE_SHA256,
    ALPSBENCH_SOURCE_COMMIT,
    PROTOCOL as ALPSBENCH_INTENT_PROTOCOL,
)
from .client import OpenAICompatibleChatBackend
from .reconstruction import (
    ALPSBENCH_REVISION,
    _content_rejection_code,
    _load_jsonl,
    _official_pipeline_summary,
    _sha256,
    _utc_now,
    download_public_task1,
    run_official_pipeline,
    verify_official_checkout,
)
from .splitting import (
    PAPER_TEST_TARGET_USERS,
    build_test_only_split_artifacts,
)
from .generation import prompt_hash
from . import alpsbench_two_stage as upstream_adapter


PAPER_MEMORY_MODEL = "deepseek-v3.2"
PAPER_DATA_MODEL = "MiniMax-M2.5"
PAPER_BIG5_MODEL = "deepseek-v4-flash"


def _terminal_content_rejection_code(exc: Exception) -> str | None:
    reason_code = _content_rejection_code(exc)
    if reason_code is not None:
        return reason_code
    # The published AlpsBench retry helper wraps the final provider exception
    # without preserving its status_code or exception chain.
    message = str(exc).lower()
    if (
        "data_inspection_failed" in message
        and "inappropriate content" in message
    ):
        return "provider_data_inspection_failed"
    return None


def _conversation_from_input(record: dict[str, Any]) -> list[dict[str, Any]]:
    source = record.get("input") or {}
    sessions = source.get("sessions") or []
    if sessions and isinstance(sessions[0], dict):
        turns = sessions[0].get("turns") or []
        if isinstance(turns, list):
            return turns
    dialogue = source.get("dialogue") or []
    return dialogue if isinstance(dialogue, list) else []


def _conversation_sha256(turns: list[dict[str, Any]]) -> str:
    value = json.dumps(turns, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class PublicMemoryFilter:
    """Apply the exact public PersonaEmp filter lists without an API call."""

    def __init__(self, official_repo: Path) -> None:
        path = official_repo / "prepare_dataset" / "filter_list.py"
        if not path.is_file():
            raise FileNotFoundError(path)
        values = runpy.run_path(str(path))
        self.memory_label_list = tuple(
            str(value).lower() for value in values["memory_label_list"]
        )
        self.unmatched_keywords = tuple(
            str(value).lower() for value in values["key_words_for_unmatched"]
        )
        self.source_path = path
        self.source_sha256 = _sha256(path)

    def accepts(self, memories: list[dict[str, Any]]) -> bool:
        for memory in memories:
            label = str(memory.get("label") or "").lower()
            suggestion = str(memory.get("label_suggestion") or "").lower()
            if label and label != "unmapped":
                if any(label.startswith(prefix) for prefix in self.memory_label_list):
                    return True
            elif label == "unmapped":
                if any(keyword in suggestion for keyword in self.unmatched_keywords):
                    return True
        return False


class OfficialIntentCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.values: dict[str, list[dict[str, Any]]] = {}
        self.rejections: dict[str, str] = {}
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("cache_key") or "")
                if not key:
                    continue
                if row.get("status") == "content_rejected":
                    self.rejections[key] = str(
                        row.get("reason_code") or "provider_data_inspection_failed"
                    )
                else:
                    self.values[key] = list(row.get("intents_ranked") or [])

    def save(
        self,
        cache_key: str,
        benchmark_id: str,
        intents_ranked: list[dict[str, Any]],
        provenance: dict[str, Any],
    ) -> None:
        with self._lock:
            if cache_key in self.values or cache_key in self.rejections:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "cache_key": cache_key,
                "benchmark_id": benchmark_id,
                "intents_ranked": intents_ranked,
                "provenance": provenance,
            }
            with self.path.open("a", encoding="utf-8", newline="\n") as destination:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
                os.fsync(destination.fileno())
            self.values[cache_key] = intents_ranked

    def save_rejection(
        self,
        cache_key: str,
        benchmark_id: str,
        reason_code: str,
        provenance: dict[str, Any],
    ) -> None:
        with self._lock:
            if cache_key in self.values or cache_key in self.rejections:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "cache_key": cache_key,
                "benchmark_id": benchmark_id,
                "status": "content_rejected",
                "reason_code": reason_code,
                "provenance": provenance,
            }
            with self.path.open("a", encoding="utf-8", newline="\n") as destination:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
                os.fsync(destination.fileno())
            self.rejections[cache_key] = reason_code


class OfficialIntentReconstructor:
    """Use the published AlpsBench intent prompt only, without memory calls."""

    def __init__(self, backend: OpenAICompatibleChatBackend, cache: OfficialIntentCache) -> None:
        if backend.model != PAPER_MEMORY_MODEL:
            raise ValueError(f"intent model must be {PAPER_MEMORY_MODEL}; got {backend.model}")
        self.backend = backend
        self.cache = cache
        upstream_adapter.upstream.client = self.backend.client

    def _identity(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        turns = _conversation_from_input(record)
        provenance: dict[str, Any] = {
            "model": self.backend.model,
            "upstream_protocol": ALPSBENCH_INTENT_PROTOCOL,
            "upstream_commit": ALPSBENCH_SOURCE_COMMIT,
            "upstream_source_sha256": ALPSBENCH_ORIGINAL_SOURCE_SHA256,
            "intent_system_prompt_sha256": prompt_hash(
                upstream_adapter.upstream.INTENT_SYSTEM_PROMPT
            ),
            "intent_user_template_sha256": prompt_hash(
                upstream_adapter.upstream.INTENT_FEW_SHOT_TEMPLATE
            ),
            "conversation_sha256": _conversation_sha256(turns),
        }
        identity = {
            "benchmark_id": str(record.get("benchmark_id") or ""),
            **provenance,
        }
        return hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(), provenance

    def classify(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        benchmark_id = str(record.get("benchmark_id") or "")
        cache_key, provenance = self._identity(record)
        if cache_key in self.cache.values:
            return self.cache.values[cache_key]
        if cache_key in self.cache.rejections:
            raise RuntimeError(self.cache.rejections[cache_key])
        source = record.get("input") or {}
        sessions = source.get("sessions") or []
        session = sessions[0] if sessions and isinstance(sessions[0], dict) else {
            "session_id": record.get("session_id"),
            "started_at": None,
            "ended_at": None,
            "turns": _conversation_from_input(record),
        }
        try:
            result = upstream_adapter.upstream.analyze_intents(
                session,
                model=self.backend.model,
            )
        except Exception as exc:
            reason_code = _terminal_content_rejection_code(exc)
            if reason_code is None:
                raise
            self.cache.save_rejection(cache_key, benchmark_id, reason_code, provenance)
            raise RuntimeError(reason_code) from exc
        intents = result.get("intents_ranked")
        if not isinstance(intents, list):
            raise ValueError("published intent prompt returned invalid intents_ranked")
        normalized: list[dict[str, Any]] = []
        for item in intents[:3]:
            if not isinstance(item, dict):
                raise ValueError("published intent output contains a non-object item")
            category = str(item.get("intent_category") or "").strip()
            subtype = str(item.get("intent_subtype") or "").strip()
            if not category or not subtype:
                raise ValueError("published intent output lacks category/subtype")
            normalized.append({**item, "intent_category": category, "intent_subtype": subtype})
        self.cache.save(cache_key, benchmark_id, normalized, provenance)
        return normalized


@dataclass(frozen=True)
class Task1GoldStats:
    source_rows: int
    joined_rows: int
    public_memory_rows: int
    memory_filter_pass: int
    memory_filter_rejected: int
    intent_rows_attempted: int
    adapted_records: int
    intent_content_rejections: int


def build_task1_gold_records(
    pairs: Iterable[tuple[Path, Path]],
    classifier: OfficialIntentReconstructor,
    memory_filter: PublicMemoryFilter,
    *,
    source_limit: int | None = None,
    intent_workers: int = 1,
    intent_failures: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], Task1GoldStats]:
    if source_limit is not None and source_limit <= 0:
        raise ValueError("source_limit must be positive")
    if intent_workers < 1:
        raise ValueError("intent_workers must be positive")
    output: list[dict[str, Any]] = []
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    seen: set[str] = set()
    source_rows = joined = public_memory = memory_pass = intent_attempted = 0
    content_rejections = 0
    for input_path, reference_path in pairs:
        references = {
            str(row.get("benchmark_id")): row
            for row in _load_jsonl(reference_path)
        }
        for row in _load_jsonl(input_path):
            if source_limit is not None and source_rows >= source_limit:
                break
            source_rows += 1
            benchmark_id = str(row.get("benchmark_id") or "")
            if not benchmark_id or benchmark_id in seen:
                raise ValueError(f"invalid or duplicate benchmark_id: {benchmark_id!r}")
            seen.add(benchmark_id)
            reference = references.get(benchmark_id)
            if reference is None:
                continue
            joined += 1
            memories = list((reference.get("gold") or {}).get("memory_items") or [])
            if not memories:
                raise ValueError(f"{benchmark_id} has no public gold memories")
            public_memory += 1
            if not memory_filter.accepts(memories):
                continue
            memory_pass += 1
            candidates.append((row, memories))
    intent_attempted = len(candidates)

    def classify_candidate(
        candidate: tuple[dict[str, Any], list[dict[str, Any]]],
    ) -> tuple[str, Any]:
        row, _ = candidate
        try:
            return "ok", classifier.classify(row)
        except RuntimeError as exc:
            if "provider_data_inspection_failed" in str(exc):
                return "content_rejected", str(exc)
            return "error", exc

    with ThreadPoolExecutor(max_workers=intent_workers) as executor:
        results = list(executor.map(classify_candidate, candidates))
    for (row, memories), (status, value) in zip(candidates, results):
        benchmark_id = str(row.get("benchmark_id") or "")
        if status == "error":
            raise value
        if status == "content_rejected":
            content_rejections += 1
            if intent_failures is not None:
                intent_failures.append({
                    "benchmark_id": benchmark_id,
                    "stage": "intent_reconstruction",
                    "reason_code": str(value),
                })
            continue
        intents = value
        source_input = row.get("input") or {}
        output.append({
            "benchmark_id": benchmark_id,
            "line_index": source_input.get("line_index"),
            "sessions": source_input.get("sessions") or [],
            "dialogue": source_input.get("dialogue") or [],
            "memory_items": memories,
            "intents_ranked": intents,
            "reconstruction_metadata": {
                "source_task": row.get("task"),
                "source_session_id": row.get("session_id"),
                "source_revision": ALPSBENCH_REVISION,
                "memory_source": "public_task1_gold",
                "memory_filter_source_sha256": memory_filter.source_sha256,
                "intent_source": "published_alpsbench_intent_prompt_only",
                "intent_workers": intent_workers,
            },
        })
    return output, Task1GoldStats(
        source_rows=source_rows,
        joined_rows=joined,
        public_memory_rows=public_memory,
        memory_filter_pass=memory_pass,
        memory_filter_rejected=public_memory - memory_pass,
        intent_rows_attempted=intent_attempted,
        adapted_records=len(output),
        intent_content_rejections=content_rejections,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--intent-env-prefix", default="PERSONAEMP_MEMORY")
    parser.add_argument("--data-env-prefix", default="PERSONAEMP_DATA")
    parser.add_argument("--big5-env-prefix", default="PERSONAEMP_BIG5")
    parser.add_argument("--adapt-only", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--source-limit", type=int)
    parser.add_argument("--intent-workers", type=int, default=4)
    parser.add_argument("--target-test-users", type=int, default=PAPER_TEST_TARGET_USERS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    official_repo = args.official_repo.resolve()
    verify_official_checkout(official_repo)
    repository_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    download_dir = (args.download_dir or output_dir / "downloads").resolve()
    pairs = download_public_task1(download_dir)
    memory_filter = PublicMemoryFilter(official_repo)
    intent_backend = OpenAICompatibleChatBackend.from_env(args.intent_env_prefix)
    classifier = OfficialIntentReconstructor(
        intent_backend,
        OfficialIntentCache(output_dir / "cache" / "intents_official.jsonl"),
    )
    failures: list[dict[str, str]] = []
    records, stats = build_task1_gold_records(
        pairs,
        classifier,
        memory_filter,
        source_limit=args.source_limit,
        intent_workers=args.intent_workers,
        intent_failures=failures,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    adapted_path = output_dir / "public_task1_gold_by_label.json"
    adapted_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failures_path = output_dir / "intent_failures.jsonl"
    failures_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    final_path: Path | None = None
    pipeline_summary: dict[str, Any] | None = None
    if not args.adapt_only:
        data_backend = OpenAICompatibleChatBackend.from_env(args.data_env_prefix)
        if data_backend.model != PAPER_DATA_MODEL:
            raise ValueError(f"data model must be {PAPER_DATA_MODEL}; got {data_backend.model}")
        final_path = run_official_pipeline(
            official_repo,
            records,
            output_dir,
            env_prefix=args.data_env_prefix,
            python=Path(sys.executable),
            final_filename="English.task1-gold-test-pool.v1.json",
        )
        pipeline_summary = _official_pipeline_summary(output_dir, final_path)
    split_manifest: dict[str, Any] | None = None
    if final_path is not None and not args.skip_splits:
        from .dataset import PersonaEmpDataset
        from .splitting import BigFiveCache, BigFiveLabeler

        big5_backend = OpenAICompatibleChatBackend.from_env(args.big5_env_prefix)
        if big5_backend.model != PAPER_BIG5_MODEL:
            raise ValueError(f"Big Five model must be {PAPER_BIG5_MODEL}; got {big5_backend.model}")
        split_manifest = build_test_only_split_artifacts(
            PersonaEmpDataset.load(final_path),
            output_dir / "test_splits",
            BigFiveLabeler(
                big5_backend,
                BigFiveCache(output_dir / "test_splits" / "cache" / "big_five.jsonl"),
            ),
            target_users=args.target_test_users,
        )
    manifest = {
        "created_at": _utc_now(),
        "protocol": "personaemp_public_task1_gold_test_only_v1",
        "repository_commit": repository_commit,
        "official_commit": "b555447f267b8057039aab39a4be44725718ea7f",
        "alpsbench_revision": ALPSBENCH_REVISION,
        "source_files": [
            {"path": str(path), "sha256": _sha256(path)}
            for pair in pairs
            for path in pair
        ],
        "memory": {
            "source": "public_task1_gold",
            "filter_list_path": str(memory_filter.source_path),
            "filter_list_sha256": memory_filter.source_sha256,
        },
        "intent": {
            "model": intent_backend.model,
            "workers": args.intent_workers,
            "protocol": ALPSBENCH_INTENT_PROTOCOL,
            "upstream_commit": ALPSBENCH_SOURCE_COMMIT,
            "upstream_source_sha256": ALPSBENCH_ORIGINAL_SOURCE_SHA256,
            "system_prompt_sha256": prompt_hash(upstream_adapter.upstream.INTENT_SYSTEM_PROMPT),
            "user_template_sha256": prompt_hash(upstream_adapter.upstream.INTENT_FEW_SHOT_TEMPLATE),
        },
        "stats": asdict(stats),
        "intent_failures": {"path": str(failures_path), "sha256": _sha256(failures_path)},
        "source_limit": args.source_limit,
        "adapted_records": len(records),
        "adapted_path": str(adapted_path),
        "final_dataset": str(final_path) if final_path else None,
        "official_pipeline": pipeline_summary,
        "test_splits": split_manifest,
        "train_artifacts_created": False,
        "table1_direct_comparison_allowed": False,
    }
    (output_dir / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

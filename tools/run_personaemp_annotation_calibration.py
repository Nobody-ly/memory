from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.experiments.personaemp.client import OpenAICompatibleChatBackend
from src.experiments.personaemp.alpsbench_two_stage import (
    AlpsBenchTwoStageExtractor,
)
from src.experiments.personaemp.wildchat_annotation_pool import (
    AnnotationCheckpoint,
    AnnotationPoolExtractor,
    _atomic_json,
    _checkpoint_identity,
    _utc_now,
    _write_jsonl,
    annotate_sample,
    load_intent_taxonomy,
    write_by_label,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object JSONL row in {path}")
                rows.append(value)
    return rows


def _source_record(row: dict[str, Any], source_row: int) -> dict[str, Any]:
    benchmark_id = str(row["benchmark_id"])
    sessions = row["input"]["sessions"]
    if len(sessions) != 1:
        raise ValueError(f"expected one session for {benchmark_id}")
    turns = [
        {
            "role": str(turn["role"]),
            "text": str(turn["text"]).strip(),
        }
        for turn in sessions[0]["turns"]
        if str(turn.get("role") or "") in {"user", "assistant"}
        and str(turn.get("text") or "").strip()
    ]
    source_key = hashlib.sha256(
        json.dumps(
            {"benchmark_id": benchmark_id, "turns": turns},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "source_key": source_key,
        "source_conversation_hash": None,
        "source_file": str(row.get("task") or "task1"),
        "source_row": source_row,
        "session_id": str(row["session_id"]),
        "source_language": None,
        "source_hashed_ip": None,
        "source_timestamp": sessions[0].get("ended_at"),
        "raw_message_count": len(turns),
        "normalized_message_count": len(turns),
        "turns": turns,
        "benchmark_id": benchmark_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate reconstructed annotations on public AlpsBench gold."
    )
    parser.add_argument("--model-input", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--intent-stats", type=Path)
    parser.add_argument(
        "--extractor",
        choices=("alpsbench_official_two_stage", "reconstructed_one_stage"),
        default="alpsbench_official_two_stage",
    )
    parser.add_argument("--env-prefix", default="PERSONAEMP_MEMORY")
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()

    model_rows = _load_jsonl(args.model_input)[: args.limit]
    references = {
        str(row["benchmark_id"]): row for row in _load_jsonl(args.reference_output)
    }
    sample = [_source_record(row, index) for index, row in enumerate(model_rows)]
    missing = [
        row["benchmark_id"]
        for row in model_rows
        if str(row["benchmark_id"]) not in references
    ]
    if missing:
        raise ValueError(f"missing reference rows: {missing}")

    categories, subtypes = load_intent_taxonomy(args.intent_stats)
    backend = OpenAICompatibleChatBackend.from_env(args.env_prefix)
    if args.extractor == "alpsbench_official_two_stage":
        extractor: Any = AlpsBenchTwoStageExtractor(backend)
    else:
        extractor = AnnotationPoolExtractor(backend, categories, subtypes)
    checkpoint_identity = _checkpoint_identity(sample, extractor)
    checkpoint = AnnotationCheckpoint(
        args.output_dir / "cache" / "annotation_successes.jsonl",
        args.output_dir / "cache" / "annotation_identity.json",
        checkpoint_identity,
    )
    records, failures, cached = annotate_sample(sample, extractor, checkpoint)
    by_source_key = {
        str(row["reconstruction_metadata"]["source_key"]): row for row in records
    }
    comparisons = []
    for source in sample:
        predicted = by_source_key.get(str(source["source_key"]))
        if predicted is None:
            continue
        comparisons.append(
            {
                "benchmark_id": source["benchmark_id"],
                "predicted_memory_items": predicted["memory_items"],
                "gold_memory_items": references[source["benchmark_id"]]["gold"][
                    "memory_items"
                ],
            }
        )

    _write_jsonl(args.output_dir / "stages" / "source_sample.jsonl", sample)
    _write_jsonl(args.output_dir / "stages" / "annotation_records.jsonl", records)
    _write_jsonl(args.output_dir / "stages" / "annotation_failures.jsonl", failures)
    _write_jsonl(args.output_dir / "quality" / "gold_comparisons.jsonl", comparisons)
    buckets = write_by_label(args.output_dir, records)
    predicted_memories = sum(len(row["memory_items"]) for row in records)
    gold_memories = sum(len(row["gold"]["memory_items"]) for row in references.values() if row["benchmark_id"] in {source["benchmark_id"] for source in sample})
    manifest = {
        "created_at": _utc_now(),
        "protocol": (
            f"{checkpoint_identity['protocol']}_public_gold_calibration_v1"
        ),
        "implementation_commit": args.implementation_commit,
        "checkpoint_identity": checkpoint_identity,
        "model_input": {
            "path": str(args.model_input.resolve()),
            "sha256": _sha256(args.model_input),
        },
        "reference_output": {
            "path": str(args.reference_output.resolve()),
            "sha256": _sha256(args.reference_output),
        },
        "model": backend.model,
        "extractor": args.extractor,
        "attempted": len(sample),
        "succeeded": len(records),
        "failed": len(failures),
        "loaded_from_checkpoint": cached,
        "predicted_memory_items": predicted_memories,
        "gold_memory_items": gold_memories,
        "by_label_files": len(buckets),
        "automated_semantic_score": None,
        "manual_verification": False,
        "table1_direct_comparison_allowed": False,
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

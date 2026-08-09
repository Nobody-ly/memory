from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.experiments.personaemp.generation import prompt_hash
from src.experiments.personaemp.wildchat_annotation_pool import (
    AnnotationCheckpoint,
    _atomic_json,
    _checkpoint_identity,
    _load_jsonl,
    _utc_now,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _IdentityOnlyExtractor:
    def __init__(self, identity: dict[str, Any]) -> None:
        self.identity = identity

    def run_identity(self) -> dict[str, Any]:
        return dict(self.identity)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed a larger annotation checkpoint from a nested run."
    )
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--target-output", type=Path, required=True)
    parser.add_argument("--target-sample", type=Path, required=True)
    args = parser.parse_args()

    source_cache = args.source_output / "cache" / "annotation_successes.jsonl"
    source_identity_path = args.source_output / "cache" / "annotation_identity.json"
    if not source_cache.is_file() or not source_identity_path.is_file():
        raise FileNotFoundError("source checkpoint or identity is missing")

    source_identity = json.loads(source_identity_path.read_text(encoding="utf-8"))
    run_identity = {
        key: value
        for key, value in source_identity.items()
        if key not in {"source_keys_sha256", "source_count"}
    }
    target_sample = _load_jsonl(args.target_sample)
    target_by_key = {str(row["source_key"]): row for row in target_sample}
    if len(target_by_key) != len(target_sample):
        raise RuntimeError("target sample contains duplicate source keys")

    target_identity = _checkpoint_identity(
        target_sample, _IdentityOnlyExtractor(run_identity)
    )
    checkpoint = AnnotationCheckpoint(
        args.target_output / "cache" / "annotation_successes.jsonl",
        args.target_output / "cache" / "annotation_identity.json",
        target_identity,
    )

    seeded = 0
    seen_source_keys: set[str] = set()
    for row in _load_jsonl(source_cache):
        source_key = str(row.get("source_key") or "")
        record = row.get("record")
        cache_key = str(row.get("cache_key") or "")
        if (
            row.get("status") != "success"
            or not source_key
            or not cache_key
            or not isinstance(record, dict)
        ):
            raise RuntimeError("source checkpoint contains an invalid row")
        if source_key not in target_by_key:
            raise RuntimeError(f"source key is not nested in target sample: {source_key}")
        if source_key in seen_source_keys:
            raise RuntimeError(f"duplicate source key in source checkpoint: {source_key}")

        metadata = record.get("reconstruction_metadata") or {}
        if metadata.get("source_key") != source_key:
            raise RuntimeError(f"record provenance mismatch for {source_key}")
        cache_identity = {
            "source_key": source_key,
            **run_identity,
            "conversation_sha256": metadata.get("conversation_sha256"),
        }
        expected_cache_key = hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if cache_key != expected_cache_key:
            raise RuntimeError(f"cache key mismatch for {source_key}")

        checkpoint.save(cache_key, source_key, record)
        seen_source_keys.add(source_key)
        seeded += 1

    manifest = {
        "created_at": _utc_now(),
        "protocol": "personaemp_annotation_checkpoint_seed_v1",
        "source_output": str(args.source_output.resolve()),
        "source_checkpoint_sha256": _sha256(source_cache),
        "source_identity_sha256": prompt_hash(
            json.dumps(source_identity, sort_keys=True)
        ),
        "target_sample": str(args.target_sample.resolve()),
        "target_sample_sha256": _sha256(args.target_sample),
        "target_identity": target_identity,
        "seeded_records": seeded,
        "target_records": len(target_sample),
    }
    _atomic_json(args.target_output / "quality" / "checkpoint_seed.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

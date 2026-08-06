from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


PROTOCOL_VERSION = "personaemp_official_chunk_resume_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_official_query(prepare_dir: Path) -> Any:
    sys.path.insert(0, str(prepare_dir))
    module_path = prepare_dir / "query.py"
    spec = importlib.util.spec_from_file_location(
        "personaemp_official_query_resumable",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official query pipeline: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._require_api_config()
    return module


def _chunk_path(directory: Path, start: int, end: int) -> Path:
    return directory / f"chunk_{start:06d}_{end:06d}.json"


def _validate_chunk(
    payload: dict[str, Any],
    *,
    stage: str,
    start: int,
    end: int,
    source_sha256: str,
    model: str,
) -> None:
    expected = {
        "protocol": PROTOCOL_VERSION,
        "stage": stage,
        "start": start,
        "end": end,
        "source_sha256": source_sha256,
        "model": model,
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "resume chunk belongs to a different reconstruction run: "
            f"expected={expected}, actual={actual}"
        )


def _run_generation(
    module: Any,
    *,
    total_records: int,
    chunk_size: int,
    resume_dir: Path,
    source_sha256: str,
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    chunks_dir = resume_dir / "generation"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    all_generated: list[dict[str, Any]] = []
    all_usage: list[dict[str, Any]] = []
    all_debug: list[dict[str, Any]] = []

    for start in range(0, total_records, chunk_size):
        end = min(start + chunk_size, total_records)
        chunk_path = _chunk_path(chunks_dir, start, end)
        if chunk_path.is_file():
            payload = _load_json(chunk_path, {})
            _validate_chunk(
                payload,
                stage="generation",
                start=start,
                end=end,
                source_sha256=source_sha256,
                model=model,
            )
        else:
            generated, usage = module.query_generation(
                start_index=start,
                end_index=end,
                batch_size=chunk_size,
            )
            debug = _load_json(Path(module.STAGE_DEBUG_FILE), [])
            payload = {
                "protocol": PROTOCOL_VERSION,
                "stage": "generation",
                "start": start,
                "end": end,
                "source_sha256": source_sha256,
                "model": model,
                "generated": generated,
                "usage": usage,
                "stage_debug": debug,
            }
            _atomic_json(chunk_path, payload)
        all_generated.extend(payload.get("generated") or [])
        all_usage.extend(payload.get("usage") or [])
        all_debug.extend(payload.get("stage_debug") or [])

    return all_generated, all_usage, all_debug


def _run_inspection(
    module: Any,
    *,
    generated: list[dict[str, Any]],
    generation_usage: list[dict[str, Any]],
    chunk_size: int,
    resume_dir: Path,
    source_sha256: str,
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks_dir = resume_dir / "inspection"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    all_inspected: list[dict[str, Any]] = []
    all_usage: list[dict[str, Any]] = []

    for start in range(0, len(generated), chunk_size):
        end = min(start + chunk_size, len(generated))
        chunk_path = _chunk_path(chunks_dir, start, end)
        if chunk_path.is_file():
            payload = _load_json(chunk_path, {})
            _validate_chunk(
                payload,
                stage="inspection",
                start=start,
                end=end,
                source_sha256=source_sha256,
                model=model,
            )
        else:
            generated_chunk = generated[start:end]
            generation_usage_chunk = [
                {
                    "stage": "generation",
                    "source_file": row.get("source_file", "unknown"),
                    "record_id": row.get("record", {}).get(
                        "line_index", start + offset
                    ),
                    "usage": row.get("generation_usage"),
                }
                for offset, row in enumerate(generated_chunk)
            ]
            inspected, usage_summary = module.query_inspection(
                generated_chunk,
                generation_usage_chunk,
            )
            payload = {
                "protocol": PROTOCOL_VERSION,
                "stage": "inspection",
                "start": start,
                "end": end,
                "source_sha256": source_sha256,
                "model": model,
                "inspected": inspected,
                "usage": usage_summary.get("inspection_calls") or [],
            }
            _atomic_json(chunk_path, payload)
        all_inspected.extend(payload.get("inspected") or [])
        all_usage.extend(payload.get("usage") or [])

    return all_inspected, all_usage


def run(
    prepare_dir: Path,
    *,
    generation_chunk_size: int,
    inspection_chunk_size: int,
) -> dict[str, Any]:
    if generation_chunk_size < 1 or inspection_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    prepare_dir = prepare_dir.resolve()
    os.chdir(prepare_dir)
    module = _load_official_query(prepare_dir)
    raw_filtered = Path(module.FILE).resolve()
    records = module.load_records(str(raw_filtered))
    if not records:
        raise RuntimeError("official raw_filtered.json contains no records")

    source_sha256 = _sha256(raw_filtered)
    model = str(module.MODEL_NAME)
    resume_dir = raw_filtered.parent / "resumable_pipeline"
    manifest_path = resume_dir / "manifest.json"
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "source_sha256": source_sha256,
        "model": model,
        "total_records": len(records),
        "generation_chunk_size": generation_chunk_size,
        "inspection_chunk_size": inspection_chunk_size,
    }
    if manifest_path.is_file():
        existing = _load_json(manifest_path, {})
        if existing != manifest:
            raise RuntimeError(
                "resume directory belongs to a different source, model, or chunk layout"
            )
    else:
        _atomic_json(manifest_path, manifest)

    generated, generation_usage, stage_debug = _run_generation(
        module,
        total_records=len(records),
        chunk_size=generation_chunk_size,
        resume_dir=resume_dir,
        source_sha256=source_sha256,
        model=model,
    )
    _atomic_json(Path(module.GENERATED_FILE), generated)
    _atomic_json(Path(module.STAGE_DEBUG_FILE), stage_debug)

    inspected, inspection_usage = _run_inspection(
        module,
        generated=generated,
        generation_usage=generation_usage,
        chunk_size=inspection_chunk_size,
        resume_dir=resume_dir,
        source_sha256=source_sha256,
        model=model,
    )
    _atomic_json(Path(module.INSPECTION_FILE), inspected)
    usage_summary = module._build_usage_summary(
        generation_usage,
        inspection_usage,
    )
    _atomic_json(Path(module.USAGE_SUMMARY_FILE), usage_summary)

    (g1, g2, g3), total, filtered_count, filtered = module.final_filter()
    module.store_in_language(filtered, module.FINAL)
    completion = {
        **manifest,
        "generated_records": len(generated),
        "inspected_records": len(inspected),
        "inspection_candidates": total,
        "filtered_queries": filtered_count,
        "category_counts": {
            "high_eq_interaction": g1,
            "emotional_support": g2,
            "social_strategy": g3,
        },
        "completed": True,
    }
    _atomic_json(resume_dir / "completion.json", completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PersonaEmp's official query pipeline with chunk-level resume "
            "without changing its prompts or per-record logic."
        )
    )
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--generation-chunk-size", type=int, default=25)
    parser.add_argument("--inspection-chunk-size", type=int, default=10)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run(
        args.prepare_dir,
        generation_chunk_size=args.generation_chunk_size,
        inspection_chunk_size=args.inspection_chunk_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

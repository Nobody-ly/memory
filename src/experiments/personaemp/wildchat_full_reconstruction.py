"""Run the public WildChat reconstruction through PersonaEmp's released stages.

The original AlpsBench memory annotation input is not public. This entrypoint
therefore reconstructs that missing upstream stage from the fixed WildChat
revision, then runs the released PersonaEmp code unchanged for filtering,
persona construction, situation/query construction, inspection, and language
splitting. It is deliberately labelled as a public-data reconstruction rather
than an author-identical dataset release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .client import OpenAICompatibleChatBackend
from .reconstruction import (
    OFFICIAL_COMMIT,
    _official_pipeline_summary,
    run_official_pipeline,
    verify_official_checkout,
)
from .wildchat_reconstruction import PAPER_MEMORY_MODEL


PAPER_DATA_MODEL = "MiniMax-M2.5"
PROTOCOL = "personaemp_wildchat_full_reconstruction_v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_memory_reconstruction(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "src.experiments.personaemp.wildchat_reconstruction",
        "--output-dir",
        str(output_dir),
        "--env-prefix",
        args.memory_env_prefix,
        "--category-cap",
        str(args.category_cap),
        "--dedup-encoder",
        args.dedup_encoder,
        "--dedup-threshold",
        str(args.dedup_threshold),
    ]
    if args.snapshot_dir:
        command.extend(["--snapshot-dir", str(args.snapshot_dir.resolve())])
    else:
        command.append("--download")
        for pattern in args.download_pattern:
            command.extend(["--download-pattern", pattern])
    if args.source_limit is not None:
        command.extend(["--source-limit", str(args.source_limit)])
    if args.enable_reconstructed_semantic_dedup:
        command.append("--enable-reconstructed-semantic-dedup")
    if bool(args.gold_input) != bool(args.gold_reference):
        raise ValueError("--gold-input and --gold-reference must be supplied together")
    if args.gold_input and args.gold_reference:
        command.extend(
            [
                "--gold-input",
                str(args.gold_input.resolve()),
                "--gold-reference",
                str(args.gold_reference.resolve()),
                "--gold-limit",
                str(args.gold_limit),
            ]
        )
    subprocess.run(command, check=True)
    manifest_path = output_dir / "wildchat_reconstruction_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("WildChat reconstruction did not write its manifest")
    manifest = _load_json(manifest_path)
    stats = manifest.get("memory_extraction_stats") or {}
    if int(stats.get("failed") or 0) != 0:
        raise RuntimeError(
            "memory reconstruction has unresolved extraction failures; "
            "resume the same run after the provider recovers"
        )
    if not manifest.get("memory_extraction_complete"):
        raise RuntimeError("memory reconstruction did not complete")
    return manifest


def _read_curated_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts") or {}
    path = Path(str(artifacts.get("by_label_json") or ""))
    if not path.is_file():
        raise RuntimeError(f"missing curated WildChat records: {path}")
    records = _load_json(path)
    if not isinstance(records, list) or not records:
        raise RuntimeError("WildChat curation produced no records for PersonaEmp")
    if not all(isinstance(record, dict) for record in records):
        raise RuntimeError("curated WildChat records are not all JSON objects")
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild PersonaEmp-style data from fixed WildChat, then execute the "
            "released PersonaEmp dataset-construction stages."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--download-pattern", action="append", default=[])
    parser.add_argument("--source-limit", type=int)
    parser.add_argument("--category-cap", type=int, default=350)
    parser.add_argument("--dedup-encoder", default="intfloat/e5-base-v2")
    parser.add_argument("--dedup-threshold", type=float, default=0.92)
    parser.add_argument(
        "--enable-reconstructed-semantic-dedup",
        action="store_true",
    )
    parser.add_argument("--gold-input", type=Path)
    parser.add_argument("--gold-reference", type=Path)
    parser.add_argument("--gold-limit", type=int, default=12)
    parser.add_argument("--memory-env-prefix", default="PERSONAEMP_MEMORY")
    parser.add_argument("--data-env-prefix", default="PERSONAEMP_DATA")
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Stop after memory extraction, curation, and quality artifacts.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    official_repo = args.official_repo.resolve()
    verify_official_checkout(official_repo)
    memory_manifest = _run_memory_reconstruction(args, output_dir)
    curated_records = _read_curated_records(memory_manifest)

    full_manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "official_commit": OFFICIAL_COMMIT,
        "paper_defined": {
            "memory_model": PAPER_MEMORY_MODEL,
            "dataset_construction_model": PAPER_DATA_MODEL,
        },
        "wildchat_memory_manifest": str(
            output_dir / "wildchat_reconstruction_manifest.json"
        ),
        "curated_records": len(curated_records),
        "table1_direct_comparison_allowed": False,
    }
    if args.memory_only:
        full_manifest["status"] = "memory_stage_complete"
        _atomic_json(output_dir / "full_reconstruction_manifest.json", full_manifest)
        print(json.dumps(full_manifest, ensure_ascii=False, indent=2))
        return 0

    data_backend = OpenAICompatibleChatBackend.from_env(args.data_env_prefix)
    if data_backend.model != PAPER_DATA_MODEL:
        raise ValueError(
            "PersonaEmp data construction requires "
            f"{PAPER_DATA_MODEL}; got {data_backend.model}"
        )
    final_dataset = run_official_pipeline(
        official_repo,
        curated_records,
        output_dir,
        env_prefix=args.data_env_prefix,
        python=Path(sys.executable),
        input_filename="wildchat_reconstruction.json",
        final_filename="English.wildchat-reconstruction.v1.json",
    )
    full_manifest.update(
        {
            "status": "complete",
            "official_pipeline": _official_pipeline_summary(output_dir, final_dataset),
            "final_dataset": str(final_dataset),
        }
    )
    _atomic_json(output_dir / "full_reconstruction_manifest.json", full_manifest)
    print(json.dumps(full_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

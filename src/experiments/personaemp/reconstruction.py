from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve

from .client import ChatBackend, OpenAICompatibleChatBackend
from .generation import _parse_json_object, prompt_hash


OFFICIAL_COMMIT = "b555447f267b8057039aab39a4be44725718ea7f"
ALPSBENCH_REVISION = "dcd1648"
ALPSBENCH_BASE = (
    "https://huggingface.co/datasets/Cosineyx/Alpsbench/resolve/"
    f"{ALPSBENCH_REVISION}/dataset"
)
INTENT_ALLOWLIST = (
    "Learning Support",
    "Conversational Engagement",
    "Personal Advice",
    "Decision Support",
    "Business",
    "Career Advice",
    "Moral and Ethical Queries",
    "Reflection and Insight",
    "Existential Questions",
    "Societal and Cultural Inquiry",
    "Philosophical",
)
INTENT_SYSTEM_PROMPT = """You classify long-term user conversations for a
personalized-empathy dataset. Select every matching intent from the supplied
allowlist. Select Other only when none applies. Base the decision only on the
conversation. Return the requested JSON and no commentary."""
INTENT_USER_TEMPLATE = """Intent allowlist:
{allowlist}

Conversation:
{conversation}
"""
INTENT_SCHEMA = {
    "name": "personaemp_intent_gate",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "intents": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [*INTENT_ALLOWLIST, "Other"],
                },
                "uniqueItems": True,
            }
        },
        "required": ["intents"],
        "additionalProperties": False,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"record at {path}:{line_number} is not an object")
            records.append(value)
    return records


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_count(records: Any, field: str = "queries") -> int:
    if not isinstance(records, list):
        return 0
    return sum(
        len(record.get(field) or [])
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get(field) or [], list)
    )


def _official_pipeline_summary(output_dir: Path, final_path: Path) -> dict[str, Any]:
    trainset = (
        output_dir
        / "official_pipeline_worktree"
        / "prepare_dataset"
        / "dataset"
        / "ei_trainset"
    )
    files = {
        "raw_filtered": trainset / "raw_filtered.json",
        "generated": trainset / "generated_queries.json",
        "inspected": trainset / "inspection_results.json",
        "stage_debug": trainset / "generation_stage_debug.json",
        "usage": trainset / "query_usage_summary.json",
        "model_compatibility": output_dir / "model_compatibility.json",
        "final": final_path,
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "official pipeline is missing artifacts: " + ", ".join(missing)
        )
    raw_filtered = _load_json(files["raw_filtered"])
    generated = _load_json(files["generated"])
    inspected = _load_json(files["inspected"])
    final = _load_json(files["final"])
    stage_debug = _load_json(files["stage_debug"])
    categories = Counter(
        canonical_category(query.get("category"))
        for record in final
        if isinstance(record, dict)
        for query in record.get("queries") or []
        if isinstance(query, dict)
    )
    skip_reasons = Counter()
    if isinstance(stage_debug, list):
        for row in stage_debug:
            stages = row.get("stage_debug") if isinstance(row, dict) else None
            if not isinstance(stages, dict):
                continue
            for stage in stages.values():
                reason = (
                    str(stage.get("skip_reason") or "").strip()
                    if isinstance(stage, dict)
                    else ""
                )
                if reason:
                    skip_reasons[reason] += 1
    query_counts = [
        len(record.get("queries") or [])
        for record in final
        if isinstance(record, dict)
    ]
    return {
        "raw_filtered_records": len(raw_filtered),
        "generated_records": len(generated),
        "generated_queries": _query_count(generated),
        "inspected_records": len(inspected),
        "inspected_queries": _query_count(inspected, "queries_inspected"),
        "final_users": len(final),
        "final_queries": _query_count(final),
        "category_distribution": dict(sorted(categories.items())),
        "queries_per_user": {
            "minimum": min(query_counts) if query_counts else 0,
            "maximum": max(query_counts) if query_counts else 0,
            "mean": (
                round(sum(query_counts) / len(query_counts), 6)
                if query_counts
                else 0
            ),
        },
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in files.items()
        },
    }


def canonical_category(value: Any) -> str:
    text = str(value or "unknown").strip()
    aliases = {
        "hei": "HEQ",
        "heq": "HEQ",
        "high-eq interaction": "HEQ",
        "emotional support": "ES",
        "es": "ES",
        "social strategy": "SS",
        "ss": "SS",
    }
    return aliases.get(text.lower(), text)


def _conversation_text(record: dict[str, Any]) -> str:
    dialogue = (record.get("input") or {}).get("dialogue") or []
    return "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('text', '')}"
        for turn in dialogue
        if isinstance(turn, dict) and str(turn.get("text") or "").strip()
    )


@dataclass(frozen=True)
class ReconstructionStats:
    input_rows: int
    joined_rows: int
    missing_references: int
    memory_items: int


class IntentCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.values: dict[str, list[str]] = {}
        if path.is_file():
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    cache_key = str(value.get("cache_key") or "")
                    if cache_key:
                        self.values[cache_key] = list(value["intents"])

    def save(
        self,
        cache_key: str,
        benchmark_id: str,
        intents: list[str],
        provenance: dict[str, str],
    ) -> None:
        if cache_key in self.values:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "cache_key": cache_key,
            "benchmark_id": benchmark_id,
            "intents": intents,
            "provenance": provenance,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as destination:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
        self.values[cache_key] = intents


class IntentReconstructor:
    def __init__(self, backend: ChatBackend, cache: IntentCache) -> None:
        self.backend = backend
        self.cache = cache

    def _cache_identity(
        self,
        record: dict[str, Any],
    ) -> tuple[str, dict[str, str]]:
        provenance = {
            "model": self.backend.model,
            "system_prompt_sha256": prompt_hash(INTENT_SYSTEM_PROMPT),
            "user_template_sha256": prompt_hash(INTENT_USER_TEMPLATE),
            "schema_sha256": prompt_hash(
                json.dumps(INTENT_SCHEMA, sort_keys=True)
            ),
            "conversation_sha256": prompt_hash(_conversation_text(record)),
        }
        value = {
            "benchmark_id": str(record.get("benchmark_id") or ""),
            **provenance,
        }
        cache_key = hashlib.sha256(
            json.dumps(value, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cache_key, provenance

    def classify(self, record: dict[str, Any]) -> list[str]:
        benchmark_id = str(record.get("benchmark_id") or "")
        cache_key, provenance = self._cache_identity(record)
        if cache_key in self.cache.values:
            return self.cache.values[cache_key]
        result = self.backend.chat(
            INTENT_SYSTEM_PROMPT,
            INTENT_USER_TEMPLATE.format(
                allowlist="\n".join(f"- {value}" for value in INTENT_ALLOWLIST),
                conversation=_conversation_text(record),
            ),
            temperature=0.0,
            max_tokens=300,
            response_schema=INTENT_SCHEMA,
        )
        parsed = _parse_json_object(result.content)
        raw_intents = parsed.get("intents")
        if not isinstance(raw_intents, list) or not raw_intents:
            raise ValueError("intent classifier returned no intents")
        allowed = {*INTENT_ALLOWLIST, "Other"}
        intents = [str(value) for value in raw_intents]
        if any(value not in allowed for value in intents):
            raise ValueError("intent classifier returned an unknown intent")
        self.cache.save(
            cache_key,
            benchmark_id,
            intents,
            provenance,
        )
        return intents


def adapt_alpsbench(
    pairs: Iterable[tuple[Path, Path]],
    classifier: IntentReconstructor,
    *,
    source_limit: int | None = None,
) -> tuple[list[dict[str, Any]], ReconstructionStats]:
    if source_limit is not None and source_limit <= 0:
        raise ValueError("source_limit must be a positive integer")
    output: list[dict[str, Any]] = []
    input_rows = joined_rows = missing_references = memory_items = 0
    seen: set[str] = set()
    for input_path, reference_path in pairs:
        references = {
            str(row.get("benchmark_id")): row
            for row in _load_jsonl(reference_path)
        }
        for row in _load_jsonl(input_path):
            if source_limit is not None and input_rows >= source_limit:
                break
            input_rows += 1
            benchmark_id = str(row.get("benchmark_id") or "")
            if not benchmark_id or benchmark_id in seen:
                raise ValueError(f"invalid or duplicate benchmark_id: {benchmark_id!r}")
            seen.add(benchmark_id)
            reference = references.get(benchmark_id)
            if reference is None:
                missing_references += 1
                continue
            gold = reference.get("gold") or {}
            memories = gold.get("memory_items") or []
            if not isinstance(memories, list) or not memories:
                raise ValueError(f"{benchmark_id} has no public gold memories")
            source_input = row.get("input") or {}
            sessions = source_input.get("sessions") or []
            dialogue = source_input.get("dialogue") or []
            intents = classifier.classify(row)
            output.append(
                {
                    "benchmark_id": benchmark_id,
                    "line_index": source_input.get("line_index"),
                    "sessions": sessions,
                    "dialogue": dialogue,
                    "memory_items": memories,
                    "intents_ranked": [
                        {
                            "intent_category": value,
                            "intent_subtype": "",
                        }
                        for value in intents
                        if value != "Other"
                    ],
                    "reconstruction_metadata": {
                        "source_task": row.get("task"),
                        "source_session_id": row.get("session_id"),
                        "source_revision": ALPSBENCH_REVISION,
                    },
                }
            )
            joined_rows += 1
            memory_items += len(memories)
        if source_limit is not None and input_rows >= source_limit:
            break
    return output, ReconstructionStats(
        input_rows=input_rows,
        joined_rows=joined_rows,
        missing_references=missing_references,
        memory_items=memory_items,
    )


def download_public_task1(directory: Path) -> list[tuple[Path, Path]]:
    directory.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []
    for split in ("dev", "validation"):
        input_path = directory / f"{split}_model_input.jsonl"
        reference_path = directory / f"{split}_reference_output.jsonl"
        for remote_name, local_path in (
            ("model_input.jsonl", input_path),
            ("reference_output.jsonl", reference_path),
        ):
            if not local_path.is_file():
                urlretrieve(
                    f"{ALPSBENCH_BASE}/{split}/task1/{remote_name}",
                    local_path,
                )
        pairs.append((input_path, reference_path))
    return pairs


def verify_official_checkout(repository: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if head != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"official checkout must be pinned to {OFFICIAL_COMMIT}, got {head}"
        )


def apply_official_model_compatibility(
    prepare_dir: Path,
    *,
    model: str,
    output_dir: Path,
) -> None:
    """Patch only transport parameters in the isolated official worktree."""
    api_client = prepare_dir / "api_call.py"
    source = api_client.read_text(encoding="utf-8")
    marker = "# PERSONAEMP_KIMI_NONTHINKING_COMPAT"
    applied = False
    if model.startswith(("kimi-k2.5", "kimi-k2.6")) and marker not in source:
        original = """                        temperature=temperature,
                        # top_p=top_p,
"""
        replacement = """                        # PERSONAEMP_KIMI_NONTHINKING_COMPAT
                        **(
                            {
                                "temperature": 0.6,
                                "extra_body": {
                                    "thinking": {"type": "disabled"}
                                },
                            }
                            if model_name.startswith(("kimi-k2.5", "kimi-k2.6"))
                            else {"temperature": temperature}
                        ),
                        # top_p=top_p,
"""
        if original not in source:
            raise RuntimeError(
                "cannot apply Kimi compatibility patch to official api_call.py"
            )
        api_client.write_text(
            source.replace(original, replacement, 1),
            encoding="utf-8",
        )
        applied = True
    compatibility = {
        "model": model,
        "transport_only": True,
        "prompt_or_protocol_changed": False,
        "kimi_nonthinking_patch_applied_this_run": applied,
        "kimi_nonthinking_patch_active": marker
        in api_client.read_text(encoding="utf-8"),
        "patched_file": str(api_client),
        "patched_file_sha256": _sha256(api_client),
    }
    (output_dir / "model_compatibility.json").write_text(
        json.dumps(compatibility, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_official_pipeline(
    repository: Path,
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    env_prefix: str,
    python: Path,
) -> Path:
    repository = repository.resolve()
    output_dir = output_dir.resolve()
    python = python.resolve()
    verify_official_checkout(repository)
    scratch = output_dir / "official_pipeline_worktree"
    archive = output_dir / "official_pipeline.tar"
    if not scratch.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                f"--output={archive}",
                "HEAD",
            ],
            check=True,
        )
        scratch.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as source:
            source.extractall(scratch, filter="data")
        archive.unlink()
    prepare_dir = scratch / "prepare_dataset"
    dataset_dir = prepare_dir / "dataset"
    by_label_dir = dataset_dir / "by_label_json"
    trainset_dir = dataset_dir / "ei_trainset"
    final_dir = trainset_dir / "final_data"
    by_label_dir.mkdir(parents=True, exist_ok=True)
    trainset_dir.mkdir(parents=True, exist_ok=True)
    input_path = by_label_dir / "public_reconstruction.json"
    input_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    subprocess.run([str(python), "filter.py"], cwd=prepare_dir, check=True)

    def env(name: str, fallback: str = "") -> str:
        return os.getenv(f"{env_prefix}_{name}", fallback).strip()

    api_key = env("API_KEY", os.getenv("API_KEY", ""))
    base_url = env("BASE_URL", os.getenv("BASE_URL", ""))
    model = env("MODEL", "kimi-k2.6")
    if not api_key or not base_url:
        raise RuntimeError(
            f"set {env_prefix}_API_KEY and {env_prefix}_BASE_URL"
        )
    apply_official_model_compatibility(
        prepare_dir,
        model=model,
        output_dir=output_dir,
    )
    process_env = dict(os.environ)
    process_env.update(
        {
            "DATA_API_KEYS": api_key,
            "DATA_BASE_URLS": base_url,
            "DATA_MODEL_NAME": model,
            "DATA_RUN_MODE": "full_pipeline",
        }
    )
    resumable_runner = Path(__file__).with_name(
        "resumable_official_pipeline.py"
    )
    subprocess.run(
        [
            str(python),
            str(resumable_runner),
            "--prepare-dir",
            str(prepare_dir),
        ],
        cwd=prepare_dir,
        env=process_env,
        check=True,
    )
    english = final_dir / "English.json"
    if not english.is_file():
        raise RuntimeError("official pipeline did not produce English.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "English.public-reconstruction.v1.json"
    shutil.copy2(english, destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct PersonaEmp inputs from public AlpsBench Task 1."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--env-prefix", default="PERSONAEMP_GENERATOR")
    parser.add_argument("--adapt-only", action="store_true")
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help=(
            "Deterministically process only the first N public source rows. "
            "Intended for protocol-faithful pilot runs; omit for the full 932."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    download_dir = (
        args.download_dir.resolve()
        if args.download_dir
        else output_dir / "downloads"
    )
    pairs = download_public_task1(download_dir)
    backend = OpenAICompatibleChatBackend.from_env(args.env_prefix)
    classifier = IntentReconstructor(
        backend,
        IntentCache(output_dir / "cache" / "intents.jsonl"),
    )
    records, stats = adapt_alpsbench(
        pairs,
        classifier,
        source_limit=args.source_limit,
    )
    adapted_path = output_dir / "by_label_json" / "public_reconstruction.json"
    adapted_path.parent.mkdir(parents=True, exist_ok=True)
    adapted_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_path: Path | None = None
    if not args.adapt_only:
        final_path = run_official_pipeline(
            args.official_repo.resolve(),
            records,
            output_dir,
            env_prefix=args.env_prefix,
            python=Path(sys.executable),
        )
    pipeline_summary = (
        _official_pipeline_summary(output_dir, final_path)
        if final_path is not None
        else None
    )
    manifest = {
        "created_at": _utc_now(),
        "protocol": "personaemp_public_reconstruction_v1",
        "official_commit": OFFICIAL_COMMIT,
        "alpsbench_revision": ALPSBENCH_REVISION,
        "source_files": [
            {"path": str(path), "sha256": _sha256(path)}
            for pair in pairs
            for path in pair
        ],
        "intent": {
            "model": backend.model,
            "system_prompt_sha256": prompt_hash(INTENT_SYSTEM_PROMPT),
            "schema_sha256": prompt_hash(
                json.dumps(INTENT_SCHEMA, sort_keys=True)
            ),
        },
        "stats": asdict(stats),
        "source_limit": args.source_limit,
        "adapted_records": len(records),
        "adapted_path": str(adapted_path),
        "final_dataset": str(final_path) if final_path else None,
        "official_pipeline": pipeline_summary,
        "table1_direct_comparison_allowed": False,
    }
    (output_dir / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .dataset import PersonaEmpDataset, PersonaEmpSample
from .generation import (
    BASE_MODEL_USER_PROMPT,
    ALIGNMENT_RESPONSE_SCHEMA,
    ALIGNMENT_MAX_TOKENS,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
    MEMORY_RESPONSE_USER_PROMPT,
    MEMORY_SUMMARY_SYSTEM_PROMPT,
    MEMORY_SUMMARY_USER_PROMPT,
    OURS_USER_PROMPT,
    PERSONAEMP_AGENT_PERSONA_DISABLED,
    PERSONAEMP_ALIGNMENT_SYSTEM_PROMPT,
    PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
    PROFILE_MAX_TOKENS,
    PROFILE_RESPONSE_SCHEMA,
    RAG_RESPONSE_USER_PROMPT,
    RAG_ENCODER_MODEL,
    RAG_ENCODER_REVISION,
    STRUCTURED_JSON_PARSER_VERSION,
    BaseModelGenerator,
    DeepEmpathyGenerator,
    MemoryGenerator,
    RAGGenerator,
    prompt_hash,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_value(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sample_id(
    dataset_fingerprint: str,
    sample: PersonaEmpSample,
    method: str,
) -> str:
    value = (
        f"{dataset_fingerprint}\0{sample.session_id}\0"
        f"{sample.query_id}\0{method}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _generation_prompt_hashes() -> dict[str, str]:
    return {
        "shared_response_system": prompt_hash(
            PERSONAEMP_RESPONSE_SYSTEM_PROMPT
        ),
        "disabled_agent_persona_input": prompt_hash(
            json.dumps(
                PERSONAEMP_AGENT_PERSONA_DISABLED,
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
        "ours_user_template": prompt_hash(OURS_USER_PROMPT),
        "base_user_template": prompt_hash(BASE_MODEL_USER_PROMPT),
        "memory_summary_system": prompt_hash(MEMORY_SUMMARY_SYSTEM_PROMPT),
        "memory_summary_user": prompt_hash(MEMORY_SUMMARY_USER_PROMPT),
        "memory_response_user": prompt_hash(MEMORY_RESPONSE_USER_PROMPT),
        "rag_response_user": prompt_hash(RAG_RESPONSE_USER_PROMPT),
        "profile_extraction_system": prompt_hash(
            PROFILE_EXTRACTION_SYSTEM_PROMPT
        ),
        "profile_extraction_user_template": prompt_hash(
            PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
        ),
        "profile_response_schema": prompt_hash(
            json.dumps(PROFILE_RESPONSE_SCHEMA, sort_keys=True)
        ),
        "profile_max_tokens": prompt_hash(str(PROFILE_MAX_TOKENS)),
        "empathy_alignment_system": prompt_hash(
            PERSONAEMP_ALIGNMENT_SYSTEM_PROMPT
        ),
        "empathy_alignment_user_template": prompt_hash(
            EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE
        ),
        "empathy_alignment_response_schema": prompt_hash(
            json.dumps(ALIGNMENT_RESPONSE_SCHEMA, sort_keys=True)
        ),
        "empathy_alignment_max_tokens": prompt_hash(
            str(ALIGNMENT_MAX_TOKENS)
        ),
        "structured_json_parser": prompt_hash(
            STRUCTURED_JSON_PARSER_VERSION
        ),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid checkpoint JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"checkpoint record must be an object at {path}:{line_number}"
                )
            records.append(value)
    return records


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(serialized + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _selected_dataset(
    dataset: PersonaEmpDataset,
    samples: Iterable[PersonaEmpSample],
) -> list[dict[str, Any]]:
    selected = {(sample.session_index, sample.query_index) for sample in samples}
    output: list[dict[str, Any]] = []
    for session_index, raw_session in enumerate(dataset.raw_sessions):
        queries = [
            query
            for query_index, query in enumerate(raw_session.get("queries", []))
            if (session_index, query_index) in selected
        ]
        if not queries:
            continue
        session_copy = dict(raw_session)
        session_copy["queries"] = queries
        output.append(session_copy)
    return output


def _prediction_rows(
    selected_dataset: list[dict[str, Any]],
    records: list[dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    response_by_query = {
        str(record["query_id"]): str(record["response"])
        for record in records
        if record.get("status") == "success" and record.get("method") == method
    }
    output: list[dict[str, Any]] = []
    for session in selected_dataset:
        responses: list[dict[str, str]] = []
        for query in session.get("queries", []):
            query_id = str(query.get("query_id") or "")
            if query_id not in response_by_query:
                raise RuntimeError(
                    f"cannot export {method}: missing successful response for {query_id}"
                )
            responses.append(
                {
                    "query_id": query_id,
                    "response": response_by_query[query_id],
                }
            )
        output.append(
            {
                "session_id": str(
                    session.get("session_id") or session.get("original_sid") or ""
                ),
                "responses": responses,
            }
        )
    return output


def _profile_preprocessing_summary(output_dir: Path) -> dict[str, Any]:
    cache_dir = output_dir / "cache" / "profiles"
    cache_entries = list(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    usage_rows: list[dict[str, Any]] = []
    legacy_entries = 0
    for path in cache_entries:
        value = json.loads(path.read_text(encoding="utf-8"))
        usage = value.get("generation_usage") if isinstance(value, dict) else None
        if isinstance(usage, dict):
            usage_rows.append(usage)
        else:
            legacy_entries += 1
    return {
        "unique_profiles": len(cache_entries),
        "profiles_with_usage": len(usage_rows),
        "profiles_missing_usage": legacy_entries,
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in usage_rows),
        "completion_tokens": sum(
            int(row.get("completion_tokens", 0)) for row in usage_rows
        ),
        "latency_seconds": round(
            sum(float(row.get("latency_seconds", 0.0)) for row in usage_rows),
            4,
        ),
        "attempts": sum(int(row.get("attempts", 0)) for row in usage_rows),
        "logical_calls": sum(
            int(row.get("logical_calls", 0)) for row in usage_rows
        ),
    }


def _json_cache_summary(directory: Path) -> dict[str, Any]:
    entries = list(directory.glob("*.json")) if directory.is_dir() else []
    usage_rows: list[dict[str, Any]] = []
    for path in entries:
        value = json.loads(path.read_text(encoding="utf-8"))
        usage = value.get("generation_usage") if isinstance(value, dict) else None
        if isinstance(usage, dict):
            usage_rows.append(usage)
    return {
        "entries": len(entries),
        "entries_with_llm_usage": len(usage_rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in usage_rows),
        "completion_tokens": sum(
            int(row.get("completion_tokens", 0)) for row in usage_rows
        ),
        "latency_seconds": round(
            sum(float(row.get("latency_seconds", 0.0)) for row in usage_rows),
            4,
        ),
    }


def _online_usage_summary(
    records: list[dict[str, Any]],
    methods: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_records = [
            record
            for record in records
            if record.get("status") == "success"
            and record.get("method") == method
        ]
        stage_rows = [
            usage
            for record in method_records
            for stage_name, usage in (record.get("stages") or {}).items()
            if stage_name not in {"profile", "summary"}
            and isinstance(usage, dict)
        ]
        summary[method] = {
            "successful_queries": len(method_records),
            "included_stages": (
                ["alignment", "response"] if method == "ours" else ["response"]
            ),
            "offline_preprocessing_excluded": True,
            "profile_preprocessing_excluded": True,
            "prompt_tokens": sum(
                int(row.get("prompt_tokens", 0)) for row in stage_rows
            ),
            "completion_tokens": sum(
                int(row.get("completion_tokens", 0)) for row in stage_rows
            ),
            "latency_seconds": round(
                sum(float(row.get("latency_seconds", 0.0)) for row in stage_rows),
                4,
            ),
            "attempts": sum(int(row.get("attempts", 0)) for row in stage_rows),
            "logical_calls": sum(
                int(row.get("logical_calls", 0)) for row in stage_rows
            ),
        }
    return summary


@dataclass(frozen=True)
class RunConfiguration:
    methods: tuple[str, ...]
    limit: int | None
    dataset_provenance: str
    expected_table1_dataset_sha256: str | None
    generator_model: str
    generator_base_url: str
    generator_enable_thinking: bool
    balanced_per_category: int | None = None

    def identity(
        self,
        dataset_fingerprint: str,
        prompt_hashes: dict[str, str],
    ) -> str:
        value = {
            "dataset_fingerprint": dataset_fingerprint,
            "methods": self.methods,
            "limit": self.limit,
            "dataset_provenance": self.dataset_provenance,
            "expected_table1_dataset_sha256": self.expected_table1_dataset_sha256,
            "generator_model": self.generator_model,
            "generator_base_url": self.generator_base_url,
            "generator_enable_thinking": self.generator_enable_thinking,
            "balanced_per_category": self.balanced_per_category,
            "prompt_hashes": prompt_hashes,
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True).encode("utf-8")
        ).hexdigest()


class PersonaEmpRunner:
    OFFICIAL_REPOSITORY = "https://github.com/ZhengWwwq/PersonalizedEmpathy"
    OFFICIAL_COMMIT = "b555447f267b8057039aab39a4be44725718ea7f"

    def __init__(
        self,
        *,
        repository_root: Path,
        dataset: PersonaEmpDataset,
        output_dir: Path,
        config: RunConfiguration,
        generators: dict[
            str,
            BaseModelGenerator
            | MemoryGenerator
            | RAGGenerator
            | DeepEmpathyGenerator,
        ],
    ) -> None:
        self.repository_root = repository_root
        self.dataset = dataset
        self.output_dir = output_dir.resolve()
        self.config = config
        self.generators = generators
        self.results_path = self.output_dir / "results.jsonl"
        self.errors_path = self.output_dir / "errors.jsonl"
        self.manifest_path = self.output_dir / "run_manifest.json"

    def _manifest(self, selected_count: int) -> dict[str, Any]:
        expected_hash = self.config.expected_table1_dataset_sha256
        table1_compatible = bool(
            expected_hash and expected_hash == self.dataset.fingerprint
        )
        generation_prompt_hashes = _generation_prompt_hashes()
        return {
            "experiment": "exp1_personaemp_deep_empathy",
            "created_at": _utc_now(),
            "run_identity": self.config.identity(
                self.dataset.fingerprint,
                generation_prompt_hashes,
            ),
            "repository_commit": _git_value(
                self.repository_root,
                "rev-parse",
                "HEAD",
            ),
            "repository_branch": _git_value(
                self.repository_root,
                "branch",
                "--show-current",
            ),
            "dataset": {
                "path": str(self.dataset.path),
                "sha256": self.dataset.fingerprint,
                "provenance": self.config.dataset_provenance,
                "total_sessions": len(self.dataset.raw_sessions),
                "total_queries": len(self.dataset.samples),
                "selected_queries": selected_count,
                "table1_direct_comparison_allowed": table1_compatible,
                "expected_table1_sha256": expected_hash,
            },
            "official_reference": {
                "repository": self.OFFICIAL_REPOSITORY,
                "commit": self.OFFICIAL_COMMIT,
                "paper": "arXiv:2606.00728v1",
            },
            "generation": {
                "protocol_version": "personaemp_benchmark_adapter_v4",
                "model": self.config.generator_model,
                "base_url": self.config.generator_base_url,
                "enable_thinking": self.config.generator_enable_thinking,
                "methods": list(self.config.methods),
                "model_inputs": {
                    "shared_raw_evidence": ["extracted_memory", "query"],
                    "excluded_generation_metadata": [
                        "persona",
                        "scenario",
                        "category",
                        "conversation",
                    ],
                    "base_model_input": [
                        "extracted_memory",
                        "query",
                    ],
                    "memory_input": [
                        "derived_flat_memory_summary",
                        "query",
                    ],
                    "rag_input": [
                        "top_3_retrieved_memory_items",
                        "query",
                    ],
                    "ours_input": [
                        "extracted_memory",
                        "query",
                        "derived_five_layer_profile",
                        "derived_deep_empathy_state",
                    ],
                    "ours_transformation": (
                        "five_layer_profile_and_deep_empathy_alignment"
                    ),
                    "dataset_persona_visible_to_generators": False,
                    "dataset_persona_visible_to_official_judges": True,
                    "external_agent_persona_visible_to_generators": False,
                    "agent_persona_visible_to_ours_alignment": False,
                },
                "response_contract": {
                    "shared_by_all_methods": True,
                    "same_language_as_query": True,
                    "paragraphs": "unrestricted",
                    "sentences": "unrestricted",
                    "brevity_instruction": False,
                    "direct_help_before_optional_question": True,
                    "maximum_follow_up_questions": 1,
                    "max_tokens": 350,
                },
                "personaemp_alignment_adapter": {
                    "agent_persona_mode": "disabled",
                    "individual_agent_persona_generated": False,
                    "self_domain_mode": "disabled_user_domain_only",
                    "current_state_inherited_across_queries": False,
                    "temporal_omega_decay_enabled": False,
                    "omega_uses_profile_completeness": True,
                },
                "structured_stage_limits": {
                    "profile_max_tokens": PROFILE_MAX_TOKENS,
                    "alignment_max_tokens": ALIGNMENT_MAX_TOKENS,
                    "parser_version": STRUCTURED_JSON_PARSER_VERSION,
                },
                "core_prompt_policy": (
                    "production_core_prompts_unchanged; "
                    "benchmark_response_contract_is_adapter_local"
                ),
                "rag": {
                    "encoder": RAG_ENCODER_MODEL,
                    "encoder_revision": RAG_ENCODER_REVISION,
                    "similarity": "cosine_on_normalized_embeddings",
                    "top_k": 3,
                    "dataset_relevant_mem_used_for_retrieval": False,
                },
                "prompt_hashes": generation_prompt_hashes,
            },
        }

    def _prepare_manifest(self, selected_count: int) -> None:
        manifest = self._manifest(selected_count)
        if self.manifest_path.is_file():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("run_identity") != manifest["run_identity"]:
                raise RuntimeError(
                    "output directory belongs to a different run configuration; "
                    "choose a new --output-dir"
                )
            return
        _atomic_json(self.manifest_path, manifest)

    def run(self) -> dict[str, Any]:
        if self.config.balanced_per_category is not None:
            if self.config.limit is not None:
                raise ValueError(
                    "limit and balanced_per_category cannot be used together"
                )
            samples = list(
                self.dataset.iter_balanced(self.config.balanced_per_category)
            )
        else:
            samples = list(self.dataset.iter_samples(self.config.limit))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_manifest(len(samples))

        prior_records = _load_jsonl(self.results_path)
        completed_ids = {
            str(record.get("sample_id"))
            for record in prior_records
            if record.get("status") == "success"
        }

        for sample in samples:
            for method in self.config.methods:
                sample_id = _sample_id(
                    self.dataset.fingerprint,
                    sample,
                    method,
                )
                if sample_id in completed_ids:
                    continue

                generator = self.generators[method]
                try:
                    output = generator.generate(sample)
                    record = {
                        "sample_id": sample_id,
                        "status": "success",
                        "completed_at": _utc_now(),
                        "session_id": sample.session_id,
                        "query_id": sample.query_id,
                        "query_sha256": hashlib.sha256(
                            sample.query.encode("utf-8")
                        ).hexdigest(),
                        **output.to_record(),
                    }
                    _append_jsonl(self.results_path, record)
                    completed_ids.add(sample_id)
                except Exception as exc:
                    error_record = {
                        "sample_id": sample_id,
                        "status": "error",
                        "failed_at": _utc_now(),
                        "session_id": sample.session_id,
                        "query_id": sample.query_id,
                        "method": method,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    _append_jsonl(self.errors_path, error_record)
                    raise

        all_records = _load_jsonl(self.results_path)
        selected_dataset = _selected_dataset(self.dataset, samples)
        evaluation_dataset_path = self.output_dir / "evaluation_dataset.json"
        _atomic_json(evaluation_dataset_path, selected_dataset)

        prediction_paths: dict[str, str] = {}
        for method in self.config.methods:
            predictions = _prediction_rows(selected_dataset, all_records, method)
            prediction_path = self.output_dir / "predictions" / f"{method}.json"
            _atomic_json(prediction_path, predictions)
            prediction_paths[method] = str(prediction_path)

        summary = {
            "selected_queries": len(samples),
            "methods": list(self.config.methods),
            "successful_results": sum(
                1
                for record in all_records
                if record.get("status") == "success"
                and record.get("method") in self.config.methods
            ),
            "evaluation_dataset": str(evaluation_dataset_path),
            "predictions": prediction_paths,
            "profile_preprocessing": _profile_preprocessing_summary(
                self.output_dir
            ),
            "memory_summary_preprocessing": _json_cache_summary(
                self.output_dir / "cache" / "memory_summaries"
            ),
            "rag_embedding_preprocessing": _json_cache_summary(
                self.output_dir / "cache" / "rag_embeddings"
            ),
            "online_inference": _online_usage_summary(
                all_records,
                self.config.methods,
            ),
        }
        _atomic_json(self.output_dir / "summary.json", summary)
        return summary

"""Ca-internal V15 development runner; outputs never enter Table 2."""
from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exp1_protocol import (
    build_message_level_points,
    build_profile_corpus,
    message_speakers,
    protocol_turns,
    select_realtalk_splits,
    session_keys,
    stable_hash,
)
from .operation_checkpoint import OperationCheckpoint
from .personaemp.client import ChatBackend
from .realtalk_ours import (
    DOMAIN_MAX_TOKENS,
    SELF_DOMAIN_SYSTEM_PROMPT,
    SELF_DOMAIN_USER_TEMPLATE,
    USER_DOMAIN_SYSTEM_PROMPT,
    USER_DOMAIN_USER_TEMPLATE,
    _backend_from_env,
    _checkpoint_unresolved,
    _failure,
    _json,
    _observable_statistics,
    _repository_commit,
    _safe_host,
    _speaker_id,
    _structured_call,
    _turns_with_ids,
    _turns_with_session_boundaries,
    _validate_observable_statistics,
    _validate_user_domain_evidence,
    _write_json,
    _write_jsonl,
)
from .realtalk_ours_schemas import (
    PROFILE_LAYERS,
    SELF_DOMAIN_SCHEMA,
    USER_DOMAIN_SCHEMA,
    empty_user_domain,
    normalize_self_domain,
    normalize_user_domain,
)
from .realtalk_v14 import (
    _latest_partner_turn,
    _load_json,
    _preflight,
    build_ca_behavior_bank,
    classify_interaction_trigger,
    summarize_behavior_bank,
    summarize_visible_target_behavior,
)
from .realtalk_v15 import (
    CONTROLLER_SYSTEM_PROMPT,
    CONTROLLER_USER_TEMPLATE,
    MODEL,
    _fact_ownership_audit,
    _prompt_hashes,
    _run_actor_with_contract_retries,
    _target_owned_history_text,
    aggregate_v15_diagnostics,
    build_v15_activation_whitelist,
    resolve_v15_profile_activation,
    _v15_actor_self_domain,
    _v15_activation_whitelist_text,
)
from .realtalk_v15_schemas import DECISION_SCHEMA, normalize_v15_decision


PROTOCOL = "realtalk_task1_ours_v15_12_ca_internal_development"
GATES = (6, 30)

CA_DEV_USER_DOMAIN_SCHEMA = copy.deepcopy(USER_DOMAIN_SCHEMA)
CA_DEV_USER_DOMAIN_SCHEMA["name"] = "realtalk_ours_v15_ca_dev_compact_user_domain_v1"
for _layer in PROFILE_LAYERS:
    _layer_schema = CA_DEV_USER_DOMAIN_SCHEMA["schema"]["properties"][_layer]
    _layer_schema["maxItems"] = 3
    _layer_schema["items"]["properties"]["evidence_ids"]["maxItems"] = 4
for _summary_key in ("added", "revised", "removed", "uncertainties"):
    CA_DEV_USER_DOMAIN_SCHEMA["schema"]["properties"]["update_summary"]["properties"][
        _summary_key
    ]["maxItems"] = 4

CA_DEV_USER_DOMAIN_SYSTEM_PROMPT = USER_DOMAIN_SYSTEM_PROMPT + """

For this causal development profile, keep the representation compact while reading the complete session:
- at most three non-overlapping facts per layer;
- each value is one concise sentence of at most about twenty words;
- at most four evidence IDs per fact;
- at most four entries in each update_summary field.
Preserve the strongest stable evidence instead of fragmenting one pattern into several facts.

Evidence ownership is strict: the target speaker's turns are context only and can never support a fact about
the partner. Every evidence_ids entry must be copied exactly from the explicit allowed-partner-ID whitelist in
the user prompt. If no allowed partner turn supports a fact, omit that fact instead of citing a target turn."""

CA_DEV_USER_DOMAIN_USER_TEMPLATE = USER_DOMAIN_USER_TEMPLATE + """

ONLY ALLOWED PARTNER EVIDENCE IDS AFTER THIS UPDATE:
{allowed_partner_evidence_ids}

Copy evidence IDs exactly from this whitelist. Any other ID belongs to the target speaker, is not yet visible,
or is malformed and must not appear in the output."""
CA_DEV_EVIDENCE_ID_NORMALIZATION = "session_turn_shorthand_v2"


@dataclass(frozen=True)
class V15CaDevConfig:
    dataset_dir: str = "dataset"
    output_dir: str = "data/realtalk_v15_ca_dev6"
    gate: int = 6
    model: str = MODEL
    operation_max_attempts: int = 3
    model_call_timeout_seconds: int = 240
    fresh: bool = False
    resume: bool = False


def run_v15_ca_dev(
    config: V15CaDevConfig, backend: ChatBackend | None = None
) -> dict[str, Any]:
    _validate_config(config)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.fresh:
        _clear_outputs(output_dir)
    elif config.resume and not (output_dir / "checkpoint.json").exists():
        raise ValueError("--resume requires an existing checkpoint.json")
    (output_dir / "CA_DEVELOPMENT_COMPLETE").unlink(missing_ok=True)

    prepared, dataset_manifest = _prepare_ca_dev(config.dataset_dir)
    gate_manifest = _ca_dev_gate_manifest(prepared)
    selected_ids = gate_manifest[str(config.gate)]
    point_index = {
        point["result_id"]: (item, point)
        for item in prepared for point in item["points"]
    }
    selected_speakers = {
        point_index[result_id][0]["speaker"] for result_id in selected_ids
    }

    backend = backend or _backend_from_env(config.model)
    if getattr(backend, "model", None) != config.model:
        raise ValueError(
            f"V15 Ca dev requires model {config.model!r}; got {getattr(backend, 'model', None)!r}"
        )
    preflight = _preflight(output_dir, backend, config)
    signature = stable_hash({
        "protocol": PROTOCOL,
        "config": {
            key: value for key, value in asdict(config).items()
            if key not in {"output_dir", "fresh", "resume", "gate"}
        },
        "dataset_manifest": dataset_manifest,
        "gate_manifest": gate_manifest,
        "prompt_hashes": _prompt_hashes(),
        "schemas": {
            "self_domain": SELF_DOMAIN_SCHEMA,
            "user_domain": CA_DEV_USER_DOMAIN_SCHEMA,
            "controller": DECISION_SCHEMA,
        },
        "implementation_commit": _repository_commit(),
        "ca_dev_evidence_id_normalization": CA_DEV_EVIDENCE_ID_NORMALIZATION,
    })
    checkpoint = OperationCheckpoint(output_dir / "checkpoint.json", signature)
    raw_audit = output_dir / "raw_responses.jsonl"
    self_domains: dict[str, dict[str, Any]] = {}
    user_domains: dict[str, dict[str, Any]] = {}

    for item in prepared:
        speaker = item["speaker"]
        if speaker not in selected_speakers:
            continue
        speaker_id = _speaker_id(speaker)
        try:
            expected_stats = _observable_statistics(item["profile"]["turns"], speaker)
            self_envelope = _structured_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"ca_dev_self:{speaker_id}",
                system_prompt=SELF_DOMAIN_SYSTEM_PROMPT,
                user_prompt=SELF_DOMAIN_USER_TEMPLATE.format(
                    speaker=speaker,
                    session_count=2,
                    full_context=_turns_with_ids(item["profile"]["turns"]),
                    observable_statistics=_json(expected_stats),
                ),
                schema=SELF_DOMAIN_SCHEMA,
                normalizer=lambda value, stats=expected_stats: _validate_observable_statistics(
                    normalize_self_domain(value), stats
                ),
                max_tokens=DOMAIN_MAX_TOKENS,
                max_attempts=config.operation_max_attempts,
                raw_audit=raw_audit,
                enable_thinking=False,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
            self_domains[speaker] = self_envelope["data"]
            domain = empty_user_domain()
            allowed_partner_ids: set[str] = set()
            for session_id in item["profile"]["sessions"]:
                turns = item["turns_by_session"][session_id]
                new_ids = {
                    turn["turn_id"] for turn in turns
                    if turn["speaker"].casefold() == item["partner"].casefold()
                }
                allowed_after = allowed_partner_ids | new_ids
                envelope = _structured_call(
                    checkpoint=checkpoint,
                    backend=backend,
                    operation_key=f"ca_dev_user:{speaker_id}:after:{session_id}",
                    system_prompt=CA_DEV_USER_DOMAIN_SYSTEM_PROMPT,
                    user_prompt=CA_DEV_USER_DOMAIN_USER_TEMPLATE.format(
                        speaker=speaker,
                        partner=item["partner"],
                        previous_domain=_json(domain),
                        completed_session=_turns_with_ids(turns),
                        allowed_partner_evidence_ids=_json(sorted(allowed_after)),
                    ),
                    schema=CA_DEV_USER_DOMAIN_SCHEMA,
                    normalizer=lambda value, allowed=allowed_after: _validate_user_domain_evidence(
                        _normalize_ca_dev_user_domain(value, allowed), allowed
                    ),
                    max_tokens=DOMAIN_MAX_TOKENS,
                    max_attempts=config.operation_max_attempts,
                    raw_audit=raw_audit,
                    enable_thinking=False,
                    hard_timeout_seconds=config.model_call_timeout_seconds,
                )
                domain = envelope["data"]
                allowed_partner_ids = allowed_after
            user_domains[speaker] = domain
        except Exception as exc:
            checkpoint.store_excluded_result(
                f"ca_dev_domain:{speaker_id}", _failure("ca_dev_domain", speaker, None, exc)
            )

    for result_id in selected_ids:
        if result_id in checkpoint.data["results"]:
            continue
        item, point = point_index[result_id]
        speaker = item["speaker"]
        if speaker not in self_domains or speaker not in user_domains:
            continue
        domain = user_domains[speaker]
        latest_partner = _latest_partner_turn(
            point["context_turns"], item["partner"], point["target_session"]
        )
        current_session_turns = [
            turn for turn in point["context_turns"]
            if turn["session_id"] == point["target_session"]
        ]
        ca_summary = summarize_behavior_bank(
            build_ca_behavior_bank(item["profile"]["turns"], speaker)
        )
        activation_facts = build_v15_activation_whitelist(domain)
        try:
            controller = _structured_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"ca_dev_controller:{result_id}",
                system_prompt=CONTROLLER_SYSTEM_PROMPT,
                user_prompt=CONTROLLER_USER_TEMPLATE.format(
                    speaker=speaker,
                    partner=item["partner"],
                    self_domain=_json(self_domains[speaker]),
                    target_identity_context=_json(
                        self_domains[speaker]["identity_context"]
                    ),
                    user_domain=_json(domain),
                    history=_turns_with_session_boundaries(point["context_turns"]),
                    target_owned_history=_target_owned_history_text(
                        point["context_turns"], speaker
                    ),
                    latest_partner_turn=(
                        _turns_with_session_boundaries([latest_partner])
                        if latest_partner else "NONE"
                    ),
                    current_trigger=classify_interaction_trigger(
                        point["context_turns"], speaker, point["target_session"]
                    ),
                    ca_behavior_summary=_json(ca_summary),
                    cb_behavior_summary=_json(
                        summarize_visible_target_behavior(current_session_turns, speaker)
                    ),
                    activation_whitelist=_v15_activation_whitelist_text(activation_facts),
                ),
                schema=DECISION_SCHEMA,
                normalizer=lambda value, facts=activation_facts: resolve_v15_profile_activation(
                    normalize_v15_decision(value), facts
                ),
                max_tokens=1600,
                max_attempts=config.operation_max_attempts,
                raw_audit=raw_audit,
                enable_thinking=False,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
            decision = controller["data"]
            actor = _run_actor_with_contract_retries(
                checkpoint=checkpoint,
                backend=backend,
                result_id=f"ca_dev:{result_id}",
                speaker=speaker,
                history=_turns_with_session_boundaries(point["context_turns"]),
                self_domain=_v15_actor_self_domain(self_domains[speaker]),
                turn_plan=decision["turn_plan"],
                raw_audit=raw_audit,
                max_attempts=config.operation_max_attempts,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
            fact_ownership_audit = _fact_ownership_audit(
                actor["generated_message"],
                latest_partner,
                point["context_turns"],
                speaker,
                self_domains[speaker],
            )
            if fact_ownership_audit["warning"]:
                raise ValueError(
                    "fact ownership audit found unsupported partner-to-target transfer: "
                    f"tokens={fact_ownership_audit['overlap_tokens']} "
                    f"concepts={fact_ownership_audit.get('unsupported_concepts', [])} "
                    f"pairs={fact_ownership_audit.get('mirrored_concept_pairs', [])}"
                )
            checkpoint.store_result(result_id, {
                "result_id": result_id,
                "speaker": speaker,
                "partner": item["partner"],
                "source_chat": item["split"]["train_chat"],
                "profile_sessions": list(item["profile"]["sessions"]),
                "target_session": point["target_session"],
                "context_turn_ids": [turn["turn_id"] for turn in point["context_turns"]],
                "context_hash": point["history_hash"],
                "context_truncated": point["context_truncated"],
                "ground_truth": point["target_message"],
                "generated_message": actor["generated_message"],
                "self_domain_hash": stable_hash(self_domains[speaker]),
                "user_domain": domain,
                "situation": decision["situation"],
                "relevant_user_domain": decision["relevant_user_domain"],
                "alignment": decision["alignment"],
                "turn_plan": decision["turn_plan"],
                "ca_behavior_summary": ca_summary,
                "visible_cb_behavior": summarize_visible_target_behavior(
                    current_session_turns, speaker
                ),
                "actor_structure_audit": actor["audit"],
                "fact_ownership_audit": fact_ownership_audit,
                "operation_audit": {
                    "controller": controller["audit"],
                    "actor": actor["operation_audits"],
                },
                "ca_internal_development_only": True,
            })
        except Exception as exc:
            checkpoint.store_excluded_result(
                result_id, _failure("ca_dev_sample", speaker, result_id, exc)
            )

    result_index = {row["result_id"]: row for row in checkpoint.result_values()}
    results = [result_index[result_id] for result_id in selected_ids if result_id in result_index]
    unresolved = _checkpoint_unresolved(checkpoint)
    diagnostics = aggregate_v15_diagnostics(results)
    _write_jsonl(output_dir / "predictions.jsonl", results)
    _write_json(output_dir / "self_domains_session12.json", self_domains)
    _write_json(output_dir / "user_domains_session12.json", user_domains)
    _write_json(output_dir / "gate_manifest.json", gate_manifest)
    _write_json(output_dir / "unresolved_errors.json", unresolved)
    _write_json(output_dir / "diagnostics.json", diagnostics)
    _write_json(output_dir / "run_manifest.json", {
        "created_at_utc": _now(),
        "protocol": PROTOCOL,
        "purpose": "prompt development only; excluded from Table 2",
        "implementation_repository_commit": _repository_commit(),
        "model": backend.model,
        "profile_sessions": 2,
        "development_target_session": 3,
        "ca_target_or_future_visible": False,
        "thinking_enabled": False,
        "prompt_hashes": _prompt_hashes(),
        "schema_hashes": {
            "self_domain": stable_hash(SELF_DOMAIN_SCHEMA),
            "user_domain": stable_hash(CA_DEV_USER_DOMAIN_SCHEMA),
            "controller": stable_hash(DECISION_SCHEMA),
        },
        "ca_dev_evidence_id_normalization": CA_DEV_EVIDENCE_ID_NORMALIZATION,
        "dataset_manifest": dataset_manifest,
        "gate": config.gate,
        "preflight": preflight,
        "base_url_host": _safe_host(getattr(backend, "base_url", "injected-test-backend")),
        "config": asdict(config),
        "run_signature": signature,
    })
    complete = len(results) == len(selected_ids) and not unresolved
    if complete:
        _write_json(output_dir / "CA_DEVELOPMENT_COMPLETE", {
            "completed_at_utc": _now(),
            "records": len(results),
            "gate": config.gate,
            "run_signature": signature,
        })
    return {
        "complete": complete,
        "records": len(results),
        "expected_records": len(selected_ids),
        "unresolved": unresolved,
        "diagnostics": diagnostics,
        "output_dir": str(output_dir),
    }


def _prepare_ca_dev(dataset_dir: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = Path(dataset_dir)
    prepared = []
    files = {}
    for split in select_realtalk_splits(dataset_dir):
        path = dataset / split["train_chat"]
        chat = _load_json(path)
        speaker = next(
            value for value in message_speakers(chat)
            if value.casefold() == split["speaker"].casefold()
        )
        partner = next(
            value for value in message_speakers(chat)
            if value.casefold() != speaker.casefold()
        )
        sessions = session_keys(chat)[:3]
        profile = build_profile_corpus(chat, speaker, profile_sessions=2)
        points = [
            point for point in build_message_level_points(
                chat, speaker, test_sessions=3, max_context_chars=0,
                merge_adjacent_bubbles=True,
            )
            if point["target_session"] == sessions[2]
        ]
        if len(points) < 3:
            raise ValueError(f"Ca Session 3 has fewer than three targets for {speaker}")
        turns = protocol_turns(chat, merge_adjacent_bubbles=True)
        selected = set(sessions)
        turns_by_session = {
            session: [turn for turn in turns if turn["session_id"] == session]
            for session in sessions if session in selected
        }
        for point in points:
            point["result_id"] = f"{_speaker_id(speaker)}:ca_dev:{point['sample_id']}"
        prepared.append({
            "split": split,
            "speaker": speaker,
            "partner": partner,
            "profile": profile,
            "points": points,
            "turns_by_session": turns_by_session,
        })
        files[path.name] = hashlib_sha256(path)
    return prepared, {
        "dataset": "REALTALK paper-assigned Ca internal development",
        "source_files_sha256": dict(sorted(files.items())),
        "profile_sessions": [1, 2],
        "development_session": 3,
        "history_compression_enabled": False,
        "history_truncation_enabled": False,
        "generated_outputs_are_never_rolled_into_history": True,
    }


def _normalize_ca_dev_user_domain(
    value: Any, allowed_turn_ids: set[str] | None = None
) -> dict[str, Any]:
    domain = normalize_user_domain(value)
    allowed = allowed_turn_ids or set()
    for layer in PROFILE_LAYERS:
        for fact in domain[layer]:
            fact["evidence_ids"] = [
                _normalize_ca_dev_evidence_id(evidence_id, allowed)
                for evidence_id in fact["evidence_ids"]
            ]
    return domain


def _normalize_ca_dev_evidence_id(
    value: str, allowed_turn_ids: set[str] | None = None
) -> str:
    match = re.fullmatch(
        r"s(?:ession)?[_ -]?(\d+)\s*[:/_-]\s*t(?:urn)?[_ -]?(\d+)",
        value.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        bare = re.fullmatch(r"(?:turn[_ -]?|t)(\d+)", value.strip(), flags=re.IGNORECASE)
        if not bare:
            return value
        suffix = f":turn_{int(bare.group(1))}"
        candidates = sorted(
            item for item in (allowed_turn_ids or set()) if item.endswith(suffix)
        )
        return candidates[0] if len(candidates) == 1 else value
    return f"session_{int(match.group(1))}:turn_{int(match.group(2))}"


def _ca_dev_gate_manifest(prepared: list[dict[str, Any]]) -> dict[str, list[str]]:
    per_speaker = {}
    for item in prepared:
        points = item["points"]
        indices = sorted({len(points) // 4, len(points) // 2, (3 * len(points)) // 4})
        if len(indices) != 3:
            indices = [0, len(points) // 2, len(points) - 1]
        per_speaker[item["speaker"]] = [points[index]["result_id"] for index in indices]
    first = [per_speaker[item["speaker"]][0] for item in prepared]
    all_ids = []
    for rank in range(3):
        all_ids.extend(per_speaker[item["speaker"]][rank] for item in prepared)
    return {"6": first[:6], "30": all_ids}


def hashlib_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_config(config: V15CaDevConfig) -> None:
    if config.gate not in GATES:
        raise ValueError(f"Ca dev gate must be one of {GATES}")
    if config.model != MODEL:
        raise ValueError(f"V15 model is frozen to {MODEL}")
    if config.operation_max_attempts != 3:
        raise ValueError("operation attempts are fixed at three")
    if config.fresh == config.resume:
        raise ValueError("exactly one of --fresh or --resume is required")


def _clear_outputs(output_dir: Path) -> None:
    for name in (
        "checkpoint.json", "checkpoint.json.tmp", "raw_responses.jsonl",
        "predictions.jsonl", "self_domains_session12.json",
        "user_domains_session12.json", "gate_manifest.json",
        "unresolved_errors.json", "diagnostics.json", "run_manifest.json",
        "CA_DEVELOPMENT_COMPLETE", "preflight.json",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> V15CaDevConfig:
    parser = argparse.ArgumentParser(description="Run REALTALK V15 Ca-internal development")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate", type=int, choices=GATES, required=True)
    parser.add_argument("--model", default=MODEL)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return V15CaDevConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        gate=args.gate,
        model=args.model,
        fresh=args.fresh,
        resume=args.resume,
    )


if __name__ == "__main__":
    print(json.dumps(run_v15_ca_dev(parse_args()), ensure_ascii=False, indent=2))

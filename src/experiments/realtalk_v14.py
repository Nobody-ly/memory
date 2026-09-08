"""V14 REALTALK replay with Ca behavior retrieval and turn-bundle generation.

The runner freezes V9 Self/User Domains and reconstructs only the private
Decision and Response Actor stages. Ca examples are retrieved by deterministic
context similarity; Cb targets, future turns, and evaluator labels never enter
retrieval or prompting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exp1_protocol import REALTALK_PERSONA_SPLITS, select_realtalk_splits, stable_hash
from .operation_checkpoint import OperationCheckpoint
from .personaemp.client import ChatBackend
from .realtalk_ours import (
    RealTalkOursConfig,
    _backend_from_env,
    _behavioral_self_domain,
    _checkpoint_unresolved,
    _failure,
    _json,
    _prepare_dataset,
    _profile_activation_whitelist,
    _repository_commit,
    _safe_host,
    _speaker_id,
    _structured_call,
    _text_call,
    _turns_with_session_boundaries,
    _validate_decision_profile_activation,
    _write_json,
    _write_jsonl,
)
from .realtalk_v14_schemas import DECISION_SCHEMA, normalize_v14_decision


MODEL = "deepseek-v4-flash"
PROTOCOL = "realtalk_task1_ours_v14_ca_behavior_turn_bundle"
EXPECTED_V9_COMMIT = "5927bbff03fda74eebaeb99e0c57203a644cfd74"
EXPECTED_V9_PREDICTIONS_SHA256 = (
    "ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81"
)
EXPECTED_RECORDS = 519
GATES = (6, 18, 30, 60, 120, 519)


DECISION_SYSTEM_PROMPT = """You are the private planning agent for a persona-simulation actor.
Your task is to decide the next conversational turn of the target person, not an ideal assistant reply.

Use the target's fixed Self Domain as identity and voice. Read the complete visible Cb history as the
current reality. Treat retrieved Ca examples as evidence of how this person tends to compose a turn after
similar conversational triggers; their concrete facts are past context, never current facts to copy.
Use at most two User Domain facts and only when they directly matter now.

Plan one natural turn. A turn may contain several consecutive chat bubbles and several compatible social
moves. Select one primary move and up to two supporting moves in their intended order. Do not force a
question, reflection, acknowledgement, or self-disclosure; include each only when the visible interaction,
the person's observed behavior, or close Ca analogues support it. Conversely, do not compress a naturally
multi-part response into a mechanical single action when the person regularly combines moves.

lambda_trace records how strongly the current partner-facing situation shapes this turn relative to the
person's stable prior. It is not a reward and is not fixed near zero. A direct question, explicit support
request, or emotionally consequential disclosure usually makes the plan balanced or partner-adaptive;
routine self-led contribution may remain self-led. The chosen orientation and lambda must visibly agree
with the message plan while preserving the target person's identity.

Reflection means briefly expressing a thought, reason, or feeling about what is being discussed. Use none
for ordinary factual or lightweight turns, surface for a simple stance or feeling, and brief only when the
interaction and the person's evidence support an actual reflective contribution. Respond to the real
conversational opening rather than automatically asking a question. Return only the strict schema."""


DECISION_USER_TEMPLATE = """TARGET SPEAKER: {speaker}
PARTNER: {partner}

FIXED V9 SELF DOMAIN:
{self_domain}

CURRENT FIVE-LAYER USER DOMAIN:
{user_domain}

REAL CAUSAL HISTORY BEFORE THE TARGET TURN:
{history}

LATEST PARTNER TURN:
{latest_partner_turn}

CURRENT INTERACTION TRIGGER (deterministic hint): {current_trigger}

TARGET'S OBSERVED CA BEHAVIOR SUMMARY:
{behavior_summary}

TOP CA BEHAVIOR ANALOGUES:
{behavior_examples}

TARGET'S OBSERVED Cb BEHAVIOR BEFORE THIS POINT:
{online_behavior}

EXACT USER DOMAIN ACTIVATION WHITELIST:
{activation_whitelist}

The analogue target texts are past behavioral evidence, not drafts and not current-world facts. Decide the
turn structure, adaptive balance, and content direction for {speaker}. Copy activated profile facts exactly
from the whitelist; if none are relevant, return an empty relevant_user_domain array."""


ACTOR_SYSTEM_TEMPLATE = """You are {speaker}. Continue the conversation.
Act as the person represented by the private Self Domain.
Follow the private turn plan naturally.
Output only the message, not the speaker name."""


ACTOR_USER_TEMPLATE = """REAL CONVERSATION HISTORY BEFORE YOUR NEXT TURN:
{history}

PRIVATE SELF DOMAIN:
{self_domain}

RELEVANT PARTNER FACTS FOR THIS TURN:
{relevant_user_domain}

PAST CA BEHAVIOR ANALOGUES:
{behavior_examples}

PRIVATE CURRENT SITUATION:
{situation}

PRIVATE TURN PLAN:
{message_plan}

Write one natural conversational turn as {speaker}. Complete the primary move and only the compatible
supporting moves in the plan. Match the planned relationship register, reflection depth, length band, and
question plan. When bubble_count is greater than one, separate consecutive chat bubbles with newline
characters; do not number or label them. Ca analogue wording and concrete events belong to old conversations:
use them only to learn interaction shape and style, never present them as current facts or copy their text.
Do not mention the plan, domains, examples, or any internal reasoning."""


@dataclass(frozen=True)
class V14Config:
    dataset_dir: str = "dataset"
    v9_predictions: str = ""
    v9_self_domains: str = ""
    output_dir: str = "data/realtalk_v14_gate6"
    gate: int = 6
    model: str = MODEL
    operation_max_attempts: int = 3
    model_call_timeout_seconds: int = 240
    fresh: bool = False
    resume: bool = False
    enforce_canonical_v9: bool = True


def run_v14(config: V14Config, backend: ChatBackend | None = None) -> dict[str, Any]:
    _validate_config(config)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.fresh:
        _clear_outputs(output_dir)
    elif config.resume and not (output_dir / "checkpoint.json").exists():
        raise ValueError("--resume requires an existing checkpoint.json")

    predictions_path = Path(config.v9_predictions).resolve()
    self_domains_path = Path(config.v9_self_domains).resolve()
    v9_rows = _load_jsonl(predictions_path)
    self_domains = _load_json(self_domains_path)
    source = _validate_v9_source(
        predictions_path,
        self_domains_path,
        v9_rows,
        self_domains,
        enforce_canonical=config.enforce_canonical_v9,
    )

    base_config = RealTalkOursConfig(
        dataset_dir=config.dataset_dir,
        compute_local_metrics=False,
        compute_bertscore=False,
        decision_thinking=False,
        model=MODEL,
    )
    dataset_manifest, prepared = _prepare_dataset(
        base_config, select_realtalk_splits(config.dataset_dir)
    )
    point_index, speaker_index = _index_prepared(prepared)
    _validate_v9_alignment(v9_rows, self_domains, point_index)
    gate_manifest = build_progressive_gate_manifest(prepared)
    selected_ids = gate_manifest[str(config.gate)]
    v9_index = {row["result_id"]: row for row in v9_rows}

    behavior_banks = {
        item["speaker"]: build_ca_behavior_bank(
            item["profile"]["turns"], item["speaker"]
        )
        for item in prepared
    }
    behavior_summaries = {
        speaker: summarize_behavior_bank(bank)
        for speaker, bank in behavior_banks.items()
    }

    backend = backend or _backend_from_env(config.model)
    if getattr(backend, "model", None) != config.model:
        raise ValueError(
            f"V14 requires model {config.model!r}; got {getattr(backend, 'model', None)!r}"
        )
    preflight = _preflight(output_dir, backend, config)
    signature = stable_hash({
        "protocol": PROTOCOL,
        "config": {
            key: value for key, value in asdict(config).items()
            if key not in {"output_dir", "fresh", "resume", "gate"}
        },
        "dataset_manifest": dataset_manifest,
        "v9_source": source,
        "progressive_gate_manifest": gate_manifest,
        "decision_schema": DECISION_SCHEMA,
        "prompt_hashes": _prompt_hashes(),
        "implementation_commit": _repository_commit(),
    })
    checkpoint = OperationCheckpoint(output_dir / "checkpoint.json", signature)
    raw_audit = output_dir / "raw_responses.jsonl"

    for result_id in selected_ids:
        if result_id in checkpoint.data["results"]:
            continue
        point = point_index[result_id]
        speaker_data = speaker_index[point["speaker"]]
        v9 = v9_index[result_id]
        self_domain = self_domains[point["speaker"]]
        current_trigger = classify_interaction_trigger(
            point["context_turns"], point["speaker"], point["target_session"]
        )
        analogues = retrieve_ca_behavior_examples(
            behavior_banks[point["speaker"]],
            point["context_turns"],
            point["speaker"],
            current_session=point["target_session"],
            limit=3,
        )
        latest_partner = _latest_partner_turn(
            point["context_turns"], speaker_data["partner"], point["target_session"]
        )
        try:
            decision_envelope = _structured_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"v14_decision:{result_id}",
                system_prompt=DECISION_SYSTEM_PROMPT,
                user_prompt=DECISION_USER_TEMPLATE.format(
                    speaker=point["speaker"],
                    partner=speaker_data["partner"],
                    self_domain=_json(self_domain),
                    user_domain=_json(v9["user_domain"]),
                    history=_turns_with_session_boundaries(point["context_turns"]),
                    latest_partner_turn=(
                        _turns_with_session_boundaries([latest_partner])
                        if latest_partner else "NONE"
                    ),
                    current_trigger=current_trigger,
                    behavior_summary=_json(behavior_summaries[point["speaker"]]),
                    behavior_examples=_json(analogues),
                    online_behavior=_json(
                        summarize_visible_target_behavior(
                            point["context_turns"], point["speaker"]
                        )
                    ),
                    activation_whitelist=_profile_activation_whitelist(
                        v9["user_domain"]
                    ),
                ),
                schema=DECISION_SCHEMA,
                normalizer=lambda value, domain=v9["user_domain"], trigger=current_trigger: (
                    _validate_v14_context(
                        _validate_decision_profile_activation(
                            normalize_v14_decision(value), domain
                        ),
                        trigger=trigger,
                        has_history=bool(point["context_turns"]),
                    )
                ),
                max_tokens=1600,
                max_attempts=config.operation_max_attempts,
                raw_audit=raw_audit,
                enable_thinking=False,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
            decision = decision_envelope["data"]
            actor_examples = [
                {
                    "trigger": item["trigger"],
                    "previous_turn": item["previous_turn"],
                    "target_turn": item["target_turn"],
                    "bubble_count": item["bubble_count"],
                }
                for item in analogues
            ]
            generation_envelope = _text_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"v14_actor:{result_id}",
                system_prompt=ACTOR_SYSTEM_TEMPLATE.format(speaker=point["speaker"]),
                user_prompt=ACTOR_USER_TEMPLATE.format(
                    speaker=point["speaker"],
                    history=_turns_with_session_boundaries(point["context_turns"]),
                    self_domain=_json(_v14_actor_self_domain(self_domain)),
                    relevant_user_domain=_json(decision["relevant_user_domain"]),
                    behavior_examples=_json(actor_examples),
                    situation=_json(decision["situation"]),
                    message_plan=_json(decision["message_plan"]),
                ),
                speaker=point["speaker"],
                max_attempts=config.operation_max_attempts,
                raw_audit=raw_audit,
                enable_thinking=False,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
            generated = generation_envelope["data"]
            result = {
                **{
                    key: v9[key] for key in (
                        "result_id", "speaker", "partner", "train_chat", "test_chat",
                        "profile_sessions", "test_sessions", "target_session",
                        "message_level_index", "target_turn_id", "context_turn_ids",
                        "context_hash", "context_truncated", "ground_truth",
                        "self_domain_hash", "user_domain",
                        "user_domain_completed_session_updates",
                    )
                },
                "generated_message": generated,
                "v9_generated_message": v9["generated_message"],
                "ca_behavior_trigger": current_trigger,
                "ca_behavior_examples": analogues,
                "online_target_behavior": summarize_visible_target_behavior(
                    point["context_turns"], point["speaker"]
                ),
                "situation": decision["situation"],
                "relevant_user_domain": decision["relevant_user_domain"],
                "alignment": decision["alignment"],
                "message_plan": decision["message_plan"],
                "actor_structure_audit": actor_structure_audit(
                    generated, decision["message_plan"]
                ),
                "operation_audit": {
                    "decision": decision_envelope["audit"],
                    "generation": generation_envelope["audit"],
                },
            }
            checkpoint.data["failures"].pop(f"sample:{result_id}", None)
            checkpoint.store_result(result_id, result)
        except Exception as exc:
            checkpoint.store_excluded_result(
                result_id, _failure("v14_sample", point["speaker"], result_id, exc)
            )

    result_index = {row["result_id"]: row for row in checkpoint.result_values()}
    results = [result_index[result_id] for result_id in selected_ids if result_id in result_index]
    unresolved = _checkpoint_unresolved(checkpoint)
    diagnostics = aggregate_v14_diagnostics(results)
    _write_jsonl(output_dir / "predictions.jsonl", results)
    _write_json(output_dir / "unresolved_errors.json", unresolved)
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(output_dir / "progressive_gate_manifest.json", gate_manifest)
    _write_json(output_dir / "behavior_banks.json", behavior_banks)
    _write_json(output_dir / "behavior_summaries.json", behavior_summaries)
    _write_json(output_dir / "diagnostics.json", diagnostics)
    _write_json(output_dir / "run_manifest.json", {
        "created_at_utc": _now(),
        "protocol": PROTOCOL,
        "implementation_repository_commit": _repository_commit(),
        "model": backend.model,
        "thinking_enabled": {"decision": False, "actor": False},
        "training_or_finetuning": False,
        "frozen_v9_upstream": True,
        "regenerated_stages": ["decision", "actor"],
        "frozen_stages": ["self_domain", "user_domain"],
        "omega_enabled": False,
        "future_user_state_enabled": False,
        "history_compression_enabled": False,
        "history_truncation_enabled": False,
        "retrieval_uses_ground_truth": False,
        "retrieval_uses_cb_future": False,
        "retrieval_uses_judge_labels": False,
        "retrieval_source": "target-speaker turns from paper-assigned Ca first three sessions",
        "source_v9": source,
        "gate": config.gate,
        "selected_result_ids_sha256": stable_hash(selected_ids),
        "selected_result_count": len(selected_ids),
        "prompt_hashes": _prompt_hashes(),
        "schema_hashes": {"decision": stable_hash(DECISION_SCHEMA)},
        "decoding": {
            "decision": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1600},
            "actor": {"temperature": 0.6, "top_p": 0.9, "max_tokens": 300},
        },
        "preflight": preflight,
        "base_url_host": _safe_host(getattr(backend, "base_url", "injected-test-backend")),
        "config": asdict(config),
        "run_signature": signature,
    })
    complete = len(results) == len(selected_ids) and not unresolved
    if complete:
        _write_json(output_dir / "GENERATION_COMPLETE", {
            "completed_at_utc": _now(),
            "records": len(results),
            "gate": config.gate,
            "model": backend.model,
            "run_signature": signature,
        })
    return {
        "generation_complete": complete,
        "records": len(results),
        "expected_records": len(selected_ids),
        "unresolved": unresolved,
        "diagnostics": diagnostics,
        "output_dir": str(output_dir),
    }


def build_ca_behavior_bank(
    turns: list[dict[str, Any]], speaker: str
) -> list[dict[str, Any]]:
    bank: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if turn["speaker"].casefold() != speaker.casefold():
            continue
        previous = turns[index - 1] if index else None
        if previous and previous["session_id"] != turn["session_id"]:
            previous = None
        text = str(turn["content"])
        bank.append({
            "example_id": f"{_speaker_id(speaker)}:{turn['turn_id']}",
            "source_session": turn["session_id"],
            "source_turn_id": turn["turn_id"],
            "trigger": _classify_from_previous(previous),
            "previous_speaker": previous["speaker"] if previous else "",
            "previous_turn": previous["content"] if previous else "",
            "target_turn": text,
            "bubble_count": max(1, len(turn.get("message_indices", []))),
            "character_count": len(text),
            "contains_question": "?" in text,
            "contains_first_person": bool(_FIRST_PERSON_RE.search(text)),
            "contains_reflective_marker": bool(_REFLECTIVE_RE.search(text)),
        })
    if not bank:
        raise ValueError(f"no Ca target turns found for {speaker}")
    return bank


def summarize_behavior_bank(bank: list[dict[str, Any]]) -> dict[str, Any]:
    by_trigger: dict[str, dict[str, Any]] = {}
    for trigger in sorted({item["trigger"] for item in bank}):
        rows = [item for item in bank if item["trigger"] == trigger]
        by_trigger[trigger] = _behavior_stats(rows)
    return {"overall": _behavior_stats(bank), "by_trigger": by_trigger}


def summarize_visible_target_behavior(
    turns: list[dict[str, Any]], speaker: str
) -> dict[str, Any]:
    rows = []
    for turn in turns:
        if turn["speaker"].casefold() != speaker.casefold():
            continue
        text = str(turn["content"])
        rows.append({
            "bubble_count": max(1, len(turn.get("message_indices", []))),
            "character_count": len(text),
            "contains_question": "?" in text,
            "contains_first_person": bool(_FIRST_PERSON_RE.search(text)),
            "contains_reflective_marker": bool(_REFLECTIVE_RE.search(text)),
        })
    return _behavior_stats(rows) if rows else {"observed_turns": 0}


def retrieve_ca_behavior_examples(
    bank: list[dict[str, Any]],
    context_turns: list[dict[str, Any]],
    speaker: str,
    *,
    current_session: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    previous = _latest_non_target_turn(context_turns, speaker, current_session)
    trigger = classify_interaction_trigger(context_turns, speaker, current_session)
    previous_text = str(previous["content"]) if previous else ""
    previous_tokens = _tokens(previous_text)

    def rank(item: dict[str, Any]) -> tuple[float, str]:
        overlap = _jaccard(previous_tokens, _tokens(item["previous_turn"]))
        score = 4.0 * float(item["trigger"] == trigger) + 3.0 * overlap
        score += float(("?" in previous_text) == ("?" in item["previous_turn"]))
        score += float(bool(_FIRST_PERSON_RE.search(previous_text)) == bool(
            _FIRST_PERSON_RE.search(item["previous_turn"])
        )) * 0.5
        return (-round(score, 8), item["example_id"])

    selected = sorted(bank, key=rank)[: min(limit, len(bank))]
    return [
        {**item, "retrieval_rank": index + 1}
        for index, item in enumerate(selected)
    ]


def classify_interaction_trigger(
    context_turns: list[dict[str, Any]],
    speaker: str,
    current_session: str | None = None,
) -> str:
    return _classify_from_previous(
        _latest_non_target_turn(context_turns, speaker, current_session)
    )


def build_progressive_gate_manifest(
    prepared: list[dict[str, Any]]
) -> dict[str, list[str]]:
    cells: dict[tuple[str, str], list[str]] = {}
    sessions_by_speaker: dict[str, list[str]] = {}
    for item in prepared:
        speaker = item["speaker"]
        sessions = list(item["points"][0]["test_sessions"])
        sessions_by_speaker[speaker] = sessions
        for session in sessions:
            cells[(speaker, session)] = [
                f"{_speaker_id(speaker)}:{point['sample_id']}"
                for point in item["points"] if point["target_session"] == session
            ]

    cell_order: list[tuple[str, str]] = []
    speakers = [item["speaker"] for item in prepared]
    for round_index in range(3):
        for speaker_index, speaker in enumerate(speakers):
            sessions = sessions_by_speaker[speaker]
            cell_order.append((speaker, sessions[(speaker_index + round_index) % 3]))

    per_cell_order = {
        cell: _nested_position_order(ids) for cell, ids in cells.items()
    }
    first_round = [per_cell_order[cell][0] for cell in cell_order]
    first_two = first_round + [per_cell_order[cell][1] for cell in cell_order]
    first_four = list(first_two)
    for rank in (2, 3):
        first_four += [per_cell_order[cell][rank] for cell in cell_order]

    all_ids = [
        f"{_speaker_id(item['speaker'])}:{point['sample_id']}"
        for item in prepared for point in item["points"]
    ]
    prefix = list(first_four)
    prefix_set = set(prefix)
    full = prefix + [result_id for result_id in all_ids if result_id not in prefix_set]
    manifest = {
        "6": first_round[:6],
        "18": first_round[:18],
        "30": first_round,
        "60": first_two,
        "120": first_four,
        "519": full,
    }
    _validate_gate_manifest(manifest, all_ids)
    return manifest


def actor_structure_audit(message: str, plan: dict[str, Any]) -> dict[str, Any]:
    bubbles = [line.strip() for line in message.splitlines() if line.strip()]
    question_count = message.count("?")
    return {
        "planned_bubble_count": plan["bubble_count"],
        "observed_nonempty_lines": len(bubbles),
        "bubble_count_match": len(bubbles) == plan["bubble_count"],
        "planned_question": plan["question_plan"] != "none",
        "observed_question_marks": question_count,
        "question_permission_match": (
            question_count == 0 if plan["question_plan"] == "none" else question_count >= 1
        ),
        "character_count": len(message),
    }


def aggregate_v14_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"records": 0}
    orientations = Counter(row["alignment"]["orientation"] for row in results)
    planned_bubbles = Counter(row["message_plan"]["bubble_count"] for row in results)
    observed_multiline = sum("\n" in row["generated_message"] for row in results)
    gt_multiline = sum("\n" in row["ground_truth"] for row in results)
    return {
        "records": len(results),
        "orientation_counts": dict(orientations),
        "lambda_mean": round(statistics.mean(
            row["alignment"]["lambda_trace"] for row in results
        ), 6),
        "lambda_min": min(row["alignment"]["lambda_trace"] for row in results),
        "lambda_max": max(row["alignment"]["lambda_trace"] for row in results),
        "planned_bubble_counts": {str(key): value for key, value in sorted(planned_bubbles.items())},
        "actor_bubble_count_match_rate": round(statistics.mean(
            row["actor_structure_audit"]["bubble_count_match"] for row in results
        ), 6),
        "actor_question_permission_match_rate": round(statistics.mean(
            row["actor_structure_audit"]["question_permission_match"] for row in results
        ), 6),
        "candidate_multiline_rate": round(observed_multiline / len(results), 6),
        "ground_truth_multiline_rate": round(gt_multiline / len(results), 6),
    }


def _index_prepared(
    prepared: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    points: dict[str, dict[str, Any]] = {}
    speakers = {item["speaker"]: item for item in prepared}
    for item in prepared:
        for point in item["points"]:
            result_id = f"{_speaker_id(item['speaker'])}:{point['sample_id']}"
            points[result_id] = {**point, "speaker": item["speaker"]}
    return points, speakers


def _validate_v9_alignment(
    rows: list[dict[str, Any]],
    self_domains: dict[str, dict[str, Any]],
    point_index: dict[str, dict[str, Any]],
) -> None:
    if set(point_index) != {row["result_id"] for row in rows}:
        raise ValueError("V9 result IDs do not exactly match reconstructed 519-point protocol")
    for row in rows:
        point = point_index[row["result_id"]]
        if row["context_hash"] != point["history_hash"]:
            raise ValueError(f"V9 context hash mismatch for {row['result_id']}")
        if row["ground_truth"] != point["target_message"]:
            raise ValueError(f"V9 ground truth mismatch for {row['result_id']}")
        if row["self_domain_hash"] != stable_hash(self_domains[row["speaker"]]):
            raise ValueError(f"V9 Self Domain hash mismatch for {row['result_id']}")


def _validate_v9_source(
    predictions_path: Path,
    self_domains_path: Path,
    rows: list[dict[str, Any]],
    self_domains: dict[str, Any],
    *,
    enforce_canonical: bool,
) -> dict[str, Any]:
    predictions_hash = _sha256_file(predictions_path)
    if enforce_canonical and predictions_hash != EXPECTED_V9_PREDICTIONS_SHA256:
        raise ValueError(
            "V14 requires the canonical V9 predictions; "
            f"expected {EXPECTED_V9_PREDICTIONS_SHA256}, found {predictions_hash}"
        )
    if enforce_canonical and len(rows) != EXPECTED_RECORDS:
        raise ValueError(f"canonical V9 must contain {EXPECTED_RECORDS} rows")
    if len({row["result_id"] for row in rows}) != len(rows):
        raise ValueError("V9 predictions contain duplicate result IDs")
    expected_speakers = {item["speaker"] for item in REALTALK_PERSONA_SPLITS}
    if enforce_canonical and set(self_domains) != expected_speakers:
        raise ValueError("V9 Self Domains do not contain exactly the ten Table 8 speakers")
    source = {
        "implementation_commit": EXPECTED_V9_COMMIT,
        "predictions_path": str(predictions_path),
        "predictions_sha256": predictions_hash,
        "predictions_records": len(rows),
        "self_domains_path": str(self_domains_path),
        "self_domains_sha256": _sha256_file(self_domains_path),
        "self_domains_hash": stable_hash(self_domains),
    }
    run_manifest_path = predictions_path.parent / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = _load_json(run_manifest_path)
        actual_commit = run_manifest.get("implementation_repository_commit")
        if enforce_canonical and actual_commit != EXPECTED_V9_COMMIT:
            raise ValueError(
                f"V9 run manifest commit mismatch: expected {EXPECTED_V9_COMMIT}, "
                f"found {actual_commit}"
            )
        source["run_manifest_path"] = str(run_manifest_path)
        source["run_manifest_sha256"] = _sha256_file(run_manifest_path)
        source["run_manifest_commit"] = actual_commit
    return source


def _validate_v14_context(
    value: dict[str, Any], *, trigger: str, has_history: bool
) -> dict[str, Any]:
    plan = value["message_plan"]
    alignment = value["alignment"]
    if not has_history and plan["primary_move"] != "open":
        raise ValueError("empty visible history requires an open primary move")
    orientation = alignment["orientation"]
    lam = alignment["lambda_trace"]
    if orientation == "self-led" and lam > 0.45:
        raise ValueError("self-led orientation requires lambda_trace <= 0.45")
    if orientation == "balanced" and not 0.25 <= lam <= 0.75:
        raise ValueError("balanced orientation requires lambda_trace in [0.25,0.75]")
    if orientation == "partner-adaptive" and lam < 0.55:
        raise ValueError("partner-adaptive orientation requires lambda_trace >= 0.55")
    if trigger in {"after-question", "after-support-request"}:
        if orientation == "self-led" or lam < 0.4:
            raise ValueError(
                "direct partner question/support trigger requires a genuinely partner-shaped plan"
            )
    return value


def _v14_actor_self_domain(self_domain: dict[str, Any]) -> dict[str, Any]:
    view = _behavioral_self_domain(self_domain)
    view["stable_identity_context"] = self_domain["identity_context"]
    view["boundaries_and_uncertainty"] = self_domain["boundaries_and_uncertainty"]
    return view


def _behavior_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"observed_turns": 0}
    chars = [int(item["character_count"]) for item in rows]
    bubbles = [int(item["bubble_count"]) for item in rows]
    return {
        "observed_turns": len(rows),
        "mean_characters": round(statistics.mean(chars), 2),
        "median_characters": round(statistics.median(chars), 2),
        "mean_bubble_count": round(statistics.mean(bubbles), 3),
        "median_bubble_count": round(statistics.median(bubbles), 3),
        "multibubble_rate": round(statistics.mean(value > 1 for value in bubbles), 4),
        "question_rate": round(statistics.mean(
            bool(item["contains_question"]) for item in rows
        ), 4),
        "first_person_rate": round(statistics.mean(
            bool(item["contains_first_person"]) for item in rows
        ), 4),
        "reflective_marker_rate": round(statistics.mean(
            bool(item["contains_reflective_marker"]) for item in rows
        ), 4),
        "bubble_count_distribution": {
            str(key): value for key, value in sorted(Counter(bubbles).items())
        },
    }


def _classify_from_previous(previous: dict[str, Any] | None) -> str:
    if previous is None:
        return "session-opening"
    text = str(previous["content"]).strip()
    lower = text.casefold()
    if _SUPPORT_RE.search(text):
        return "after-support-request"
    if "?" in text or _QUESTION_OPEN_RE.search(text):
        return "after-question"
    if _CLOSING_RE.search(text):
        return "after-closing"
    if _AFFECT_RE.search(text):
        return "after-explicit-affect"
    if _FIRST_PERSON_RE.search(text):
        return "after-partner-disclosure"
    if any(token in lower for token in ("hello", "hi ", "hey ", "good morning", "good evening")):
        return "after-greeting"
    return "after-partner-statement"


def _latest_non_target_turn(
    context_turns: list[dict[str, Any]],
    speaker: str,
    current_session: str | None = None,
) -> dict[str, Any] | None:
    return next(
        (
            turn for turn in reversed(context_turns)
            if turn["speaker"].casefold() != speaker.casefold()
            and (current_session is None or turn["session_id"] == current_session)
        ),
        None,
    )


def _latest_partner_turn(
    context_turns: list[dict[str, Any]], partner: str, current_session: str | None = None
) -> dict[str, Any] | None:
    return next(
        (
            turn for turn in reversed(context_turns)
            if turn["speaker"].casefold() == partner.casefold()
            and (current_session is None or turn["session_id"] == current_session)
        ),
        None,
    )


def _nested_position_order(values: list[str]) -> list[str]:
    if len(values) < 4:
        raise ValueError("every REALTALK speaker-session cell needs at least four targets")
    selected = [len(values) // 2]
    while len(selected) < len(values):
        remaining = [index for index in range(len(values)) if index not in selected]
        best = max(
            remaining,
            key=lambda index: (min(abs(index - used) for used in selected), -index),
        )
        selected.append(best)
    return [values[index] for index in selected]


def _validate_gate_manifest(
    manifest: dict[str, list[str]], all_ids: list[str]
) -> None:
    previous: set[str] = set()
    for gate in GATES:
        values = manifest[str(gate)]
        if len(values) != gate or len(values) != len(set(values)):
            raise ValueError(f"gate {gate} must contain exactly {gate} unique IDs")
        current = set(values)
        if not previous.issubset(current):
            raise ValueError(f"gate {gate} is not nested")
        previous = current
    if set(manifest["519"]) != set(all_ids):
        raise ValueError("gate 519 does not contain the complete protocol")


def _preflight(
    output_dir: Path, backend: ChatBackend, config: V14Config
) -> dict[str, Any]:
    path = output_dir / "preflight.json"
    if path.exists():
        existing = _load_json(path)
        if existing.get("model") == config.model and existing.get("succeeded") is True:
            return existing
    available_fn = getattr(backend, "available_models", None)
    available = list(available_fn()) if callable(available_fn) else [backend.model]
    if config.model not in available:
        raise ValueError(f"configured credential cannot access {config.model}")
    result = backend.chat(
        "Return exactly READY.",
        "READY",
        temperature=0.0,
        top_p=0.9,
        max_tokens=8,
        enable_thinking=False,
    )
    if "READY" not in result.content.upper():
        raise ValueError("model preflight failed")
    value = {
        "checked_at_utc": _now(),
        "model": backend.model,
        "succeeded": True,
        "thinking_enabled": False,
    }
    _write_json(path, value)
    return value


def _prompt_hashes() -> dict[str, str]:
    return {
        "decision_system": stable_hash(DECISION_SYSTEM_PROMPT),
        "decision_user": stable_hash(DECISION_USER_TEMPLATE),
        "actor_system": stable_hash(ACTOR_SYSTEM_TEMPLATE),
        "actor_user": stable_hash(ACTOR_USER_TEMPLATE),
    }


def _validate_config(config: V14Config) -> None:
    if config.gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}")
    if config.model != MODEL:
        raise ValueError(f"V14 model is frozen to {MODEL}")
    if config.operation_max_attempts != 3:
        raise ValueError("operation attempts are fixed at three")
    if config.fresh == config.resume:
        raise ValueError("exactly one of --fresh or --resume is required")
    if not config.v9_predictions or not config.v9_self_domains:
        raise ValueError("V9 predictions and Self Domains are required")


def _clear_outputs(output_dir: Path) -> None:
    for name in (
        "checkpoint.json", "checkpoint.json.tmp", "raw_responses.jsonl",
        "predictions.jsonl", "unresolved_errors.json", "dataset_manifest.json",
        "progressive_gate_manifest.json", "behavior_banks.json",
        "behavior_summaries.json", "diagnostics.json", "run_manifest.json",
        "GENERATION_COMPLETE", "preflight.json",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z']+", text) if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


_FIRST_PERSON_RE = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|me|my|mine|myself)\b", re.I)
_REFLECTIVE_RE = re.compile(
    r"\b(?:I think|I feel|I guess|I believe|I realize|I wonder|for me|makes me|because)\b",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"\b(?:help me|need help|what should I do|advice|support|can you help)\b", re.I
)
_QUESTION_OPEN_RE = re.compile(
    r"^(?:who|what|when|where|why|how|do|does|did|are|is|can|could|would|will|have|has)\b",
    re.I,
)
_CLOSING_RE = re.compile(
    r"\b(?:good night|goodnight|talk later|see you|bye|gotta go|have a good)\b", re.I
)
_AFFECT_RE = re.compile(
    r"\b(?:sad|upset|worried|anxious|afraid|angry|frustrated|lonely|hurt|happy|excited|glad|stressed|tired)\b",
    re.I,
)


def parse_args() -> V14Config:
    parser = argparse.ArgumentParser(description="Run REALTALK Ours V14 progressive replay")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--v9-predictions", required=True)
    parser.add_argument("--v9-self-domains", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate", type=int, choices=GATES, required=True)
    parser.add_argument("--model", default=MODEL, choices=(MODEL,))
    parser.add_argument("--model-call-timeout-seconds", type=int, default=240)
    parser.add_argument("--no-canonical-v9-check", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return V14Config(
        dataset_dir=args.dataset_dir,
        v9_predictions=args.v9_predictions,
        v9_self_domains=args.v9_self_domains,
        output_dir=args.output_dir,
        gate=args.gate,
        model=args.model,
        model_call_timeout_seconds=args.model_call_timeout_seconds,
        fresh=args.fresh,
        resume=args.resume,
        enforce_canonical_v9=not args.no_canonical_v9_check,
    )


def main() -> None:
    print(json.dumps(run_v14(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

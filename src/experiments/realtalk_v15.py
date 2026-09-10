"""REALTALK V15 behavior controller with Ca priors and causal Cb evidence."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exp1_protocol import select_realtalk_splits, stable_hash
from .operation_checkpoint import OperationCheckpoint
from .personaemp.client import ChatBackend
from .realtalk_ours import (
    RealTalkOursConfig,
    _backend_from_env,
    _checkpoint_unresolved,
    _failure,
    _json,
    _prepare_dataset,
    _repository_commit,
    _safe_host,
    _structured_call,
    _text_call,
    _turns_with_session_boundaries,
    _write_json,
    _write_jsonl,
)
from .realtalk_v14 import (
    _index_prepared,
    _information_question_count,
    _latest_partner_turn,
    _load_json,
    _load_jsonl,
    _preflight,
    _validate_v9_alignment,
    _validate_v9_source,
    build_ca_behavior_bank,
    build_progressive_gate_manifest,
    classify_interaction_trigger,
    summarize_behavior_bank,
    summarize_visible_target_behavior,
)
from .realtalk_v15_schemas import DECISION_SCHEMA, QUESTION_ACTS, normalize_v15_decision


MODEL = "deepseek-v4-flash"
PROTOCOL = "realtalk_task1_ours_v15_11_cb_posterior_controller"
GATES = (6, 18, 30, 60, 120, 519)


CONTROLLER_SYSTEM_PROMPT = """You are the private Behavior Controller for a persona-simulation actor.
Predict the target person's next conversational turn, not an ideal assistant response.

Use the fixed Self Domain as the target's cross-partner identity and voice. Read the complete visible Cb
history as the current reality. Use the Ca behavior summary only as a weak cold-start prior. The visible Cb
behavior summary describes how this person has actually behaved with the current partner before this point;
when it has substantial evidence and conflicts with Ca, prefer the current Cb pattern. Always consider the
reported observation counts instead of treating a small sample as stable.

Use at most two User Domain facts by selecting their fact_id exactly from the indexed whitelist; do not copy
or rewrite the fact text. Select facts only when they directly affect this turn. The User Domain conditions
the interaction but never replaces the target's identity. Do not copy an old
Ca event or partner fact as a current target fact. Current autobiographical detail must be supported by the
visible Cb history; otherwise keep self-expression low-specificity. In particular, never mirror the partner's
current location, weather, activity, plan, health, work situation, or possession into a first-person target
claim merely to create reciprocity. When no target-owned evidence supports such a current fact, acknowledge,
answer, or state a stable low-specificity preference instead.

When the partner proposes a new activity, plan, identity, or creative idea for the target, treat it as a new
suggestion. The target may accept, reject, or tentatively consider it, but must not claim to have previously
considered, planned, performed, or possessed it unless the target-owned evidence explicitly supports that
past or existing fact. A future-facing reaction such as considering the idea is not evidence of prior history.

First identify the current conversational obligation. A merged partner turn can contain several original
chat bubbles in chronological order. Judge the still-active obligation from that sequence, especially its
closing social move, rather than classifying the whole turn from any earlier question mark. An earlier
question can be followed or superseded by praise, acknowledgement, correction, or a closing statement; do
not mechanically answer it when the latest move makes a short social response the natural continuation.
When the closing bubbles recommend something and then praise or encourage the target, classify the active
partner act as praise-or-encouragement with an acknowledge obligation unless a later explicit question still
requires an answer. Do not classify praise directed at the target as a support request from the partner.
Before accepting any question premise about the target, verify it against the Self Domain and the target's
own visible Cb statements. If the premise is unsupported or conflicts with target-owned evidence, do not
adopt it as fact: answer only the supported part, correct it naturally, or ask one clarification when needed.
Then make one coherent turn plan containing one to six chat-bubble units. A unit has exactly one communicative
act. Multiple units may repeat an act when the target's chat rhythm naturally splits one contribution across
bubbles. Do not add acknowledgement, explanation,
self-disclosure, warmth, or a question merely to make the turn comprehensive. Conversely, do not compress a
person who commonly combines compatible moves when the current exchange supports them.

Treat one unit as the default. Use multiple units only when both the observed bubble statistics and the
current exchange support separate conversational moves. Do not create an extra bubble merely because a
secondary act could be relevant; omit that act when it is not necessary. Match the target's demonstrated
message rhythm instead of maximizing the number of acts or bubbles.

Only clarification-question, follow-up-question, and reciprocal-question acts may ask an information-seeking
question. Give every question act one concrete question_target; every other act must have an empty
question_target. Use clarification-question only to ask for missing meaning, follow-up-question only to ask
further about the partner's current topic, and reciprocal-question only to return a conversational slot after
the target's own answer or disclosure. For every question act, begin content_slot with "ask" and describe
exactly one question to ask. The question_target must name that one information slot, not merely the partner;
never combine a name question with an origin question or join two interrogative clauses with "and". Never
place an acknowledgement, answer, or opinion in a question unit. A
partner's direct question normally creates a respond obligation, but the plan may include a reciprocal
question only when the current slot is genuinely symmetric and the visible history supports returning that
same slot. Aggregate question_rate, including an after-question rate, is only an upper style tendency: it does
not by itself license a reciprocal question after an answer. If one-bubble evidence favors one unit, keep the
answer and omit an optional reciprocal question; never squeeze the question into the answer unit. Ordinary
factual or casual messages do not automatically require psychological interpretation or emotional support.

The topic-shift-statement act is declarative. If a topic transition is made by asking the partner something,
use follow-up-question or reciprocal-question instead.

Choose disclosure depth, relationship register, length, and tone from the target's observed behavior and the
current relationship. lambda_trace records how strongly partner-facing evidence changes this turn relative to
the stable Self prior. It is an audit trace, not a reward. Do not force it into a preset interval. Name the
source of adaptation and the plan dimensions actually affected. Orientation describes the overall stance of
the turn; it is not a numeric bucket for lambda_trace. A balanced or partner-adaptive turn may still have zero
departure when the current need already matches the stable Self prior. If lambda_trace is zero, return no
affected_dimensions. If it is nonzero, name at least one dimension whose change is visible in the structured
plan rather than explained as an abstract ideal.

Return only the strict schema. Do not mention evaluation metrics or reconstruct any known reference answer."""


CONTROLLER_USER_TEMPLATE = """TARGET SPEAKER: {speaker}
CURRENT PARTNER: {partner}

FIXED V9 SELF DOMAIN:
{self_domain}

AUTHORITATIVE TARGET-OWNED IDENTITY CONTEXT:
{target_identity_context}

CURRENT FIVE-LAYER USER DOMAIN:
{user_domain}

REAL CAUSAL HISTORY BEFORE THE TARGET TURN:
{history}

VISIBLE TARGET-OWNED Cb STATEMENTS BEFORE THIS TURN:
{target_owned_history}

LATEST PARTNER TURN IN THE CURRENT SESSION:
{latest_partner_turn}

CURRENT INTERACTION TRIGGER: {current_trigger}

TARGET'S AGGREGATE CA BEHAVIOR PRIOR (statistics only; no source text):
{ca_behavior_summary}

TARGET'S VISIBLE Cb BEHAVIOR BEFORE THIS POINT:
{cb_behavior_summary}

INDEXED USER DOMAIN ACTIVATION WHITELIST:
{activation_whitelist}

Plan the next turn as {speaker}. Ca supplies a weak cross-partner prior; the complete visible history and
current Cb evidence determine the present interaction. Return only selected fact_id values from the
whitelist, or return an empty relevant_user_domain array."""


ACTOR_SYSTEM_TEMPLATE = """You are {speaker}. Continue the conversation.
Act as the person represented by the private Self Domain.
Follow the private turn plan naturally.
Output only the message, not the speaker name."""


ACTOR_USER_TEMPLATE = """REAL CONVERSATION HISTORY BEFORE YOUR NEXT TURN:
{history}

PRIVATE BEHAVIORAL SELF DOMAIN:
{self_domain}

PRIVATE TURN PLAN:
{turn_plan}

Write the next conversational turn as {speaker}. Realize each turn_unit as exactly one non-empty chat bubble,
in order, separated by newline characters. Do not number or label bubbles. Preserve the intended act and
content_slot of each unit without adding another act. Only units whose act is clarification-question,
follow-up-question, or reciprocal-question may contain an information-seeking question, and each such unit
must ask exactly one question about its question_target. Realize the question act itself; do not substitute a
declarative sentence from content_slot. Use one direct grammatical question and exactly one question mark;
begin directly with that question. Do not restate or confirm the partner's message as a preliminary question,
do not prepend a separate "Really?" or "What about you?", and do not add an alternative second question.
Other units must not ask a question.

Match the planned disclosure depth, relationship register, length band, and tone while keeping the target's
natural voice. The full history is the only source of current facts. Do not turn the partner's workplace,
location, weather, activity, feeling, plan, preference, health, possession, or experience into the target's
own current fact. Do not mirror a partner statement by adding "me too", "here too", or an unsupported
first-person version of it. A new partner suggestion may become a tentative future choice, but never turn it
into an unsupported claim that the target already considered, planned, did, or owned it. Do not mention the
plan, domains, statistics, or internal reasoning."""


ACTOR_REPAIR_TEMPLATE = """

STRUCTURE REPAIR REQUIRED
The prior draft did not follow the frozen turn plan:
{errors}

Prior draft:
{draft}

Generate the turn again with the same meaning and the same frozen turn units. Correct only the reported
format or question-permission errors. A question unit must contain one direct grammatical question with one
question mark. Delete any preliminary confirmation question and keep only the one question that directly
targets question_target; never emit a sequence such as "Really? Do you...?" or "What about you? Do you...?".
Do not add facts, change the acts, or redesign the plan."""


@dataclass(frozen=True)
class V15Config:
    dataset_dir: str = "dataset"
    v9_predictions: str = ""
    v9_self_domains: str = ""
    output_dir: str = "data/realtalk_v15_gate6"
    gate: int = 6
    model: str = MODEL
    operation_max_attempts: int = 3
    model_call_timeout_seconds: int = 240
    fresh: bool = False
    resume: bool = False
    enforce_canonical_v9: bool = True


def run_v15(config: V15Config, backend: ChatBackend | None = None) -> dict[str, Any]:
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

    ca_summaries = {
        item["speaker"]: summarize_behavior_bank(
            build_ca_behavior_bank(item["profile"]["turns"], item["speaker"])
        )
        for item in prepared
    }

    backend = backend or _backend_from_env(config.model)
    if getattr(backend, "model", None) != config.model:
        raise ValueError(
            f"V15 requires model {config.model!r}; got {getattr(backend, 'model', None)!r}"
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
        "gate_manifest": gate_manifest,
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
        latest_partner = _latest_partner_turn(
            point["context_turns"], speaker_data["partner"], point["target_session"]
        )
        activation_facts = build_v15_activation_whitelist(v9["user_domain"])
        try:
            decision_envelope = _structured_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"v15_controller:{result_id}",
                system_prompt=CONTROLLER_SYSTEM_PROMPT,
                user_prompt=CONTROLLER_USER_TEMPLATE.format(
                    speaker=point["speaker"],
                    partner=speaker_data["partner"],
                    self_domain=_json(self_domain),
                    target_identity_context=_json(self_domain["identity_context"]),
                    user_domain=_json(v9["user_domain"]),
                    history=_turns_with_session_boundaries(point["context_turns"]),
                    target_owned_history=_target_owned_history_text(
                        point["context_turns"], point["speaker"]
                    ),
                    latest_partner_turn=(
                        _turns_with_session_boundaries([latest_partner])
                        if latest_partner else "NONE"
                    ),
                    current_trigger=classify_interaction_trigger(
                        point["context_turns"], point["speaker"], point["target_session"]
                    ),
                    ca_behavior_summary=_json(ca_summaries[point["speaker"]]),
                    cb_behavior_summary=_json(
                        summarize_visible_target_behavior(
                            point["context_turns"], point["speaker"]
                        )
                    ),
                    activation_whitelist=_v15_activation_whitelist_text(activation_facts),
                ),
                schema=DECISION_SCHEMA,
                normalizer=lambda value, facts=activation_facts: (
                    resolve_v15_profile_activation(normalize_v15_decision(value), facts)
                ),
                max_tokens=1600,
                max_attempts=config.operation_max_attempts,
                raw_audit=raw_audit,
                enable_thinking=False,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
            decision = decision_envelope["data"]
            actor_result = _run_actor_with_contract_retries(
                checkpoint=checkpoint,
                backend=backend,
                result_id=result_id,
                speaker=point["speaker"],
                history=_turns_with_session_boundaries(point["context_turns"]),
                self_domain=_v15_actor_self_domain(self_domain),
                turn_plan=decision["turn_plan"],
                raw_audit=raw_audit,
                max_attempts=config.operation_max_attempts,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
            fact_ownership_audit = _fact_ownership_audit(
                actor_result["generated_message"], latest_partner,
                point["context_turns"], point["speaker"], self_domain
            )
            if fact_ownership_audit["warning"]:
                raise ValueError(
                    "fact ownership audit found unsupported partner-to-target transfer: "
                    f"{fact_ownership_audit['overlap_tokens']}"
                )
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
                "generated_message": actor_result["generated_message"],
                "situation": decision["situation"],
                "relevant_user_domain": decision["relevant_user_domain"],
                "alignment": decision["alignment"],
                "turn_plan": decision["turn_plan"],
                "ca_behavior_summary": ca_summaries[point["speaker"]],
                "visible_cb_behavior": summarize_visible_target_behavior(
                    point["context_turns"], point["speaker"]
                ),
                "actor_structure_audit": actor_result["audit"],
                "fact_ownership_audit": fact_ownership_audit,
                "operation_audit": {
                    "controller": decision_envelope["audit"],
                    "actor": actor_result["operation_audits"],
                },
            }
            checkpoint.data["failures"].pop(f"sample:{result_id}", None)
            checkpoint.store_result(result_id, result)
        except Exception as exc:
            checkpoint.store_excluded_result(
                result_id, _failure("v15_sample", point["speaker"], result_id, exc)
            )

    result_index = {row["result_id"]: row for row in checkpoint.result_values()}
    results = [result_index[result_id] for result_id in selected_ids if result_id in result_index]
    unresolved = _checkpoint_unresolved(checkpoint)
    diagnostics = aggregate_v15_diagnostics(results)
    selection = {
        "mode": "progressive_gate",
        "gate": config.gate,
        "selected_result_count": len(selected_ids),
        "selected_result_ids_sha256": stable_hash(selected_ids),
    }
    _write_jsonl(output_dir / "predictions.jsonl", results)
    _write_jsonl(
        output_dir / "v9_baseline_predictions.jsonl",
        [v9_index[result_id] for result_id in selected_ids],
    )
    _write_json(output_dir / "unresolved_errors.json", unresolved)
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(output_dir / "progressive_gate_manifest.json", gate_manifest)
    _write_json(output_dir / "selection_manifest.json", selection)
    _write_json(output_dir / "ca_behavior_summaries.json", ca_summaries)
    _write_json(output_dir / "diagnostics.json", diagnostics)
    _write_json(output_dir / "run_manifest.json", {
        "created_at_utc": _now(),
        "protocol": PROTOCOL,
        "implementation_repository_commit": _repository_commit(),
        "model": backend.model,
        "thinking_enabled": {"controller": False, "actor": False},
        "training_or_finetuning": False,
        "frozen_v9_upstream": True,
        "v9_decision_visible": False,
        "v9_generated_text_visible": False,
        "regenerated_stages": ["controller", "actor"],
        "frozen_stages": ["self_domain", "user_domain"],
        "ca_raw_examples_visible": False,
        "ca_statistics_visible": True,
        "visible_cb_statistics_causal": True,
        "semantic_verification_enabled": False,
        "conditional_actor_structure_retry": True,
        "omega_enabled": False,
        "future_user_state_enabled": False,
        "history_compression_enabled": False,
        "history_truncation_enabled": False,
        "source_v9": source,
        "gate": config.gate,
        "selection": selection,
        "prompt_hashes": _prompt_hashes(),
        "schema_hashes": {"controller": stable_hash(DECISION_SCHEMA)},
        "decoding": {
            "controller": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1600},
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


def _run_actor_with_contract_retries(
    *,
    checkpoint: OperationCheckpoint,
    backend: ChatBackend,
    result_id: str,
    speaker: str,
    history: str,
    self_domain: dict[str, Any],
    turn_plan: dict[str, Any],
    raw_audit: Path,
    max_attempts: int,
    hard_timeout_seconds: int,
) -> dict[str, Any]:
    base_prompt = ACTOR_USER_TEMPLATE.format(
        speaker=speaker,
        history=history,
        self_domain=_json(self_domain),
        turn_plan=_json(turn_plan),
    )
    repair = ""
    operation_audits = []
    last_audit: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        envelope = _text_call(
            checkpoint=checkpoint,
            backend=backend,
            operation_key=f"v15_actor:{result_id}:attempt:{attempt}",
            system_prompt=ACTOR_SYSTEM_TEMPLATE.format(speaker=speaker),
            user_prompt=base_prompt + repair,
            speaker=speaker,
            max_attempts=1,
            raw_audit=raw_audit,
            enable_thinking=False,
            hard_timeout_seconds=hard_timeout_seconds,
        )
        generated = envelope["data"]
        audit = actor_structure_audit(generated, turn_plan, speaker)
        operation_audits.append(envelope["audit"])
        last_audit = audit
        if audit["blocking_contract_passed"]:
            return {
                "generated_message": generated,
                "audit": {**audit, "actor_attempts": attempt},
                "operation_audits": operation_audits,
            }
        repair = ACTOR_REPAIR_TEMPLATE.format(
            errors="; ".join(audit["blocking_errors"]),
            draft=generated,
        )
    assert last_audit is not None
    raise ValueError(
        "actor failed frozen structure contract after "
        f"{max_attempts} attempts: {last_audit['blocking_errors']}"
    )


def actor_structure_audit(
    message: str, turn_plan: dict[str, Any], speaker: str
) -> dict[str, Any]:
    bubbles = [line.strip() for line in message.splitlines() if line.strip()]
    units = turn_plan["turn_units"]
    expected_questions = sum(unit["act"] in QUESTION_ACTS for unit in units)
    observed_questions = _information_question_count(message)
    compound_questions = sum(
        bool(re.search(
            r"\band\s+(?:if|whether|what|where|when|why|how|who|do|does|did|is|are|has|have|can|could|would|will)\b",
            bubble,
            flags=re.I,
        ))
        for bubble in bubbles if "?" in bubble
    )
    role_label = bool(re.match(
        rf"^(?:{re.escape(speaker)}|assistant|speaker|target|response|message)\s*:",
        message.strip(),
        flags=re.I,
    ))
    blocking_errors = []
    if len(bubbles) != len(units):
        blocking_errors.append(
            f"expected {len(units)} non-empty bubbles, found {len(bubbles)}"
        )
    if observed_questions != expected_questions:
        blocking_errors.append(
            f"expected {expected_questions} information questions, found {observed_questions}"
        )
    if compound_questions:
        blocking_errors.append(
            f"found {compound_questions} question bubbles combining multiple information slots"
        )
    if role_label:
        blocking_errors.append("output contains a role or speaker label")
    return {
        "planned_bubbles": len(units),
        "observed_nonempty_lines": len(bubbles),
        "bubble_count_match": len(bubbles) == len(units),
        "planned_information_questions": expected_questions,
        "observed_information_questions": observed_questions,
        "question_permission_match": observed_questions == expected_questions,
        "compound_question_bubbles": compound_questions,
        "role_label_leak": role_label,
        "character_count": len(message),
        "length_band": turn_plan["length_band"],
        "blocking_errors": blocking_errors,
        "blocking_contract_passed": not blocking_errors,
    }


def _fact_ownership_audit(
    message: str,
    latest_partner: dict[str, Any] | None,
    context_turns: list[dict[str, Any]],
    speaker: str,
    self_domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if latest_partner is None:
        return {"status": "no-latest-partner-turn", "warning": False}
    partner_text = str(latest_partner["content"])
    partner_tokens = _distinctive_tokens(partner_text)
    previous_target_turns = [
        str(turn["content"]) for turn in context_turns
        if turn["speaker"].casefold() == speaker.casefold()
    ]
    previous_target = " ".join(previous_target_turns)
    stable_self = _json(self_domain.get("identity_context", {})) if self_domain else ""
    prior_tokens = _distinctive_tokens(previous_target + " " + stable_self)
    # A merged turn is one source contribution: concepts stated in separate bubbles of
    # that turn can still be mirrored as one invented target relation. Preserve target
    # turn boundaries so unrelated concepts from older turns do not falsely support it.
    partner_concept_sets = [_ownership_concepts(partner_text)]
    prior_concept_sets = [
        concepts for text in previous_target_turns
        if (concepts := _ownership_concepts(text))
    ] + _clause_concept_sets(stable_self)
    suspicious_clauses = []
    for clause in re.split(r"(?:[.!]+|\n+)", message):
        clause = clause.strip()
        if not clause or "?" in clause:
            continue
        first_person = bool(
            re.search(r"\b(?:i|i'm|i've|i'd|me|my|mine)\b", clause, re.I)
        )
        if not first_person:
            continue
        if _is_external_opinion_clause(clause) or _is_tentative_future_reaction(clause):
            continue
        overlap = sorted((_distinctive_tokens(clause) & partner_tokens) - prior_tokens)
        clause_concepts = _ownership_concepts(clause)
        partner_concepts = set().union(*partner_concept_sets) if partner_concept_sets else set()
        unsupported_concepts = sorted(
            concept for concept in clause_concepts & partner_concepts
            if not any(concept in prior for prior in prior_concept_sets)
        )
        mirrored_pairs = sorted(
            "+".join(sorted(pair))
            for pair in _concept_pairs(clause_concepts)
            if any(pair.issubset(partner) for partner in partner_concept_sets)
            and not any(pair.issubset(prior) for prior in prior_concept_sets)
        )
        explicit_mirroring = bool(re.search(
            r"\b(?:too|also|same|as\s+well|me\s+too|here\s+too)\b",
            clause,
            re.I,
        ))
        backdated_claim = _has_backdated_claim_language(clause)
        if (
            (len(overlap) >= 2 and backdated_claim)
            or unsupported_concepts
            or mirrored_pairs
        ):
            suspicious_clauses.append({
                "clause": clause[:300],
                "overlap_tokens": overlap[:20],
                "unsupported_concepts": unsupported_concepts,
                "mirrored_concept_pairs": mirrored_pairs,
                "explicit_mirroring_language": explicit_mirroring,
                "backdated_claim_language": backdated_claim,
            })
    suspicious = sorted({
        token for item in suspicious_clauses for token in item["overlap_tokens"]
    })
    warning = bool(suspicious_clauses)
    return {
        "status": "blocking-manual-review-required" if warning else "passed",
        "warning": warning,
        "overlap_tokens": suspicious[:20],
        "unsupported_concepts": sorted({
            concept
            for item in suspicious_clauses
            for concept in item["unsupported_concepts"]
        }),
        "mirrored_concept_pairs": sorted({
            pair
            for item in suspicious_clauses
            for pair in item["mirrored_concept_pairs"]
        }),
        "suspicious_self_clauses": suspicious_clauses,
        "automatic_rewrite": False,
        "requires_manual_review_when_warning": True,
    }


def _distinctive_tokens(text: str) -> set[str]:
    stop = {
        "about", "after", "again", "also", "because", "been", "being", "could",
        "from", "have", "just", "really", "that", "their", "there", "these", "they",
        "this", "those", "what", "when", "where", "which", "with", "would", "your",
        "glad", "it's",
    }
    return {
        token for token in re.findall(r"[a-z][a-z'-]{3,}", text.casefold())
        if token not in stop
    }


def _is_external_opinion_clause(clause: str) -> bool:
    remainder = re.sub(
        r"^\s*(?:i\s+(?:think|believe|guess|feel)|in\s+my\s+opinion|to\s+me)\b[, ]*",
        "",
        clause,
        flags=re.I,
    )
    if remainder == clause:
        return False
    return not bool(re.search(r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine)\b", remainder, re.I))


def _is_tentative_future_reaction(clause: str) -> bool:
    if _has_backdated_claim_language(clause):
        return False
    return bool(re.search(
        r"\b(?:i'll\s+(?:think|consider|try|look|check)|"
        r"i\s+(?:might|may|could|would)\s+(?:consider|try|look|check)|"
        r"i'm\s+(?:considering|thinking\s+about))\b",
        clause,
        re.I,
    ))


def _has_backdated_claim_language(clause: str) -> bool:
    return bool(
        re.search(r"\b(?:already|before|used\s+to|for\s+a\s+while)\b", clause, re.I)
        or re.search(
            r"\b(?:i've|i\s+have)\s+(?:previously\s+)?(?:considered|planned|"
            r"been\s+planning|thought\s+about|started|done|tried|owned)\b",
            clause,
            re.I,
        )
    )


_OWNERSHIP_CONCEPT_PATTERNS = {
    "location:new-york": r"\b(?:new\s+york|nyc)\b",
    "weather:cold": r"\b(?:cold|freez(?:e|es|ing)|chill(?:y|ier|iest)?)\b",
    "place:office": r"\b(?:office|workplace)\b",
}


def _ownership_concepts(text: str) -> set[str]:
    return {
        concept for concept, pattern in _OWNERSHIP_CONCEPT_PATTERNS.items()
        if re.search(pattern, text, flags=re.I)
    }


def _clause_concept_sets(text: str) -> list[set[str]]:
    return [
        concepts
        for clause in re.split(r"(?:[.!?]+|\n+)", text)
        if (concepts := _ownership_concepts(clause))
    ]


def _concept_pairs(concepts: set[str]) -> set[frozenset[str]]:
    ordered = sorted(concepts)
    return {
        frozenset((ordered[left], ordered[right]))
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    }


def _target_owned_history_text(
    context_turns: list[dict[str, Any]], speaker: str
) -> str:
    target_turns = [
        turn for turn in context_turns
        if turn["speaker"].casefold() == speaker.casefold()
    ]
    return (
        _turns_with_session_boundaries(target_turns)
        if target_turns else "NONE: the target has not spoken in visible Cb history yet."
    )


def _v15_actor_self_domain(self_domain: dict[str, Any]) -> dict[str, Any]:
    """Project the frozen domain to identity and expression fields visible to V15 Actor."""
    return {
        "identity_context": self_domain["identity_context"],
        "communication_signature": self_domain["communication_signature"],
        "boundaries_and_uncertainty": self_domain["boundaries_and_uncertainty"],
        "observable_statistics": self_domain["observable_statistics"],
    }


def aggregate_v15_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"records": 0}
    orientations = Counter(row["alignment"]["orientation"] for row in results)
    sources = Counter(row["alignment"]["adaptation_source"] for row in results)
    attempts = [row["actor_structure_audit"]["actor_attempts"] for row in results]
    return {
        "records": len(results),
        "orientation_counts": dict(orientations),
        "adaptation_source_counts": dict(sources),
        "lambda_mean": round(statistics.mean(row["alignment"]["lambda_trace"] for row in results), 6),
        "lambda_min": min(row["alignment"]["lambda_trace"] for row in results),
        "lambda_max": max(row["alignment"]["lambda_trace"] for row in results),
        "mean_turn_units": round(statistics.mean(len(row["turn_plan"]["turn_units"]) for row in results), 6),
        "candidate_multibubble_rate": round(statistics.mean("\n" in row["generated_message"] for row in results), 6),
        "ground_truth_multibubble_rate": round(statistics.mean("\n" in row["ground_truth"] for row in results), 6),
        "actor_structure_pass_rate": round(statistics.mean(row["actor_structure_audit"]["blocking_contract_passed"] for row in results), 6),
        "actor_retry_rate": round(statistics.mean(value > 1 for value in attempts), 6),
        "actor_mean_attempts": round(statistics.mean(attempts), 6),
        "fact_ownership_warning_count": sum(row["fact_ownership_audit"]["warning"] for row in results),
    }


def _prompt_hashes() -> dict[str, str]:
    return {
        "controller_system": stable_hash(CONTROLLER_SYSTEM_PROMPT),
        "controller_user": stable_hash(CONTROLLER_USER_TEMPLATE),
        "actor_system": stable_hash(ACTOR_SYSTEM_TEMPLATE),
        "actor_user": stable_hash(ACTOR_USER_TEMPLATE),
        "actor_repair": stable_hash(ACTOR_REPAIR_TEMPLATE),
    }


def build_v15_activation_whitelist(user_domain: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "fact_id": f"{layer}:{index}",
            "layer": layer,
            "value": fact["value"],
        }
        for layer in ("core", "regulation", "cognition", "identity", "behavior")
        for index, fact in enumerate(user_domain[layer])
    ]


def _v15_activation_whitelist_text(facts: list[dict[str, str]]) -> str:
    return "NONE (relevant_user_domain must be [])" if not facts else _json(facts)


def resolve_v15_profile_activation(
    decision: dict[str, Any], facts: list[dict[str, str]]
) -> dict[str, Any]:
    available = {fact["fact_id"]: fact for fact in facts}
    selected_ids = [fact["fact_id"] for fact in decision["relevant_user_domain"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("relevant_user_domain must not repeat fact_id values")
    unknown = sorted(set(selected_ids) - set(available))
    if unknown:
        raise ValueError(f"decision activated unknown User Domain fact IDs: {unknown}")
    decision["relevant_user_domain"] = [dict(available[fact_id]) for fact_id in selected_ids]
    return decision


def _validate_config(config: V15Config) -> None:
    if config.gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}")
    if config.model != MODEL:
        raise ValueError(f"V15 model is frozen to {MODEL}")
    if config.operation_max_attempts != 3:
        raise ValueError("operation attempts are fixed at three")
    if config.fresh == config.resume:
        raise ValueError("exactly one of --fresh or --resume is required")
    if not config.v9_predictions or not config.v9_self_domains:
        raise ValueError("V9 predictions and Self Domains are required")


def _clear_outputs(output_dir: Path) -> None:
    for name in (
        "checkpoint.json", "checkpoint.json.tmp", "raw_responses.jsonl",
        "predictions.jsonl", "v9_baseline_predictions.jsonl",
        "unresolved_errors.json", "dataset_manifest.json",
        "progressive_gate_manifest.json", "selection_manifest.json",
        "ca_behavior_summaries.json", "diagnostics.json", "run_manifest.json",
        "GENERATION_COMPLETE", "preflight.json",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> V15Config:
    parser = argparse.ArgumentParser(description="Run REALTALK Ours V15 progressive replay")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--v9-predictions", required=True)
    parser.add_argument("--v9-self-domains", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate", type=int, choices=GATES, required=True)
    parser.add_argument("--model", default=MODEL)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--no-enforce-canonical-v9", action="store_true")
    args = parser.parse_args()
    return V15Config(
        dataset_dir=args.dataset_dir,
        v9_predictions=args.v9_predictions,
        v9_self_domains=args.v9_self_domains,
        output_dir=args.output_dir,
        gate=args.gate,
        model=args.model,
        fresh=args.fresh,
        resume=args.resume,
        enforce_canonical_v9=not args.no_enforce_canonical_v9,
    )


if __name__ == "__main__":
    print(json.dumps(run_v15(parse_args()), ensure_ascii=False, indent=2))

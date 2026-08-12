"""Protocol-aligned REALTALK Task 1 evaluation for the Deep Empathy method."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .exp1_protocol import (
    REALTALK_PERSONA_SPLITS,
    build_message_level_points,
    build_profile_corpus,
    format_turns,
    message_speakers,
    protocol_turns,
    select_realtalk_splits,
    stable_hash,
)
from .exp2_generation import (
    bertscore_runtime_metadata,
    compute_bertscore_f1,
)
from .operation_checkpoint import OperationCheckpoint
from .personaemp.client import (
    ChatBackend,
    ChatResult,
    OpenAICompatibleChatBackend,
)
from .realtalk_evaluator import RealTalkLabelEvaluator
from .realtalk_ours_schemas import (
    ALIGNMENT_SCHEMA,
    SELF_DOMAIN_SCHEMA,
    USER_DOMAIN_SCHEMA,
    Normalizer,
    empty_user_domain,
    normalize_alignment,
    normalize_self_domain,
    normalize_user_domain,
)
from ..metrics import compute_rouge_l


EXPECTED_MODEL = "qwen3-max-2026-01-23"
ALLOWED_MODELS = frozenset({EXPECTED_MODEL, "deepseek-v4-flash"})
OFFICIAL_REALTALK_COMMIT = "b903e06a9770bf4e5fe9018c3e132889666d3b4a"
EXPECTED_FULL_TARGETS = 519
EXPECTED_SPEAKER_TARGETS = {
    "Emi": 37,
    "Nicolas": 117,
    "Kevin": 25,
    "Akib": 37,
    "Muhhamed": 37,
    "Nebraas": 51,
    "Paola": 23,
    "Vanessa": 116,
    "elise": 36,
    "Fahim Khan": 40,
}
PAPER_TABLE2_ROWS = {
    "w/o fine-tune": {
        "lexical": "0.14 +/- 0.04",
        "semantic": "0.76 +/- 0.08",
        "reflective": "0.62 +/- 0.13",
        "grounding": "0.40 +/- 0.13",
        "sentiment": "0.53 +/- 0.22",
        "emotion": "0.43 +/- 0.22",
        "intimacy": "0.06 +/- 0.01",
        "empathy": "1.80 +/- 0.55",
    },
    "w/ fine-tune": {
        "lexical": "0.14 +/- 0.05",
        "semantic": "0.78 +/- 0.04",
        "reflective": "0.77 +/- 0.09",
        "grounding": "0.62 +/- 0.08",
        "sentiment": "0.59 +/- 0.18",
        "emotion": "0.46 +/- 0.21",
        "intimacy": "0.07 +/- 0.01",
        "empathy": "1.24 +/- 0.12",
    },
}

SELF_DOMAIN_SYSTEM_PROMPT = """Compile a private Self Domain for a persona-simulation agent.
Model the target speaker as a person: identity context, communication signature, interaction policy,
and affective-social signature. Use the complete conversation to understand context, while grounding
claims about the target in the target's own messages. Distinguish stable tendencies from one-off events
and preserve uncertainty. Describe observed surface behavior rather than ideal social qualities: do not call
the target supportive, reflective, validating, warm, or inquisitive unless repeated target messages directly
show it. Qualitative descriptions must not override the deterministic rates. Copy the supplied observable
statistics exactly. Return only the schema."""

SELF_DOMAIN_USER_TEMPLATE = """TARGET SPEAKER: {speaker}
SOURCE: the first {session_count} sessions of the target's paper-assigned Ca conversation.

COMPLETE LOSSLESSLY MERGED CONVERSATION WITH SESSION AND TURN IDS:
{full_context}

DETERMINISTIC OBSERVABLE STATISTICS (copy exactly):
{observable_statistics}

Build the fixed cross-partner Self Domain for acting as {speaker}."""

USER_DOMAIN_SYSTEM_PROMPT = """Update a private five-layer User Domain for the conversation partner.
The fixed layers are Core, Regulation, Cognition, Identity, and Behavior. Use the complete finished session
for context, but every partner fact must cite partner evidence IDs. Preserve supported prior facts, revise
conflicts, consolidate overlap, and leave layers empty when evidence is insufficient. Return only the schema."""

USER_DOMAIN_USER_TEMPLATE = """TARGET SPEAKER (the agent): {speaker}
PARTNER (the modeled user): {partner}

PREVIOUS USER DOMAIN:
{previous_domain}

COMPLETE FINISHED SESSION WITH SESSION AND TURN IDS:
{completed_session}

Update the five layers using this completed session. Evidence IDs must identify partner turns."""

ALIGNMENT_SYSTEM_PROMPT = """You are the private Decision Agent for a persona-simulation actor.
Decide what the target speaker, represented by the Self Domain, would naturally choose to do next.
Treat the Self Domain as the default behavioral identity, read the complete real history, and use partner
profile facts only when they are relevant now. Internally balance self-led behavior and partner adaptation,
record that balance as lambda_trace, and commit to one concrete next action. The task is identity simulation,
not designing an ideal assistant response.

Choose exactly one primary move. Routine conversational relevance, answering a question, or understanding
the latest turn does not by itself require a balanced or partner-adaptive orientation. In ordinary factual,
casual, or daily-life exchange, default to self-led behavior and let the target's interaction prior determine
whether to answer, self-disclose, continue, or shift topic. Use balanced or partner-adaptive orientation only
for a clear current relational, emotional, or practical need that this target would actually accommodate.
Do not make every message acknowledge the partner and end with a question. `partner_has_open_thread` means
the latest partner turn contains a concrete unanswered request, explicit invitation to elaborate, or genuinely
unfinished reference whose resolution is needed now. A detail that could merely be interesting to ask about is
not an open thread. If the latest turn has no explicit question or request, default to no open thread. A direct
question that the primary `answer` move already resolves does not by itself license a reciprocal question.
Calibrate question decisions to the Self Domain's observed question rate as an upper tendency, not a quota.
This calibration applies separately from reciprocal-question: when question_rate >= 0.65, a `follow-up`
primary move is a common option when it directly develops the latest partner content; from 0.30 to 0.65 it
is occasional; below 0.30 it is rare and requires a particularly salient unfinished partner point. A follow-up
does not require emotional framing. `follow-up` is appropriate only when asking is the chosen primary move,
otherwise use `none`. Also use the
observed first-person rate and initiative to
preserve the target's cadence of self-disclosure and topic movement: the latest partner turn is context, not
an obligation to answer or reflect it. Do not optimize an ideal assistant response.

Prefer local conversational continuity over constructing a novel scene. When the visible exchange invites
the target to contribute, choose a low-specificity first-person stance, inclination, preference, immediate
intention, or ordinary status that advances the current topic. Preserve the target's observed initiative, but
do not equate initiative with inventing a named venue, destination, media title, detailed outing, past visit,
or elaborate present event. If the partner presents options or shares an analogous personal choice, consider
whether this target would naturally state their own choice or ask one same-topic reciprocal question rather
than replacing the topic with a new anecdote.

Do not infer an emotional or support need from ordinary enthusiasm, preferences, factual exchange, or casual
complaints. Set explicit_affect only when the partner explicitly expresses an affective state that matters to
the next move. For ordinary conversation, avoid therapeutic validation, emotional interpretation, praise, and
generic positive appraisal. Match the Self Domain's reflective_marker_rate and evaluative_opener_rate: they are
observed behavioral ceilings, not targets to satisfy in every response. If reflective_marker_rate < 0.10,
do not request reflective motivation or meaning unless the visible history explicitly calls for it. If
evaluative_opener_rate < 0.10, start directly with content rather than a generic positive evaluation.

Interpret lambda_trace only as how far this action departs from the target's normal behavior to accommodate
the partner. Reading the partner accurately is not adaptation. Relevant partner facts are not adaptation.
For routine exchange, use self-led with lambda_trace in [0, 0.25]. Use balanced only when a visible need
causes a real but still persona-consistent departure. Reserve partner-adaptive for an explicit, substantial
need and strong evidence that this target would respond that way. Keep partner_adaptation empty when no
departure is needed.

Primary moves are exclusive contracts:
- open: begin with one natural greeting or check-in; do not invent a current activity or setting.
- self-disclose: contribute one self-focused update or view; do not interpret, comfort, or question partner.
- answer: answer the latest question directly; content_direction must state the semantic slot being answered
  (for example wellbeing, opinion, or plan), not propose an unrelated activity or persona topic; do not add a
  return question or partner interpretation.
- acknowledge: give one concise reaction to partner content; do not add advice, an anecdote, or a question.
- follow-up: ask one relevant question; do not add a self-focused update or extended interpretation.
- topic-shift: introduce one target-led topic; do not first summarize or validate the partner.

After the primary move, `continuation_move` may be `reciprocal-question` only when all are true: the latest
partner turn explicitly licenses continued inquiry, the primary move does not already resolve that interaction,
missing_information names one necessary detail, continuation_value is high, and asking strongly fits this
target's observed question behavior. It must be one short, directly related question. Otherwise use none. A
follow-up primary move cannot also have a continuation move.
Keep the fields internally consistent:
- no open thread -> missing_information="", continuation_value="none", continuation_move="none";
- open thread with low value -> name the missing detail, but continuation_move="none";
- open thread with medium value -> continuation_move="none";
- open thread with high value -> reciprocal-question is optional, never automatic.

The Self Domain is a behavioral prior, not current-world evidence or a checklist of topics to demonstrate.
Use it primarily for voice, initiative, interaction pattern, and message scale. Never convert a Ca location,
weather pattern, job, hobby, food preference, routine, or anecdote into a present event merely because it is
in the Self Domain, and never combine several such attributes into a scene. If visible Cb history does not
establish a specific present fact about the target, keep any new self-expression low-specificity and ordinary
rather than inventing concrete weather, activities, objects, plans, or anecdotes. When history is empty,
primary_move must be open and the next action is a simple greeting or check-in with no constructed current
scenario. `content_direction` must name only an abstract conversational function, and `self_expression` must
name only delivery style. Neither field may draft the message or introduce a concrete event, activity, place,
object, food, weather, plan, anecdote, or hobby. Calibrate
message_scale against the supplied observable character statistics: typical stays near the target median;
extended requires a visible exchange that naturally supports it. Return only the schema."""

ALIGNMENT_USER_TEMPLATE = """TARGET SPEAKER: {speaker}
PARTNER: {partner}

FIXED SELF DOMAIN:
{self_domain}

CURRENT USER DOMAIN:
{user_domain}

REAL CAUSAL HISTORY BEFORE THE TARGET MESSAGE:
{history}

CURRENT SESSION: {current_session}
TARGET SPEAKER HAS ALREADY SPOKEN IN THIS SESSION: {target_spoke_in_current_session}

LATEST PARTNER MESSAGE (may be empty):
{latest_partner_message}

EXACT USER DOMAIN ACTIVATION WHITELIST:
{activation_whitelist}

Only copy relevant_user_domain entries verbatim from the whitelist. If it says NONE, return an empty array.
The only valid question_mode pairing is primary_move="follow-up" with question_mode="follow-up"; every other
primary_move requires question_mode="none". Independently decide whether one short reciprocal-question is
licensed by the three continuation fields and the Self Domain. Understand the current situation, record the
adaptive balance, and submit one next action for {speaker}."""

GENERATION_SYSTEM_TEMPLATE = """You are {speaker}. Continue the conversation.
Act as the person represented by the private Self Domain.
Follow the private next-action decision naturally.
Output only the message, not the speaker name."""

GENERATION_USER_TEMPLATE = """REAL CONVERSATION HISTORY BEFORE YOUR NEXT MESSAGE:
{history}

CURRENT SESSION: {current_session}
YOU HAVE ALREADY SPOKEN IN THIS SESSION: {target_spoke_in_current_session}

PRIVATE BEHAVIORAL SELF DOMAIN:
{behavioral_self_domain}

PRIVATE CURRENT SITUATION:
{situation}

PRIVATE NEXT ACTION:
{next_action}

ACTION CONTRACT:
{action_contract}

Realize the primary move and only its explicitly licensed continuation_move as one natural message at the
requested scale and in the Self Domain's
communication signature. This behavioral view intentionally omits identity facts, interests, and old events;
do not reconstruct or guess them. Do not add any other social move. Do not add a generic compliment,
validation, emotional interpretation, or reflective explanation merely to sound warm or conversational.
Realize unsupported self-disclosure at low specificity: express an ordinary current stance, inclination,
preference, intention, or status without adding a new proper-named place, venue, destination, institution,
media title, detailed outing, prior visit, or decorative scene that is absent from the visible history. Reuse
specific entities when they are already in the visible conversation. Do not invent detail merely to make the
message vivid. A first-person explanation should represent a plausible current motivation or choice, not a
rhetorical self-analysis added for style.
The deterministic behavioral calibration is authoritative when qualitative descriptions conflict with it.
Match its question, reflective-marker, evaluative-opener, and character-scale guidance instead of amplifying
abstract persona adjectives."""

FORMAT_REPAIR_TEMPLATE = """

FORMAT REPAIR REQUIRED
The previous response failed the strict contract:
{error}

Previous response:
{raw}

Return the same intended content corrected only to satisfy the requested schema. Do not add new evidence,
new facts, a new state interpretation, or a different behavior policy."""


@dataclass(frozen=True)
class RealTalkOursConfig:
    dataset_dir: str = "dataset"
    output_dir: str = "data/realtalk_ours_agentic_v2_qwen3_max"
    profile_sessions: int = 3
    test_sessions: int = 3
    max_eval_points_per_speaker: int = 0
    eval_points_per_session: int = 0
    eval_point_position_mode: str = "full-span"
    operation_max_attempts: int = 3
    speaker_filter: tuple[str, ...] = ()
    compute_local_metrics: bool = True
    compute_bertscore: bool = True
    continue_on_error: bool = True
    preflight_only: bool = False
    fresh: bool = False
    resume: bool = False
    decision_thinking: bool = True
    model_call_timeout_seconds: int = 240
    model: str = EXPECTED_MODEL


def run_realtalk_ours(
    config: RealTalkOursConfig,
    backend: ChatBackend | None = None,
    label_evaluator: RealTalkLabelEvaluator | None = None,
    bertscore_fn: Callable[[list[str], list[str]], list[float]] = compute_bertscore_f1,
) -> dict[str, Any]:
    """Run the Ours-only REALTALK persona-simulation protocol."""
    _validate_config(config)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.fresh:
        _clear_runtime_artifacts(output_dir)
    elif config.resume and not (output_dir / "checkpoint.json").exists():
        raise ValueError("--resume requires an existing checkpoint.json")

    backend = backend or _backend_from_env(config.model)
    if config.model not in ALLOWED_MODELS:
        raise ValueError(f"unsupported Ours model {config.model!r}")
    if getattr(backend, "model", None) != config.model:
        raise ValueError(
            f"REALTALK Ours requires exact configured model {config.model!r}; "
            f"got {getattr(backend, 'model', None)!r}"
        )
    preflight = _run_preflight(
        output_dir,
        backend,
        config.model,
        config.decision_thinking,
        config.model_call_timeout_seconds,
    )
    if config.preflight_only:
        return preflight

    splits = select_realtalk_splits(
        config.dataset_dir,
        speaker_filter=list(config.speaker_filter) or None,
    )
    dataset_manifest, prepared = _prepare_dataset(config, splits)
    full_protocol = _is_full_protocol(config, splits)
    if full_protocol and dataset_manifest["total_targets"] != EXPECTED_FULL_TARGETS:
        raise ValueError(
            f"full REALTALK reconstruction requires {EXPECTED_FULL_TARGETS} targets; "
            f"found {dataset_manifest['total_targets']}"
        )

    signature = _run_signature(config, backend, dataset_manifest)
    checkpoint = OperationCheckpoint(output_dir / "checkpoint.json", signature)
    raw_audit = output_dir / "raw_responses.jsonl"
    self_domains: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []

    for speaker_data in prepared:
        speaker = speaker_data["speaker"]
        speaker_id = _speaker_id(speaker)
        observable_statistics = _observable_statistics(
            speaker_data["profile"]["turns"], speaker
        )
        try:
            self_envelope = _structured_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"self_domain:{speaker_id}",
                system_prompt=SELF_DOMAIN_SYSTEM_PROMPT,
                user_prompt=SELF_DOMAIN_USER_TEMPLATE.format(
                    speaker=speaker,
                    session_count=config.profile_sessions,
                    full_context=_turns_with_ids(speaker_data["profile"]["turns"]),
                    observable_statistics=_json(observable_statistics),
                ),
                schema=SELF_DOMAIN_SCHEMA,
                normalizer=lambda value: _validate_observable_statistics(
                    normalize_self_domain(value), observable_statistics
                ),
                max_tokens=1800,
                max_attempts=config.operation_max_attempts,
                raw_audit=raw_audit,
                enable_thinking=False,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
        except Exception as exc:
            failure = _failure("self_domain", speaker, None, exc)
            unresolved.append(failure)
            checkpoint.store_excluded_result(f"self:{speaker_id}", failure)
            if not config.continue_on_error:
                raise
            continue

        self_domain = self_envelope["data"]
        self_domains[speaker] = self_domain
        user_domain = empty_user_domain()
        allowed_partner_turn_ids: set[str] = set()
        completed_session_updates: list[str] = []
        active_session = speaker_data["points"][0]["target_session"]

        for point in speaker_data["points"]:
            result_id = f"{speaker_id}:{point['sample_id']}"
            try:
                if point["target_session"] != active_session:
                    completed_turns = speaker_data["test_turns_by_session"][active_session]
                    completed_partner_ids = {
                        turn["turn_id"] for turn in completed_turns
                        if turn["speaker"].casefold() == speaker_data["partner"].casefold()
                    }
                    allowed_after_update = allowed_partner_turn_ids | completed_partner_ids
                    update_envelope = _structured_call(
                        checkpoint=checkpoint,
                        backend=backend,
                        operation_key=f"user_domain:{speaker_id}:after:{active_session}",
                        system_prompt=USER_DOMAIN_SYSTEM_PROMPT,
                        user_prompt=USER_DOMAIN_USER_TEMPLATE.format(
                            speaker=speaker,
                            partner=speaker_data["partner"],
                            previous_domain=_json(user_domain),
                            completed_session=_turns_with_ids(completed_turns),
                        ),
                        schema=USER_DOMAIN_SCHEMA,
                        normalizer=lambda value: _validate_user_domain_evidence(
                            normalize_user_domain(value),
                            allowed_after_update,
                        ),
                        max_tokens=1800,
                        max_attempts=config.operation_max_attempts,
                        raw_audit=raw_audit,
                        enable_thinking=False,
                        hard_timeout_seconds=config.model_call_timeout_seconds,
                    )
                    user_domain = update_envelope["data"]
                    allowed_partner_turn_ids = allowed_after_update
                    completed_session_updates.append(active_session)
                    active_session = point["target_session"]

                visible_partner_turns = [
                    turn for turn in point["context_turns"]
                    if turn["speaker"].casefold() == speaker_data["partner"].casefold()
                ]
                latest_partner_message = (
                    _turns_with_ids([visible_partner_turns[-1]])
                    if visible_partner_turns else ""
                )
                alignment_envelope = _structured_call(
                    checkpoint=checkpoint,
                    backend=backend,
                    operation_key=f"decision:{result_id}",
                    system_prompt=ALIGNMENT_SYSTEM_PROMPT,
                    user_prompt=ALIGNMENT_USER_TEMPLATE.format(
                        speaker=speaker,
                        partner=speaker_data["partner"],
                        self_domain=_json(self_domain),
                        user_domain=_json(user_domain),
                        history=_turns_with_session_boundaries(point["context_turns"]),
                        current_session=point["target_session"],
                        target_spoke_in_current_session=_target_spoke_in_session(
                            point["context_turns"], speaker, point["target_session"]
                        ),
                        latest_partner_message=latest_partner_message,
                        activation_whitelist=_profile_activation_whitelist(user_domain),
                    ),
                    schema=ALIGNMENT_SCHEMA,
                    normalizer=lambda value: _validate_decision_context(
                        _validate_decision_profile_activation(
                            normalize_alignment(value), user_domain
                        ),
                        has_history=bool(point["context_turns"]),
                    ),
                    max_tokens=1600,
                    max_attempts=config.operation_max_attempts,
                    raw_audit=raw_audit,
                    enable_thinking=config.decision_thinking,
                    hard_timeout_seconds=config.model_call_timeout_seconds,
                )
                decision = alignment_envelope["data"]
                generation_envelope = _text_call(
                    checkpoint=checkpoint,
                    backend=backend,
                    operation_key=f"generation:{result_id}",
                    system_prompt=GENERATION_SYSTEM_TEMPLATE.format(speaker=speaker),
                    user_prompt=GENERATION_USER_TEMPLATE.format(
                        speaker=speaker,
                        history=_turns_with_session_boundaries(point["context_turns"]),
                        current_session=point["target_session"],
                        target_spoke_in_current_session=_target_spoke_in_session(
                            point["context_turns"], speaker, point["target_session"]
                        ),
                        behavioral_self_domain=_json(_behavioral_self_domain(self_domain)),
                        situation=_json(decision["situation"]),
                        next_action=_json(decision["next_action"]),
                        action_contract=_action_contract(
                            decision["next_action"]["primary_move"],
                            decision["next_action"]["continuation_move"],
                        ),
                    ),
                    speaker=speaker,
                    max_attempts=config.operation_max_attempts,
                    raw_audit=raw_audit,
                    enable_thinking=False,
                    hard_timeout_seconds=config.model_call_timeout_seconds,
                )
                result = {
                    "result_id": result_id,
                    "speaker": speaker,
                    "partner": speaker_data["partner"],
                    "train_chat": speaker_data["split"]["train_chat"],
                    "test_chat": speaker_data["split"]["test_chat"],
                    "profile_sessions": list(speaker_data["profile"]["sessions"]),
                    "test_sessions": list(point["test_sessions"]),
                    "target_session": point["target_session"],
                    "message_level_index": point["message_level_index"],
                    "target_turn_id": point["target"]["turn_id"],
                    "context_turn_ids": [
                        turn["turn_id"] for turn in point["context_turns"]
                    ],
                    "context_hash": point["history_hash"],
                    "context_truncated": point["context_truncated"],
                    "ground_truth": point["target_message"],
                    "generated_message": generation_envelope["data"],
                    "self_domain_hash": stable_hash(self_domain),
                    "user_domain": user_domain,
                    "user_domain_completed_session_updates": list(completed_session_updates),
                    "situation": decision["situation"],
                    "relevant_user_domain": decision["relevant_user_domain"],
                    "alignment": decision["alignment"],
                    "next_action": decision["next_action"],
                    "operation_audit": {
                        "decision": alignment_envelope["audit"],
                        "generation": generation_envelope["audit"],
                    },
                }
                checkpoint.data["failures"].pop(f"sample:{result_id}", None)
                checkpoint.store_result(result_id, result)
            except Exception as exc:
                failure = _failure("sample", speaker, result_id, exc)
                unresolved.append(failure)
                checkpoint.store_excluded_result(result_id, failure)
                if not config.continue_on_error:
                    raise

    results = sorted(
        checkpoint.result_values(),
        key=lambda item: (
            _split_order(item["speaker"]),
            int(item["message_level_index"]),
        ),
    )
    unresolved = _checkpoint_unresolved(checkpoint)
    _write_generation_outputs(
        output_dir,
        results,
        self_domains,
        unresolved,
        dataset_manifest,
        config,
        backend,
        preflight,
        signature,
        checkpoint,
    )
    expected = dataset_manifest["total_targets"]
    generation_complete = len(results) == expected and not unresolved
    if generation_complete:
        _write_json(output_dir / "GENERATION_COMPLETE", {
            "completed_at_utc": _now(),
            "records": len(results),
            "model": backend.model,
            "run_signature": signature,
        })
    else:
        _remove(output_dir / "GENERATION_COMPLETE")

    local_summary: dict[str, Any] | None = None
    if config.compute_local_metrics and generation_complete:
        evaluator = label_evaluator or RealTalkLabelEvaluator()
        local_summary = _run_local_metrics(
            output_dir,
            results,
            checkpoint,
            evaluator,
            config,
            bertscore_fn,
        )
        _write_json(output_dir / "LOCAL_METRICS_COMPLETE", {
            "completed_at_utc": _now(),
            "records": len(results),
            "metrics": [
                "rouge_l",
                "bertscore_f1" if config.compute_bertscore else None,
                "sentiment_accuracy",
                "emotion_accuracy",
                "intimacy_absolute_difference",
            ],
            "run_signature": signature,
        })
    else:
        _remove(output_dir / "LOCAL_METRICS_COMPLETE")

    _remove(output_dir / "PIPELINE_COMPLETE")
    _write_json(output_dir / "GPT_EVALUATION_PENDING.json", {
        "required_model": "gpt-4o-mini",
        "pending_metrics": [
            "reflectiveness_accuracy",
            "grounding_accuracy",
            "empathy_absolute_difference",
        ],
        "pipeline_complete_allowed": False,
        "reason": "No verified gpt-4o-mini evaluation endpoint was configured for this run.",
    })
    _write_partial_report(output_dir, local_summary, dataset_manifest)
    return {
        "generation_complete": generation_complete,
        "records": len(results),
        "expected_records": expected,
        "unresolved": unresolved,
        "local_metrics": local_summary,
        "output_dir": str(output_dir),
    }


def _prepare_dataset(
    config: RealTalkOursConfig,
    splits: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = Path(config.dataset_dir)
    prepared = []
    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in splits:
        train_path = dataset / split["train_chat"]
        test_path = dataset / split["test_chat"]
        train_chat = _load_json(train_path)
        test_chat = _load_json(test_path)
        speaker = next(
            value
            for value in message_speakers(test_chat)
            if value.casefold() == split["speaker"].casefold()
        )
        partner = next(
            value
            for value in message_speakers(test_chat)
            if value.casefold() != speaker.casefold()
        )
        profile = build_profile_corpus(
            train_chat,
            speaker,
            profile_sessions=config.profile_sessions,
            merge_adjacent_bubbles=True,
        )
        points = build_message_level_points(
            test_chat,
            speaker,
            test_sessions=config.test_sessions,
            max_context_chars=0,
            max_eval_points=config.max_eval_points_per_speaker,
            merge_adjacent_bubbles=True,
        )
        if config.eval_points_per_session:
            points = _select_even_points_per_session(
                points,
                selected_count=config.eval_points_per_session,
                position_mode=config.eval_point_position_mode,
            )
        if not points:
            raise ValueError(f"no test targets for {speaker}")
        if any(point["context_truncated"] for point in points):
            raise ValueError(f"V2 forbids truncated history for {speaker}")
        selected_sessions = list(points[0]["test_sessions"])
        selected_set = set(selected_sessions)
        test_turns = [
            turn for turn in protocol_turns(test_chat, merge_adjacent_bubbles=True)
            if turn["session_id"] in selected_set
        ]
        test_turns_by_session = {
            session_id: [
                turn for turn in test_turns if turn["session_id"] == session_id
            ]
            for session_id in selected_sessions
        }
        counts[split["speaker"]] = len(points)
        for path in (train_path, test_path):
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared.append({
            "split": dict(split),
            "speaker": speaker,
            "partner": partner,
            "profile": profile,
            "points": points,
            "test_turns": test_turns,
            "test_turns_by_session": test_turns_by_session,
        })
    manifest = {
        "dataset": "REALTALK public preprocessed conversations",
        "official_repository_commit": OFFICIAL_REALTALK_COMMIT,
        "table8_speaker_specific_splits": [dict(item) for item in splits],
        "profile_sessions": config.profile_sessions,
        "test_sessions": config.test_sessions,
        "sample_unit": "merged consecutive same-speaker message M_t",
        "merge_consecutive_same_speaker_within_session": True,
        "merged_bubbles_preserve_text_with_newlines_and_source_ids": True,
        "history_compression_enabled": False,
        "history_truncation_enabled": False,
        "full_three_session_history": True,
        "evaluation_point_selection": (
            {
                "method": "deterministic_even_position_within_session",
                "points_per_session": config.eval_points_per_session,
                "position_mode": config.eval_point_position_mode,
                "uses_message_text": False,
                "uses_ground_truth": False,
                "uses_judge_labels": False,
            }
            if config.eval_points_per_session
            else {"method": "prefix_or_full_protocol"}
        ),
        "generated_outputs_are_never_rolled_into_history": True,
        "source_files_sha256": dict(sorted(files.items())),
        "source_files_aggregate_sha256": stable_hash(dict(sorted(files.items()))),
        "targets_by_speaker": counts,
        "total_targets": sum(counts.values()),
        "paper_reported_exact_target_count": None,
        "reconstructed_target_count": True,
    }
    return manifest, prepared


def _run_preflight(
    output_dir: Path,
    backend: ChatBackend,
    expected_model: str,
    decision_thinking: bool,
    hard_timeout_seconds: int,
) -> dict[str, Any]:
    path = output_dir / "preflight.json"
    if path.exists():
        value = _load_json(path)
        if (
            value.get("model") == expected_model
            and value.get("stage_thinking") == {
                "self_domain": False,
                "user_domain": False,
                "decision": decision_thinking,
                "generation": False,
            }
            and value.get("nonthinking_generation_succeeded") is True
        ):
            return value
    available_fn = getattr(backend, "available_models", None)
    available = list(available_fn()) if callable(available_fn) else [backend.model]
    if expected_model not in available:
        raise ValueError(
            f"configured credential cannot access {expected_model}; visible={available[:20]}"
        )
    response = _call_with_hard_timeout(
        lambda: backend.chat(
            "Return exactly READY.",
            "READY",
            temperature=0.0,
            top_p=0.9,
            max_tokens=8,
            enable_thinking=False,
        ),
        hard_timeout_seconds,
        "preflight",
    )
    if "READY" not in response.content.upper():
        raise ValueError(f"minimal model preflight failed: {response.content!r}")
    value = {
        "checked_at_utc": _now(),
        "model": backend.model,
        "base_url_host": _safe_host(getattr(backend, "base_url", "injected-test-backend")),
        "stage_thinking": {
            "self_domain": False,
            "user_domain": False,
            "decision": decision_thinking,
            "generation": False,
        },
        "model_visible": True,
        "nonthinking_generation_succeeded": True,
        "minimal_generation_attempts": response.attempts,
    }
    _write_json(path, value)
    return value


def _structured_call(
    *,
    checkpoint: OperationCheckpoint,
    backend: ChatBackend,
    operation_key: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    normalizer: Normalizer,
    max_tokens: int,
    max_attempts: int,
    raw_audit: Path,
    enable_thinking: bool,
    hard_timeout_seconds: int = 0,
) -> dict[str, Any]:
    repair = {"raw": "", "error": ""}
    logical_attempt = {"value": 0}

    def operation() -> ChatResult:
        logical_attempt["value"] += 1
        prompt = user_prompt
        if repair["error"]:
            prompt += FORMAT_REPAIR_TEMPLATE.format(
                error=repair["error"], raw=repair["raw"][:12000]
            )
        return _call_with_hard_timeout(
            lambda: backend.chat(
                system_prompt,
                prompt,
                temperature=0.2,
                top_p=0.9,
                max_tokens=max_tokens,
                response_schema=schema,
                enable_thinking=enable_thinking,
            ),
            hard_timeout_seconds,
            operation_key,
        )

    def validate(result: ChatResult) -> dict[str, Any]:
        _append_jsonl(raw_audit, {
            "operation_key": operation_key,
            "logical_attempt": logical_attempt["value"],
            "model": result.model,
            "raw_response": result.content,
            "reasoning_content": result.reasoning_content,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_seconds": result.latency_seconds,
            "network_attempts": result.attempts,
            "recorded_at_utc": _now(),
        })
        try:
            parsed = json.loads(result.content)
            data = normalizer(parsed)
        except Exception as exc:
            repair["raw"] = result.content
            repair["error"] = f"{type(exc).__name__}: {exc}"
            raise
        return {
            "data": data,
            "audit": {
                "model": result.model,
                "logical_attempts": logical_attempt["value"],
                "network_attempts_last_call": result.attempts,
                "prompt_tokens_last_call": result.prompt_tokens,
                "completion_tokens_last_call": result.completion_tokens,
                "latency_seconds_last_call": result.latency_seconds,
                "schema": schema["name"],
                "thinking_enabled": enable_thinking,
                "reasoning_sha256": stable_hash(result.reasoning_content),
            },
        }

    return checkpoint.execute(
        operation_key,
        operation,
        validate,
        max_attempts,
        usage_supplier=lambda: dict(getattr(backend, "token_usage", {})),
    )


class ModelCallHardTimeout(TimeoutError):
    """Raised when an upstream SDK call exceeds the experiment watchdog."""


class _HardTimeoutInterrupt(BaseException):
    """Escapes SDK retry loops that catch ordinary Exception subclasses."""


def _call_with_hard_timeout(
    operation: Callable[[], ChatResult],
    timeout_seconds: int,
    operation_key: str,
) -> ChatResult:
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return operation()

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise _HardTimeoutInterrupt()

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        try:
            return operation()
        except _HardTimeoutInterrupt as exc:
            raise ModelCallHardTimeout(
                f"model call {operation_key!r} exceeded {timeout_seconds}s hard timeout"
            ) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _text_call(
    *,
    checkpoint: OperationCheckpoint,
    backend: ChatBackend,
    operation_key: str,
    system_prompt: str,
    user_prompt: str,
    speaker: str,
    max_attempts: int,
    raw_audit: Path,
    enable_thinking: bool,
    hard_timeout_seconds: int = 0,
) -> dict[str, Any]:
    logical_attempt = {"value": 0}

    def operation() -> ChatResult:
        logical_attempt["value"] += 1
        return _call_with_hard_timeout(
            lambda: backend.chat(
                system_prompt,
                user_prompt,
                temperature=0.6,
                top_p=0.9,
                max_tokens=300,
                enable_thinking=enable_thinking,
            ),
            hard_timeout_seconds,
            operation_key,
        )

    def validate(result: ChatResult) -> dict[str, Any]:
        _append_jsonl(raw_audit, {
            "operation_key": operation_key,
            "logical_attempt": logical_attempt["value"],
            "model": result.model,
            "raw_response": result.content,
            "reasoning_content": result.reasoning_content,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_seconds": result.latency_seconds,
            "network_attempts": result.attempts,
            "recorded_at_utc": _now(),
        })
        return {
            "data": _normalize_generated_message(result.content, speaker),
            "audit": {
                "model": result.model,
                "logical_attempts": logical_attempt["value"],
                "network_attempts_last_call": result.attempts,
                "prompt_tokens_last_call": result.prompt_tokens,
                "completion_tokens_last_call": result.completion_tokens,
                "latency_seconds_last_call": result.latency_seconds,
                "thinking_enabled": enable_thinking,
                "reasoning_sha256": stable_hash(result.reasoning_content),
            },
        }

    return checkpoint.execute(
        operation_key,
        operation,
        validate,
        max_attempts,
        usage_supplier=lambda: dict(getattr(backend, "token_usage", {})),
    )


def _run_local_metrics(
    output_dir: Path,
    results: list[dict[str, Any]],
    checkpoint: OperationCheckpoint,
    evaluator: RealTalkLabelEvaluator,
    config: RealTalkOursConfig,
    bertscore_fn: Callable[[list[str], list[str]], list[float]],
) -> dict[str, Any]:
    scored = []
    for result in results:
        result_id = result["result_id"]
        reference_labels = checkpoint.execute(
            f"local_labels:reference:{result_id}",
            lambda text=result["ground_truth"]: evaluator.annotate(text),
            _normalize_local_labels,
            config.operation_max_attempts,
        )
        candidate_labels = checkpoint.execute(
            f"local_labels:candidate:{result_id}",
            lambda text=result["generated_message"]: evaluator.annotate(text),
            _normalize_local_labels,
            config.operation_max_attempts,
        )
        scored.append({
            **result,
            "local_labels": {
                "reference": reference_labels,
                "candidate": candidate_labels,
            },
            "local_metrics": {
                "rouge_l": compute_rouge_l(
                    result["ground_truth"], result["generated_message"]
                ),
                "sentiment_accuracy": float(
                    reference_labels["sentiment"] == candidate_labels["sentiment"]
                ),
                "emotion_accuracy": float(
                    reference_labels["emotion"] == candidate_labels["emotion"]
                ),
                "intimacy_absolute_difference": round(
                    abs(reference_labels["intimacy"] - candidate_labels["intimacy"]),
                    6,
                ),
            },
        })
    if config.compute_bertscore:
        values = checkpoint.execute(
            "local_metrics:bertscore_batch",
            lambda: bertscore_fn(
                [item["ground_truth"] for item in scored],
                [item["generated_message"] for item in scored],
            ),
            lambda value: _normalize_bertscore(value, len(scored)),
            config.operation_max_attempts,
        )
        for item, value in zip(scored, values):
            item["local_metrics"]["bertscore_f1"] = value
    summary = _aggregate_local_metrics(scored)
    summary["classifier_metadata"] = evaluator.metadata()
    summary["bertscore_metadata"] = (
        bertscore_runtime_metadata() if config.compute_bertscore else None
    )
    _write_jsonl(output_dir / "results_with_local_metrics.jsonl", scored)
    _write_json(output_dir / "local_metrics_summary.json", summary)
    return summary


def _aggregate_local_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = list(results[0]["local_metrics"]) if results else []
    by_speaker: dict[str, dict[str, float]] = {}
    for speaker in dict.fromkeys(item["speaker"] for item in results):
        speaker_results = [item for item in results if item["speaker"] == speaker]
        by_speaker[speaker] = {
            metric: round(statistics.mean(
                item["local_metrics"][metric] for item in speaker_results
            ), 6)
            for metric in metric_names
        }
    macro = {}
    for metric in metric_names:
        values = [scores[metric] for scores in by_speaker.values()]
        macro[metric] = {
            "mean": round(statistics.mean(values), 6),
            "std_population": round(statistics.pstdev(values), 6),
            "display": (
                f"{statistics.mean(values):.2f} +/- {statistics.pstdev(values):.2f}"
            ),
        }
    micro = {
        metric: round(statistics.mean(
            item["local_metrics"][metric] for item in results
        ), 6)
        for metric in metric_names
    }
    return {
        "aggregation_for_table2": "speaker_macro_mean_and_population_std",
        "speaker_count": len(by_speaker),
        "message_count": len(results),
        "by_speaker": by_speaker,
        "speaker_macro": macro,
        "message_micro": micro,
    }


def _write_generation_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    self_domains: dict[str, dict[str, Any]],
    unresolved: list[dict[str, Any]],
    dataset_manifest: dict[str, Any],
    config: RealTalkOursConfig,
    backend: ChatBackend,
    preflight: dict[str, Any],
    signature: str,
    checkpoint: OperationCheckpoint,
) -> None:
    _write_jsonl(output_dir / "predictions.jsonl", results)
    _write_json(output_dir / "self_domains.json", self_domains)
    _write_json(output_dir / "unresolved_errors.json", unresolved)
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(output_dir / "run_manifest.json", {
        "created_at_utc": _now(),
        "protocol": "realtalk_task1_ours_agentic_v8_low_specificity_continuity",
        "comparison_status": "protocol_aligned_not_runtime_identical",
        "paper_persona_simulation_model_disclosed": False,
        "implementation_repository_commit": _repository_commit(),
        "ours_model": backend.model,
        "all_ours_stages_use_same_model": True,
        "stage_thinking": {
            "self_domain": False,
            "user_domain": False,
            "decision": config.decision_thinking,
            "generation": False,
        },
        "training_or_finetuning": False,
        "omega_enabled": False,
        "future_user_state_enabled": False,
        "history_compression_enabled": False,
        "history_truncation_enabled": False,
        "full_three_session_history": True,
        "user_domain_update_frequency": "after_session_1_and_after_session_2",
        "multiple_policy_candidates_enabled": False,
        "verification_or_rewrite_enabled": False,
        "response_length_restriction": None,
        "generated_output_rollout": False,
        "decoding": {
            "self_domain": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1800},
            "user_domain": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1800},
            "decision": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1600},
            "generation": {"temperature": 0.6, "top_p": 0.9, "max_tokens": 300},
            "seed": None,
        },
        "structured_logical_attempts": config.operation_max_attempts,
        "config": asdict(config),
        "dataset_manifest_sha256": stable_hash(dataset_manifest),
        "prompt_hashes": _prompt_hashes(),
        "schema_hashes": {
            "self_domain": stable_hash(SELF_DOMAIN_SCHEMA),
            "user_domain": stable_hash(USER_DOMAIN_SCHEMA),
            "decision": stable_hash(ALIGNMENT_SCHEMA),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preflight": preflight,
        "run_signature": signature,
        "token_usage": _checkpoint_usage(checkpoint=checkpoint, backend=backend),
        "gpt_evaluation_status": "pending_verified_gpt-4o-mini_endpoint",
        "pipeline_complete": False,
    })


def _write_partial_report(
    output_dir: Path,
    local_summary: dict[str, Any] | None,
    dataset_manifest: dict[str, Any],
) -> None:
    ours = {
        "lexical": _metric_display(local_summary, "rouge_l"),
        "semantic": _metric_display(local_summary, "bertscore_f1"),
        "reflective": "pending gpt-4o-mini",
        "grounding": "pending gpt-4o-mini",
        "sentiment": _metric_display(local_summary, "sentiment_accuracy"),
        "emotion": _metric_display(local_summary, "emotion_accuracy"),
        "intimacy": _metric_display(local_summary, "intimacy_absolute_difference"),
        "empathy": "pending gpt-4o-mini",
    }
    payload = {
        "status": "partial_local_metrics_only",
        "paper_rows": PAPER_TABLE2_ROWS,
        "ours": ours,
        "disclosure": (
            "Protocol-aligned comparison on reconstructed public REALTALK Task 1 points. "
            "The paper does not disclose its persona-simulation base model; Ours uses qwen3-max-2026-01-23."
        ),
        "reconstructed_targets": dataset_manifest["total_targets"],
    }
    _write_json(output_dir / "table2_partial.json", payload)
    lines = [
        "# REALTALK Table 2 + Ours (Partial)",
        "",
        "This is a protocol-aligned comparison on reconstructed public REALTALK Task 1 points. ",
        "The paper does not disclose its persona-simulation base model; Ours uses `qwen3-max-2026-01-23`.",
        "",
        "| Method | Lexical | Semantic | Reflective | Grounding | Sentiment | Emotion | Intimacy | Empathy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in [*PAPER_TABLE2_ROWS.items(), ("Ours", ours)]:
        lines.append(
            "| " + method + " | " + " | ".join(
                row[key]
                for key in (
                    "lexical", "semantic", "reflective", "grounding",
                    "sentiment", "emotion", "intimacy", "empathy",
                )
            ) + " |"
        )
    lines.extend([
        "",
        f"Reconstructed targets: {dataset_manifest['total_targets']}.",
        "The three GPT-scored metrics remain pending and `PIPELINE_COMPLETE` is intentionally absent.",
    ])
    (output_dir / "REPORT_PARTIAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _observable_statistics(
    turns: list[dict[str, Any]], speaker: str
) -> dict[str, Any]:
    target_turns = [
        turn for turn in turns
        if turn["speaker"].casefold() == speaker.casefold()
    ]
    if not target_turns:
        raise ValueError(f"no Self Domain evidence for {speaker}")
    lengths = [len(turn["content"]) for turn in target_turns]
    first_person = re.compile(r"\b(?:i|i'm|i've|i'd|me|my|mine|we|our|ours|us)\b", re.I)
    reflective_marker = re.compile(
        r"\b(?:i feel|i think|i realize|i(?:'|’)m aware|i(?:'|’)ve realized|"
        r"i believe|i decided|because i|makes me feel)\b", re.I
    )
    evaluative_opener = re.compile(
        r"^\s*(?:oh\s+)?(?:wow[,! ]+)?(?:that(?:'|’)s|this is|it(?:'|’)s)\s+"
        r"(?:so\s+)?(?:amazing|awesome|beautiful|cool|fantastic|fascinating|"
        r"great|incredible|lovely|wonderful)\b", re.I
    )
    merged_counts = [len(turn.get("message_indices", [0])) for turn in target_turns]
    return {
        "target_message_count": len(target_turns),
        "mean_characters": round(statistics.mean(lengths), 4),
        "median_characters": round(float(statistics.median(lengths)), 4),
        "question_rate": round(
            statistics.mean("?" in turn["content"] for turn in target_turns), 4
        ),
        "first_person_rate": round(
            statistics.mean(bool(first_person.search(turn["content"])) for turn in target_turns), 4
        ),
        "reflective_marker_rate": round(
            statistics.mean(bool(reflective_marker.search(turn["content"])) for turn in target_turns), 4
        ),
        "evaluative_opener_rate": round(
            statistics.mean(bool(evaluative_opener.search(turn["content"])) for turn in target_turns), 4
        ),
        "median_merged_bubbles": round(float(statistics.median(merged_counts)), 4),
    }


def _validate_observable_statistics(
    value: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    actual = value["observable_statistics"]
    if actual != expected:
        raise ValueError(
            f"observable_statistics changed: expected={expected}, actual={actual}"
        )
    return value


def _action_contract(primary_move: str, continuation_move: str = "none") -> str:
    contracts = {
        "open": (
            "Only begin with one natural greeting or check-in. Do not invent a current "
            "activity, setting, event, plan, anecdote, or topic detail."
        ),
        "self-disclose": (
            "Only contribute one self-focused update or view. Do not interpret, "
            "comfort, advise, acknowledge, or question the partner."
        ),
        "answer": (
            "Only answer the latest question directly. Do not add a return "
            "question, partner interpretation, or unrelated self-disclosure."
        ),
        "acknowledge": (
            "Only give one concise reaction to the partner content. Do not add "
            "advice, an anecdote, or a question."
        ),
        "follow-up": (
            "Only ask one relevant question. Do not add a self-focused update, "
            "comfort statement, or extended interpretation."
        ),
        "topic-shift": (
            "Only introduce one target-led topic. Do not first summarize, "
            "validate, advise, or question the partner."
        ),
    }
    try:
        primary_contract = contracts[primary_move]
    except KeyError as exc:
        raise ValueError(f"unknown primary move: {primary_move}") from exc
    if continuation_move == "none":
        return primary_contract + " Do not add a follow-up question."
    if continuation_move == "reciprocal-question":
        return (
            primary_contract
            + " Keep the primary part to one short, direct sentence. Do not add praise, generic "
            "validation, explain motivations, interpret emotions, reflect on meaning, or add an anecdote. Then ask exactly one short "
            "question about the explicitly identified missing information."
        )
    raise ValueError(f"unknown continuation move: {continuation_move}")


def _behavioral_self_domain(self_domain: dict[str, Any]) -> dict[str, Any]:
    stats = self_domain["observable_statistics"]
    return {
        "communication_signature": self_domain["communication_signature"],
        "interaction_policy_prior": self_domain["interaction_policy_prior"],
        "affective_social_signature": self_domain["affective_social_signature"],
        "observable_statistics": stats,
        "deterministic_behavior_calibration": _behavior_calibration(stats),
    }


def _behavior_calibration(stats: dict[str, Any]) -> dict[str, Any]:
    required = {
        "question_rate", "reflective_marker_rate", "evaluative_opener_rate",
        "mean_characters", "median_characters",
    }
    if not required.issubset(stats):
        return {
            "statistics_available": False,
            "question_tendency": "unknown",
            "reflective_explanation": "unknown",
            "generic_positive_opener": "unknown",
            "short_character_guide": None,
            "typical_character_guide": None,
            "scale_is_guidance_not_hard_limit": True,
        }
    question_rate = float(stats["question_rate"])
    reflective_rate = float(stats["reflective_marker_rate"])
    opener_rate = float(stats["evaluative_opener_rate"])
    median = float(stats["median_characters"])
    mean = float(stats["mean_characters"])
    return {
        "statistics_available": True,
        "question_tendency": (
            "common_when_latest_content_naturally_invites_it"
            if question_rate >= 0.65 else
            "occasional_for_a_salient_unfinished_point"
            if question_rate >= 0.30 else
            "rare_without_a_direct_need"
        ),
        "reflective_explanation": (
            "available_when_contextually_relevant"
            if reflective_rate >= 0.20 else
            "rare_unless_explicitly_called_for"
        ),
        "generic_positive_opener": (
            "available_when_natural" if opener_rate >= 0.20 else "normally_omit"
        ),
        "short_character_guide": round(max(12.0, min(mean, median) * 0.6), 1),
        "typical_character_guide": round(statistics.mean((mean, median)), 1),
        "scale_is_guidance_not_hard_limit": True,
    }


def _normalize_generated_message(value: str, speaker: str) -> str:
    if not isinstance(value, str):
        raise ValueError("generated message must be text")
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.I).strip()
    text = re.sub(rf"^{re.escape(speaker)}\s*:\s*", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if not text:
        raise ValueError("generated message must not be empty")
    if text.startswith("{") or text.startswith("["):
        raise ValueError("generated message leaked structured output")
    leaked = ("lambda_trace", "next_action", "user_domain", "self_domain")
    if any(marker in text.casefold() for marker in leaked):
        raise ValueError("generated message leaked internal state")
    return text


def _validate_user_domain_evidence(
    value: dict[str, Any], allowed_turn_ids: set[str]
) -> dict[str, Any]:
    for layer in ("core", "regulation", "cognition", "identity", "behavior"):
        for fact in value[layer]:
            evidence = set(fact["evidence_ids"])
            if not evidence:
                raise ValueError(f"{layer} fact has no evidence turn IDs")
            unknown = evidence - allowed_turn_ids
            if unknown:
                raise ValueError(
                    f"{layer} fact cites unobserved or non-partner turns: {sorted(unknown)}"
                )
    return value


def _validate_decision_profile_activation(
    value: dict[str, Any], user_domain: dict[str, Any]
) -> dict[str, Any]:
    available = {
        (layer, fact["value"])
        for layer in ("core", "regulation", "cognition", "identity", "behavior")
        for fact in user_domain[layer]
    }
    selected = {
        (fact["layer"], fact["value"])
        for fact in value["relevant_user_domain"]
    }
    unknown = selected - available
    if unknown:
        raise ValueError(f"decision activated unknown User Domain facts: {sorted(unknown)}")
    return value


def _profile_activation_whitelist(user_domain: dict[str, Any]) -> str:
    facts = [
        {"layer": layer, "value": fact["value"]}
        for layer in ("core", "regulation", "cognition", "identity", "behavior")
        for fact in user_domain[layer]
    ]
    return "NONE (relevant_user_domain must be [])" if not facts else _json(facts)


def _validate_decision_context(
    value: dict[str, Any], *, has_history: bool
) -> dict[str, Any]:
    if not has_history and value["next_action"]["primary_move"] != "open":
        raise ValueError("empty history requires open primary_move")
    situation = value["situation"]
    continuation = value["next_action"]["continuation_move"]
    if continuation == "reciprocal-question":
        if not situation["partner_has_open_thread"]:
            raise ValueError("reciprocal question requires an open partner thread")
        if not situation["missing_information"]:
            raise ValueError("reciprocal question requires named missing information")
        if situation["continuation_value"] != "high":
            raise ValueError("reciprocal question requires high continuation value")
    if not situation["partner_has_open_thread"]:
        if situation["missing_information"]:
            raise ValueError("closed partner thread cannot declare missing information")
        if situation["continuation_value"] != "none":
            raise ValueError("closed partner thread requires continuation value none")
    return value


def _normalize_local_labels(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"emotion", "sentiment", "intimacy"}:
        raise ValueError("local labels must contain exactly emotion, sentiment, intimacy")
    intimacy = value["intimacy"]
    if isinstance(intimacy, bool) or not isinstance(intimacy, (int, float)):
        raise ValueError("intimacy must be numeric")
    intimacy = float(intimacy)
    if not 0 <= intimacy <= 1:
        raise ValueError("intimacy must be in [0,1]")
    return {
        "emotion": str(value["emotion"]).strip().lower(),
        "sentiment": str(value["sentiment"]).strip().lower(),
        "intimacy": intimacy,
    }


def _normalize_bertscore(value: Any, expected: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError("BERTScore output does not align with predictions")
    normalized = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("BERTScore values must be numeric")
        score = float(item)
        if not -1 <= score <= 1:
            raise ValueError("BERTScore values must be in [-1,1]")
        normalized.append(round(score, 6))
    return normalized


def _backend_from_env(model: str = EXPECTED_MODEL) -> OpenAICompatibleChatBackend:
    def env(name: str, fallback: str = "") -> str:
        return os.getenv(f"REALTALK_OURS_{name}", fallback).strip()

    return OpenAICompatibleChatBackend(
        api_key=env("API_KEY", os.getenv("API_KEY", "")),
        base_url=env("BASE_URL", os.getenv("BASE_URL", "")),
        model=model,
        timeout_seconds=float(env("TIMEOUT_SECONDS", "180")),
        max_attempts=int(env("NETWORK_MAX_ATTEMPTS", "6")),
        enable_thinking=False,
    )


def _run_signature(
    config: RealTalkOursConfig,
    backend: ChatBackend,
    dataset_manifest: dict[str, Any],
) -> str:
    cfg = asdict(config)
    for key in (
        "output_dir", "fresh", "resume", "continue_on_error", "preflight_only"
    ):
        cfg.pop(key, None)
    return stable_hash({
        "config": cfg,
        "model": backend.model,
        "stage_thinking": {
            "self_domain": False,
            "user_domain": False,
            "decision": config.decision_thinking,
            "generation": False,
        },
        "dataset_manifest": dataset_manifest,
        "prompts": _prompt_hashes(),
        "schemas": {
            "self": stable_hash(SELF_DOMAIN_SCHEMA),
            "user": stable_hash(USER_DOMAIN_SCHEMA),
            "decision": stable_hash(ALIGNMENT_SCHEMA),
        },
        "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_repository_commit": _repository_commit(),
    })


def _prompt_hashes() -> dict[str, str]:
    return {
        "self_domain_system": stable_hash(SELF_DOMAIN_SYSTEM_PROMPT),
        "self_domain_user": stable_hash(SELF_DOMAIN_USER_TEMPLATE),
        "user_domain_system": stable_hash(USER_DOMAIN_SYSTEM_PROMPT),
        "user_domain_user": stable_hash(USER_DOMAIN_USER_TEMPLATE),
        "decision_system": stable_hash(ALIGNMENT_SYSTEM_PROMPT),
        "decision_user": stable_hash(ALIGNMENT_USER_TEMPLATE),
        "generation_system": stable_hash(GENERATION_SYSTEM_TEMPLATE),
        "generation_user": stable_hash(GENERATION_USER_TEMPLATE),
        "format_repair": stable_hash(FORMAT_REPAIR_TEMPLATE),
    }


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"invalid implementation repository commit: {value!r}")
    return value


def _validate_config(config: RealTalkOursConfig) -> None:
    if config.profile_sessions != 3 or config.test_sessions != 3:
        raise ValueError("main REALTALK Ours protocol requires exactly three Ca and Cb sessions")
    if config.operation_max_attempts != 3:
        raise ValueError("structured logical attempts are fixed at three")
    if config.model_call_timeout_seconds < 30:
        raise ValueError("model_call_timeout_seconds must be at least 30")
    if config.max_eval_points_per_speaker < 0:
        raise ValueError("max_eval_points_per_speaker must be non-negative")
    if config.eval_points_per_session < 0:
        raise ValueError("eval_points_per_session must be non-negative")
    if config.eval_point_position_mode not in {"full-span", "interior"}:
        raise ValueError(
            "eval_point_position_mode must be 'full-span' or 'interior'"
        )
    if not config.eval_points_per_session and config.eval_point_position_mode != "full-span":
        raise ValueError(
            "eval_point_position_mode requires eval_points_per_session"
        )
    if config.max_eval_points_per_speaker and config.eval_points_per_session:
        raise ValueError(
            "max_eval_points_per_speaker and eval_points_per_session are mutually exclusive"
        )


def _is_full_protocol(
    config: RealTalkOursConfig, splits: list[dict[str, str]]
) -> bool:
    return (
        not config.speaker_filter
        and config.max_eval_points_per_speaker == 0
        and config.eval_points_per_session == 0
        and len(splits) == len(REALTALK_PERSONA_SPLITS)
    )


def _checkpoint_unresolved(checkpoint: OperationCheckpoint) -> list[dict[str, Any]]:
    return [
        {"operation_key": key, **value}
        for key, value in sorted(checkpoint.data["failures"].items())
    ]


def _checkpoint_usage(
    checkpoint: OperationCheckpoint | None,
    backend: ChatBackend,
) -> dict[str, int]:
    if checkpoint is None:
        return {
            key: int(value)
            for key, value in getattr(backend, "token_usage", {}).items()
        }
    total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    for operation in checkpoint.data["operations"].values():
        for key in total:
            total[key] += int(operation.get("token_usage", {}).get(key, 0))
    return total


def _failure(
    stage: str, speaker: str, result_id: str | None, exc: Exception
) -> dict[str, Any]:
    return {
        "stage": stage,
        "speaker": speaker,
        "result_id": result_id,
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
        "recorded_at_utc": _now(),
    }


def _turns_with_ids(turns: Iterable[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{turn['turn_id']}] {turn['speaker']}: {turn['content']}" for turn in turns
    )


def _turns_with_session_boundaries(turns: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    previous_session = ""
    for turn in turns:
        session_id = str(turn["session_id"])
        if session_id != previous_session:
            lines.append(f"--- {session_id} ---")
            previous_session = session_id
        lines.append(f"{turn['speaker']}: {turn['content']}")
    return "\n".join(lines)


def _target_spoke_in_session(
    turns: Iterable[dict[str, Any]], speaker: str, session_id: str
) -> bool:
    return any(
        turn["session_id"] == session_id
        and turn["speaker"].casefold() == speaker.casefold()
        for turn in turns
    )


def _select_even_points_per_session(
    points: list[dict[str, Any]], *, selected_count: int,
    position_mode: str = "full-span",
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    sessions = list(dict.fromkeys(point["target_session"] for point in points))
    for session_id in sessions:
        session_points = [
            point for point in points if point["target_session"] == session_id
        ]
        minimum_points = selected_count + 2 if position_mode == "interior" else selected_count
        if len(session_points) < minimum_points:
            raise ValueError(
                f"session {session_id} has {len(session_points)} targets; "
                f"cannot select {selected_count} {position_mode} points"
            )
        if position_mode == "interior":
            indices = [
                round((index + 1) * (len(session_points) - 1) / (selected_count + 1))
                for index in range(selected_count)
            ]
        elif selected_count == 1:
            indices = [len(session_points) // 2]
        else:
            indices = [
                round(index * (len(session_points) - 1) / (selected_count - 1))
                for index in range(selected_count)
            ]
        selected.extend(session_points[index] for index in indices)
    return selected


def _metric_display(summary: dict[str, Any] | None, metric: str) -> str:
    if not summary:
        return "pending"
    value = summary.get("speaker_macro", {}).get(metric)
    return value["display"] if value else "not computed"


def _split_order(speaker: str) -> int:
    names = [item["speaker"].casefold() for item in REALTALK_PERSONA_SPLITS]
    return names.index(speaker.casefold())


def _speaker_id(speaker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", speaker.casefold()).strip("_")


def _safe_host(base_url: str) -> str:
    match = re.match(r"https?://([^/]+)", base_url)
    return match.group(1) if match else base_url


def _clear_runtime_artifacts(output_dir: Path) -> None:
    for name in (
        "checkpoint.json",
        "checkpoint.json.tmp",
        "raw_responses.jsonl",
        "predictions.jsonl",
        "results_with_local_metrics.jsonl",
        "self_domains.json",
        "unresolved_errors.json",
        "dataset_manifest.json",
        "run_manifest.json",
        "local_metrics_summary.json",
        "table2_partial.json",
        "REPORT_PARTIAL.md",
        "GENERATION_COMPLETE",
        "LOCAL_METRICS_COMPLETE",
        "PIPELINE_COMPLETE",
        "GPT_EVALUATION_PENDING.json",
        "preflight.json",
    ):
        _remove(output_dir / name)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _remove(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> RealTalkOursConfig:
    parser = argparse.ArgumentParser(
        description="Run REALTALK Task 1 Ours Agentic V2 with fixed qwen3-max"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/realtalk_ours_agentic_v2_qwen3_max")
    parser.add_argument("--max-eval-points-per-speaker", type=int, default=0)
    parser.add_argument("--eval-points-per-session", type=int, default=0)
    parser.add_argument(
        "--eval-point-position-mode",
        choices=("full-span", "interior"),
        default="full-span",
    )
    parser.add_argument("--speaker", action="append", dest="speaker_filter")
    parser.add_argument("--skip-local-metrics", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument("--fresh", action="store_true")
    run_mode.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--decision-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--model-call-timeout-seconds", type=int, default=240)
    parser.add_argument("--model", choices=sorted(ALLOWED_MODELS), default=EXPECTED_MODEL)
    args = parser.parse_args()
    return RealTalkOursConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_eval_points_per_speaker=args.max_eval_points_per_speaker,
        eval_points_per_session=args.eval_points_per_session,
        eval_point_position_mode=args.eval_point_position_mode,
        speaker_filter=tuple(args.speaker_filter or ()),
        compute_local_metrics=not args.skip_local_metrics,
        compute_bertscore=not args.skip_bertscore,
        continue_on_error=not args.stop_on_error,
        preflight_only=args.preflight_only,
        fresh=args.fresh,
        resume=args.resume,
        decision_thinking=args.decision_thinking,
        model_call_timeout_seconds=args.model_call_timeout_seconds,
        model=args.model,
    )


if __name__ == "__main__":
    print(json.dumps(run_realtalk_ours(parse_args()), ensure_ascii=False, indent=2))

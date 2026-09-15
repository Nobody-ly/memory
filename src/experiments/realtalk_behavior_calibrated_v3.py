"""REALTALK Task 1 Ours V3: behavior-calibrated persona simulation.

The data preparation and checkpoint primitives are reused from the frozen V2
implementation. V3 never reads V2 outputs; it regenerates all private state.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Iterable

from . import realtalk_evidence_conditioned as base
from .operation_checkpoint import OperationCheckpoint
from .realtalk_evidence_schemas import (
    PROFILE_LAYERS,
    USER_DOMAIN_SCHEMA,
    empty_user_domain,
    normalize_user_domain,
    validate_evidence_ids,
)
from .realtalk_ours import _backend_from_env, _structured_call as _base_structured_call
from .realtalk_ours import _call_with_hard_timeout


PROTOCOL = "realtalk_task1_ours_behavior_calibrated_v3_2_contract_reliability"
MODEL = "deepseek-v4-flash"
_structured_call = partial(_base_structured_call, validate_schema=True, repair_raw_chars=1000000)
DECODING = {"self": 4096, "user": 8192, "decision": 2048, "actor": 300,
            "structured_attempts": 3, "actor_attempts": 3, "actor_repair_temperature": 0.0,
            "network_attempts_per_call": 3, "sdk_retries": 0}
SCENES = (
    "session_opening", "direct_question", "partner_affect",
    "partner_disclosure", "opinion_or_advice", "conversation_closure",
    "topic_continuation",
)
USUAL_ACTIONS = (
    "greet", "answer", "answer_then_self_disclose", "acknowledge",
    "acknowledge_then_self_disclose", "state_view_or_opinion",
    "brief_reflection", "ask_follow_up", "close", "mixed_or_unclear",
)
PRIMARY_ACTIONS = (
    "greet", "answer", "acknowledge", "self_disclose", "state_or_opinion",
    "brief_reflect", "close", "clarify", "mixed_answer",
)
QUESTION_SLOTS = ("q1", "q2", "q3", "q4")


def _schema_properties() -> dict[str, Any]:
    fact = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "maxLength": 160},
            "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["value", "evidence_ids", "confidence"],
        "additionalProperties": False,
    }
    scene = {
        "type": "object",
        "properties": {
            "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
            "sample_count": {"type": "integer", "minimum": 1, "maximum": 200},
            "usual_action": {"type": "string", "enum": list(USUAL_ACTIONS)},
            "question_tendency": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
            "reflection_tendency": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
            "disclosure_tendency": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
            "length_tendency": {"type": "string", "enum": ["short", "typical", "long", "unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["evidence_ids", "sample_count", "usual_action", "question_tendency",
                      "reflection_tendency", "disclosure_tendency", "length_tendency", "confidence"],
        "additionalProperties": False,
    }
    scene_or_empty = {"oneOf": [{"type": "object", "properties": {}, "required": [], "additionalProperties": False}, scene]}
    return {
        "type": "object",
        "properties": {
            "identity_facts": {"type": "array", "items": fact, "maxItems": 6},
            "voice_profile": {"type": "array", "items": fact, "maxItems": 4},
            "social_profile": {"type": "array", "items": fact, "maxItems": 4},
            "behavior_by_scene": {"type": "object", "properties": {s: scene_or_empty for s in SCENES}, "required": list(SCENES), "additionalProperties": False},
            "uncertainties": {"type": "array", "items": {"type": "string", "maxLength": 240}, "maxItems": 4},
            "observable_statistics": {"type": "object", "additionalProperties": True},
        },
        "required": ["identity_facts", "voice_profile", "social_profile", "behavior_by_scene", "uncertainties", "observable_statistics"],
        "additionalProperties": False,
    }


SELF_DOMAIN_SCHEMA = {"name": "realtalk_behavior_calibrated_self_domain_v3", "strict": True, "schema": _schema_properties()}


DECISION_SCHEMA = {
    "name": "realtalk_behavior_calibrated_decision_v3",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "situation": {"type": "object", "properties": {
                "scene": {"type": "string", "enum": list(SCENES) + ["unclear"]},
                "partner_act": {"type": "string", "enum": ["question", "statement", "affect", "opinion_request", "closure", "opening", "unclear"]},
                "topic": {"type": "string", "maxLength": 240}, "uncertainty": {"type": "string", "enum": ["low", "medium", "high"]},
            }, "required": ["scene", "partner_act", "topic", "uncertainty"], "additionalProperties": False},
            "user_state": {"type": "object", "properties": {
                "interaction_need": {"type": "string", "enum": ["information_exchange", "emotional_acknowledgment", "reciprocal_sharing", "clarification", "topic_continuation", "closing", "unclear"]},
                "affect": {"type": "string", "enum": ["positive", "neutral", "negative", "mixed", "unclear"]},
                "affect_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "topic_continuity": {"type": "string", "enum": ["continue", "new", "unclear"]},
                "response_pressure": {"type": "string", "enum": ["low", "medium", "high"]},
            }, "required": ["interaction_need", "affect", "affect_confidence", "topic_continuity", "response_pressure"], "additionalProperties": False},
            "relevant_user_domain": {"type": "array", "items": {"type": "string", "maxLength": 240}, "maxItems": 2},
            "alignment": {"type": "object", "properties": {
                "orientation": {"type": "string", "enum": ["self_led", "balanced", "partner_adaptive"]},
                "lambda_trace": {"type": "number", "minimum": 0, "maximum": 1},
                "basis": {"type": "string", "maxLength": 320},
                "affected_dimensions": {"type": "array", "items": {"type": "string", "enum": ["content_focus", "questioning", "length", "tone", "self_disclosure", "reflection", "topic", "empathy", "intimacy"]}, "minItems": 1, "maxItems": 4},
            }, "required": ["orientation", "lambda_trace", "basis", "affected_dimensions"], "additionalProperties": False},
            "behavior_policy": {"type": "object", "properties": {
                "primary_action": {"type": "string", "enum": list(PRIMARY_ACTIONS)},
                "selected_question_slots": {"type": "array", "items": {"type": "string", "enum": list(QUESTION_SLOTS)}, "maxItems": 4},
                "outbound_question_mode": {"type": "string", "enum": ["none", "opening", "reciprocal", "clarifying", "follow_up"]},
                "outbound_question_focus": {"type": "string", "maxLength": 240},
                "reflection_mode": {"type": "string", "enum": ["none", "brief", "supported"]},
                "self_disclosure_mode": {"type": "string", "enum": ["none", "brief", "natural"]},
                "grounding_mode": {"type": "string", "enum": ["none", "specific_acknowledgment", "clarifying_question"]},
                "empathy_mode": {"type": "string", "enum": ["none", "light", "moderate"]},
                "intimacy_mode": {"type": "string", "enum": ["match", "slightly_warm", "reserved"]},
                "message_length": {"type": "string", "enum": ["short", "typical"]},
                "required_content_slots": {"type": "array", "items": {"type": "string", "maxLength": 240}, "minItems": 1, "maxItems": 4},
                "forbidden_additions": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 6},
            }, "required": ["primary_action", "selected_question_slots", "outbound_question_mode", "outbound_question_focus", "reflection_mode", "self_disclosure_mode", "grounding_mode", "empathy_mode", "intimacy_mode", "message_length", "required_content_slots", "forbidden_additions"], "additionalProperties": False},
            "evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        },
        "required": ["situation", "user_state", "relevant_user_domain", "alignment", "behavior_policy", "evidence_ids"],
        "additionalProperties": False,
    },
}


SELF_SYSTEM_PROMPT = """You compile an evidence-grounded private Self Domain for a REALTALK persona simulation task.

The target is a real conversational participant, not an assistant, therapist, coach, evaluator, or idealized
empathetic personality. Use only the target speaker's own messages as evidence for identity, style, and behavior.
Partner messages may explain the interaction scene but must never become target facts.

identity_facts contains durable self facts. voice_profile contains only surface language form such as wording,
register, punctuation, brevity, message shape, and stylistic markers. Do not put asks_questions, answers,
follows_up, reflects, self_discloses, supports, comforts, or greets in voice_profile; those turn behaviors belong
only in behavior_by_scene. social_profile contains stable relational stance or boundaries, not a turn instruction.

Fill the seven fixed behavior scene keys independently: session_opening, direct_question, partner_affect,
partner_disclosure, opinion_or_advice, conversation_closure, and topic_continuation. A direct question must be
classified as direct_question. A factual partner sharing is partner_disclosure. An opinion request is
opinion_or_advice. A closing exchange is conversation_closure. Never put every observation under partner_affect.
If evidence for a scene is insufficient, return an empty object. Do not infer a tendency from one example.
sample_count counts target messages supporting that scene. Describe what the target actually does. Do not turn
empathy principles or evaluation criteria into personality traits. Preserve uncertainty and conflicting identities.
Do not draft a reply or invent quotations. Return only strict JSON."""


USER_SYSTEM_PROMPT = """You update the private five-layer long-term User Domain of the current conversation partner.
Use exactly core, regulation, cognition, identity, and behavior. Store durable partner facts, repeated preferences,
recurring interaction patterns, stable communication tendencies, and repeated decision patterns. Do not store the
latest question, temporary mood, one-off event, or current conversational act as a stable fact. Do not infer a
psychological trait from one message. Do not copy facts about the target speaker into the partner model. Revise
conflicts and leave unsupported layers empty. Every evidence ID must come from the whitelist. Return only strict JSON."""


DECISION_SYSTEM_PROMPT = """You are the private situation, user-state, adaptive-alignment, and behavior-policy controller
for a REALTALK persona simulation task.

Predict what the target person would naturally say at this exact point. The target is a real person, not an assistant
optimizing empathy scores. Read the complete real history before the target turn. Use the Self Domain as the stable
identity prior, the five-layer User Domain only when relevant, the deterministic scene gate and question slots,
the Ca behavior prior, and already observed Cb behavior. Do not use any future turn, target answer, Judge label,
previous generated message, or evaluation metric.

A new session with no current partner message is session_opening, not an incoming question. A factual partner
statement is not automatically emotional support. selected_question_slots may contain only question slots that
exist in the supplied partner message and must be answered. They never authorize the target to ask a question.
Use outbound_question_mode and outbound_question_focus for a target-originated opening, reciprocal, clarifying,
or follow-up question; use none and an empty focus when no outbound question is selected. Choose exactly one
primary_action and one concrete set of content slots. Optional behavior is not permission to add it: if it is not
selected, it must not be generated. Do not add reflection, grounding, self-disclosure, empathy, praise, therapy
language, or a follow-up question merely to sound helpful. Grounding is a specific response to partner content
or a specific clarification; it is not equivalent to any question mark.

lambda_trace is an auditable balance between stable target behavior and current-turn adaptation. It is not a score,
quota, empathy value, or sentence control. Keep identity as a hard constraint while allowing small evidence-based
adaptation. The orientation and affected dimensions must match the one concrete policy.

Use one policy object only. Never place a behavior in both a selected field and a forbidden field. grounding_mode
may be clarifying_question only when outbound_question_mode is clarifying. If reflection_mode is none, do not require reflection.
Moderate empathy requires an explicit affective partner signal. required_content_slots is a variable-length array
of 1 to 4 nonempty content intentions, not a fixed number of bubbles. Decide the intentions once; do not emit
message_shape. The pipeline derives composition from this array. message_length only chooses short or typical.
If outbound_question_mode is none, outbound_question_focus must be empty. Otherwise it must name the question
topic. grounding_mode is clarifying_question if and only if outbound_question_mode is clarifying.
Answering the partner's selected_question_slots never authorizes a new question.
Return only strict JSON and do not draft the final message."""


ACTOR_SYSTEM_PROMPT = """You are {speaker}. Continue the conversation as this person.

Produce only the target person's next message at this exact point. Use the complete real history, private Self Domain,
current User State, and the selected Behavior Policy. Execute the selected primary_action and required content slots.
The selected Behavior Policy is authoritative for this turn and overrides any general behavioral tendency.
Do not invent another question, reflection, emotional interpretation, support statement, topic, or personal fact when
it was not selected. When multiple question slots are selected, answer them in order. Match the target's observed
length and message-shape tendency for this scene while preserving natural language.

The target is a real conversational participant, not an assistant, therapist, evaluator, or empathy system. Do not
mention private fields, policies, metrics, or reasoning. Output only message text, with no JSON and no speaker label."""


@dataclass(frozen=True)
class Config:
    dataset_dir: str = "dataset"
    output_dir: str = "data/realtalk_behavior_calibrated_v3_gate6"
    mode: str = "cb"
    gate: int = 6
    model: str = MODEL
    operation_max_attempts: int = 3
    timeout_seconds: int = 240
    fresh: bool = False
    resume: bool = False
    parent_output: str | None = None
    preflight_only: bool = False
    selected_ids_file: str | None = None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_json(value) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _exact(value: Any, required: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != required:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{path} fields mismatch: missing={sorted(required-actual)} extra={sorted(actual-required)}")
    return value


def _text(value: Any, path: str, max_length: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be non-empty text")
    if len(value.strip()) > max_length:
        raise ValueError(f"{path} exceeds max length")
    return value.strip()


def _enum(value: Any, values: tuple[str, ...] | list[str], path: str) -> str:
    value = _text(value, path, 100)
    if value not in values:
        raise ValueError(f"{path} must be one of {list(values)}")
    return value


def _float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{path} must be in [0,1]")
    return round(float(value), 6)


def _strings(value: Any, path: str, max_items: int = 20, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or len(value) > max_items:
        raise ValueError(f"{path} must be a list with at most {max_items} items")
    result = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{path}[{i}] must be text")
        if nonempty and not item.strip():
            raise ValueError(f"{path}[{i}] is empty")
        result.append(item.strip())
    return result


def _normalize_user_v3(value: Any, allowed_ids: set[str]) -> dict[str, Any]:
    """Merge only exact duplicate profile facts before strict validation."""
    if not isinstance(value, dict):
        return validate_evidence_ids(normalize_user_domain(value), allowed_ids)

    prepared = dict(value)
    expected_fact_keys = {"value", "evidence_ids", "confidence"}
    for layer in PROFILE_LAYERS:
        raw_facts = value.get(layer)
        if not isinstance(raw_facts, list):
            continue

        merged: list[Any] = []
        positions: dict[str, int] = {}
        for raw in raw_facts:
            if (
                not isinstance(raw, dict)
                or set(raw) != expected_fact_keys
                or not isinstance(raw.get("value"), str)
                or not isinstance(raw.get("evidence_ids"), list)
                or not all(isinstance(item, str) for item in raw["evidence_ids"])
                or not isinstance(raw.get("confidence"), (int, float))
            ):
                merged.append(raw)
                continue

            key = " ".join(raw["value"].split()).casefold()
            if not key or key not in positions:
                positions[key] = len(merged)
                merged.append(dict(raw))
                continue

            previous = merged[positions[key]]
            combined_ids = list(previous["evidence_ids"])
            for evidence_id in raw["evidence_ids"]:
                if evidence_id not in combined_ids:
                    combined_ids.append(evidence_id)
            invalid = set(combined_ids) - allowed_ids
            if invalid:
                raise ValueError(f"invalid evidence IDs: {sorted(invalid)}")
            previous["evidence_ids"] = combined_ids[:4]
            previous["confidence"] = max(previous["confidence"], raw["confidence"])

        prepared[layer] = merged

    return validate_evidence_ids(normalize_user_domain(prepared), allowed_ids)


def _normalize_self(value: Any, allowed_ids: set[str]) -> dict[str, Any]:
    root = _exact(value, {"identity_facts", "voice_profile", "social_profile", "behavior_by_scene", "uncertainties", "observable_statistics"}, "self_domain")
    result: dict[str, Any] = {"identity_facts": [], "voice_profile": [], "social_profile": [], "uncertainties": _strings(root["uncertainties"], "uncertainties", 4)}
    section_limits = {"identity_facts": 6, "voice_profile": 4, "social_profile": 4}
    for section in ("identity_facts", "voice_profile", "social_profile"):
        if not isinstance(root[section], list) or len(root[section]) > section_limits[section]:
            raise ValueError(f"{section} must be a compact list of at most {section_limits[section]} items")
        for i, raw in enumerate(root[section]):
            item = _exact(raw, {"value", "evidence_ids", "confidence"}, f"{section}[{i}]")
            ids = _strings(item["evidence_ids"], f"{section}[{i}].evidence_ids", 4, True)
            if set(ids) - allowed_ids:
                raise ValueError(f"{section}[{i}] cites invisible evidence")
            normalized_value = _text(item["value"], f"{section}[{i}].value", 160)
            result[section].append({"value": normalized_value, "evidence_ids": ids, "confidence": _float(item["confidence"], f"{section}[{i}].confidence")})
    scenes = _exact(root["behavior_by_scene"], set(SCENES), "behavior_by_scene")
    result["behavior_by_scene"] = {}
    for scene_name in SCENES:
        raw = scenes[scene_name]
        if raw == {}:
            result["behavior_by_scene"][scene_name] = {}
            continue
        item = _exact(raw, {"evidence_ids", "sample_count", "usual_action", "question_tendency", "reflection_tendency", "disclosure_tendency", "length_tendency", "confidence"}, scene_name)
        if not isinstance(item["sample_count"], int) or not 1 <= item["sample_count"] <= 200:
            raise ValueError(f"{scene_name}.sample_count invalid")
        ids = _strings(item["evidence_ids"], f"{scene_name}.evidence_ids", 4, True)
        if set(ids) - allowed_ids:
            raise ValueError(f"{scene_name} cites invisible evidence")
        result["behavior_by_scene"][scene_name] = {"evidence_ids": ids, "sample_count": item["sample_count"], "usual_action": _enum(item["usual_action"], USUAL_ACTIONS, f"{scene_name}.usual_action"), "question_tendency": _enum(item["question_tendency"], ("low", "medium", "high", "unknown"), f"{scene_name}.question_tendency"), "reflection_tendency": _enum(item["reflection_tendency"], ("low", "medium", "high", "unknown"), f"{scene_name}.reflection_tendency"), "disclosure_tendency": _enum(item["disclosure_tendency"], ("low", "medium", "high", "unknown"), f"{scene_name}.disclosure_tendency"), "length_tendency": _enum(item["length_tendency"], ("short", "typical", "long", "unknown"), f"{scene_name}.length_tendency"), "confidence": _float(item["confidence"], f"{scene_name}.confidence")}
    if not isinstance(root["observable_statistics"], dict):
        raise ValueError("observable_statistics must be an object")
    result["observable_statistics"] = root["observable_statistics"]
    return result


def _normalize_decision(value: Any, allowed_ids: set[str], context: dict[str, Any]) -> dict[str, Any]:
    root = _exact(value, {"situation", "user_state", "relevant_user_domain", "alignment", "behavior_policy", "evidence_ids"}, "decision")
    situation = _exact(root["situation"], {"scene", "partner_act", "topic", "uncertainty"}, "situation")
    state = _exact(root["user_state"], {"interaction_need", "affect", "affect_confidence", "topic_continuity", "response_pressure"}, "user_state")
    alignment = _exact(root["alignment"], {"orientation", "lambda_trace", "basis", "affected_dimensions"}, "alignment")
    policy = _exact(root["behavior_policy"], {"primary_action", "selected_question_slots", "outbound_question_mode", "outbound_question_focus", "reflection_mode", "self_disclosure_mode", "grounding_mode", "empathy_mode", "intimacy_mode", "message_length", "required_content_slots", "forbidden_additions"}, "behavior_policy")
    scene = _enum(situation["scene"], list(SCENES) + ["unclear"], "situation.scene")
    slots = _strings(policy["selected_question_slots"], "selected_question_slots", 4)
    if any(slot not in QUESTION_SLOTS for slot in slots) or len(set(slots)) != len(slots):
        raise ValueError("selected_question_slots must be unique q slots")
    available_slots = {slot["slot_id"] for slot in context.get("question_slots", [])}
    if set(slots) - available_slots:
        raise ValueError("selected_question_slots cites a nonexistent partner question")
    outbound_mode = _enum(policy["outbound_question_mode"], ("none", "opening", "reciprocal", "clarifying", "follow_up"), "outbound_question_mode")
    if not isinstance(policy["outbound_question_focus"], str) or len(policy["outbound_question_focus"].strip()) > 240:
        raise ValueError("outbound_question_focus must be text with at most 240 characters")
    outbound_focus = policy["outbound_question_focus"].strip()
    if (outbound_mode == "none") != (outbound_focus == ""):
        raise ValueError("outbound question mode and focus disagree")
    if outbound_mode == "opening" and scene != "session_opening":
        raise ValueError("opening question is valid only at session opening")
    if scene == "session_opening" and outbound_mode not in {"none", "opening"}:
        raise ValueError("session opening allows only none or opening outbound question mode")
    grounding = _enum(policy["grounding_mode"], ("none", "specific_acknowledgment", "clarifying_question"), "grounding_mode")
    if (grounding == "clarifying_question") != (outbound_mode == "clarifying"):
        raise ValueError("clarifying grounding and outbound question mode disagree")
    reflection = _enum(policy["reflection_mode"], ("none", "brief", "supported"), "reflection_mode")
    required_slots = _strings(policy["required_content_slots"], "required_content_slots", 4, True)
    if reflection == "none" and any("reflect" in x.casefold() for x in required_slots):
        raise ValueError("reflection is required while reflection_mode is none")
    if not required_slots:
        raise ValueError("required_content_slots needs 1 to 4 nonempty intentions")
    message_length = _enum(policy["message_length"], ("short", "typical"), "message_length")
    message_shape = "multi_content" if len(required_slots) > 1 else f"single_{message_length}"
    if policy["empathy_mode"] == "moderate" and scene not in {"partner_affect"}:
        raise ValueError("moderate empathy requires partner_affect")
    evidence = _strings(root["evidence_ids"], "evidence_ids", 8)
    if set(evidence) - allowed_ids:
        raise ValueError("decision cites invisible evidence")
    if not isinstance(state["affect_confidence"], (int, float)):
        raise ValueError("affect_confidence invalid")
    return {
        "situation": {"scene": scene, "partner_act": _enum(situation["partner_act"], ("question", "statement", "affect", "opinion_request", "closure", "opening", "unclear"), "partner_act"), "topic": _text(situation["topic"], "topic", 240), "uncertainty": _enum(situation["uncertainty"], ("low", "medium", "high"), "uncertainty")},
        "user_state": {"interaction_need": _enum(state["interaction_need"], ("information_exchange", "emotional_acknowledgment", "reciprocal_sharing", "clarification", "topic_continuation", "closing", "unclear"), "interaction_need"), "affect": _enum(state["affect"], ("positive", "neutral", "negative", "mixed", "unclear"), "affect"), "affect_confidence": _float(state["affect_confidence"], "affect_confidence"), "topic_continuity": _enum(state["topic_continuity"], ("continue", "new", "unclear"), "topic_continuity"), "response_pressure": _enum(state["response_pressure"], ("low", "medium", "high"), "response_pressure")},
        "relevant_user_domain": _strings(root["relevant_user_domain"], "relevant_user_domain", 2),
        "alignment": {"orientation": _enum(alignment["orientation"], ("self_led", "balanced", "partner_adaptive"), "orientation"), "lambda_trace": _float(alignment["lambda_trace"], "lambda_trace"), "basis": _text(alignment["basis"], "basis", 320), "affected_dimensions": _strings(alignment["affected_dimensions"], "affected_dimensions", 4, True)},
        "behavior_policy": {"primary_action": _enum(policy["primary_action"], PRIMARY_ACTIONS, "primary_action"), "selected_question_slots": slots, "outbound_question_mode": outbound_mode, "outbound_question_focus": outbound_focus, "reflection_mode": reflection, "self_disclosure_mode": _enum(policy["self_disclosure_mode"], ("none", "brief", "natural"), "self_disclosure_mode"), "grounding_mode": grounding, "empathy_mode": _enum(policy["empathy_mode"], ("none", "light", "moderate"), "empathy_mode"), "intimacy_mode": _enum(policy["intimacy_mode"], ("match", "slightly_warm", "reserved"), "intimacy_mode"), "message_shape": message_shape, "required_content_slots": required_slots, "forbidden_additions": _strings(policy["forbidden_additions"], "forbidden_additions", 6)},
        "evidence_ids": evidence,
    }


def _words(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text.casefold()))


def _scene_for(text: str, opening: bool = False) -> str:
    if opening:
        return "session_opening"
    if _words(text, r"\b(bye|goodbye|good night|goodnight|talk later|see you)\b"):
        return "conversation_closure"
    if "?" in text or "？" in text or _words(text, r"\b(what do you think|how do you feel|why do you|could you tell me)\b"):
        if _words(text, r"\b(should|recommend|suggest|advice|what do you think)\b"):
            return "opinion_or_advice"
        return "direct_question"
    if _words(text, r"\b(sad|happy|excited|worried|afraid|upset|love|hate|anxious|lonely|stressed|frustrated|miss)\b"):
        return "partner_affect"
    if len(text.strip()) >= 35:
        return "partner_disclosure"
    return "topic_continuation"


def behavior_calibrator(turns: list[dict[str, Any]], speaker: str, session: str, ca_turns: list[dict[str, Any]]) -> dict[str, Any]:
    current = [t for t in turns if t["session_id"] == session]
    partner = next((t for t in reversed(current) if t["speaker"].casefold() != speaker.casefold()), None)
    partner_text = str(partner["content"]) if partner else ""
    scene = _scene_for(partner_text, opening=partner is None)
    question_count = partner_text.count("?") + partner_text.count("？")
    slots = [{"slot_id": f"q{i+1}", "text_span": partner_text[:240], "topic_hint": scene} for i in range(min(question_count, 4))]
    target_ca = [t for t in ca_turns if t["speaker"].casefold() == speaker.casefold()]
    target_cb = [t for t in turns if t["speaker"].casefold() == speaker.casefold()]
    def marker(t: dict[str, Any], name: str) -> bool:
        text = str(t["content"])
        return {"question": "?" in text or "？" in text, "reflection": _words(text, r"\b(i think|i feel|i guess|because|in my opinion)\b"), "disclosure": _words(text, r"\b(i|i'm|i've|my|mine)\b"), "multi": len(t.get("dia_ids", [])) > 1}[name]
    def beta_rate(items: list[dict[str, Any]], name: str) -> float:
        return (sum(marker(t, name) for t in items) + 1) / (len(items) + 2)
    ca_rate = beta_rate(target_ca, "question")
    observed = [t for t in target_cb if t.get("session_id") == session]
    observed_count = len(observed)
    cb_rate = (8 * ca_rate + sum(marker(t, "question") for t in observed)) / (8 + observed_count)
    return {"scene": scene, "session_opening": partner is None, "partner_has_question_mark": question_count > 0, "partner_has_multiple_questions": question_count > 1, "partner_has_affect_signal": _scene_for(partner_text) == "partner_affect", "partner_has_specific_detail": len(partner_text) >= 35, "partner_has_closure_signal": scene == "conversation_closure", "current_partner_text_hash": _hash(partner_text), "latest_partner_message": partner_text, "question_slot_count": question_count, "question_slots": slots, "behavior_prior": {"ca_question_rate": round(ca_rate, 4), "cb_posterior_question_rate": round(cb_rate, 4), "cb_observed_count": observed_count, "ca_target_message_count": len(target_ca)}}


def _stats(turns: list[dict[str, Any]], speaker: str) -> dict[str, Any]:
    target = [t for t in turns if t["speaker"].casefold() == speaker.casefold()]
    lengths = [len(str(t["content"])) for t in target]
    return {"target_message_count": len(target), "mean_characters": round(sum(lengths) / len(lengths), 3) if lengths else 0, "median_characters": sorted(lengths)[len(lengths)//2] if lengths else 0, "question_rate": round(sum("?" in str(t["content"]) or "？" in str(t["content"]) for t in target) / len(target), 3) if target else 0, "reflection_marker_rate": round(sum(_words(str(t["content"]), r"\b(i think|i feel|i guess|because|in my opinion)\b") for t in target) / len(target), 3) if target else 0, "self_disclosure_rate": round(sum(_words(str(t["content"]), r"\b(i|i'm|i've|my|mine)\b") for t in target) / len(target), 3) if target else 0, "multi_bubble_rate": round(sum(len(t.get("dia_ids", [])) > 1 for t in target) / len(target), 3) if target else 0}


def _prompt_hashes() -> dict[str, str]:
    prompts = {
        "self_system": SELF_SYSTEM_PROMPT,
        "user_system": USER_SYSTEM_PROMPT,
        "decision_system": DECISION_SYSTEM_PROMPT,
        "actor_system": ACTOR_SYSTEM_PROMPT,
        "self_user_template": inspect.getsource(_domain_prompt),
        "user_user_template": inspect.getsource(_user_prompt),
        "decision_user_template": inspect.getsource(_decision_prompt),
        "actor_user_template": inspect.getsource(_actor_prompt),
        "actor_retry_contract": inspect.getsource(_actor_retry_instruction),
        "actor_retry_template": inspect.getsource(_actor_retry_suffix),
    }
    return {name: _hash(value) for name, value in prompts.items()}


def _normalize_actor(text: str, speaker: str, decision: dict[str, Any]) -> str:
    message = text.strip()
    if not message:
        raise ValueError("actor returned empty text")
    if re.match(rf"^{re.escape(speaker)}\s*:", message, re.I):
        raise ValueError("actor leaked speaker label")
    if message.startswith(("{", "[")) or re.search(r'"(?:behavior_policy|user_state|self_domain|lambda_trace)"\s*:', message, re.I):
        raise ValueError("actor leaked private structure")
    policy = decision["behavior_policy"]
    question_count = message.count("?") + message.count("？")
    outbound_question_allowed = policy["outbound_question_mode"] != "none"
    if not outbound_question_allowed and question_count:
        raise ValueError("actor added an unselected outbound question")
    if outbound_question_allowed and question_count < 1:
        raise ValueError(f"selected outbound question requires at least one question about {policy['outbound_question_focus']!r}")
    # Lexical reflection markers are not a reliable semantic rejection rule.
    return message


def _actor_retry_instruction(error: str) -> str:
    if "unselected outbound question" in error:
        return (
            "Remove every question and every question mark. Do not copy or paraphrase an "
            "interrogative from the history or Self Domain. Keep only the selected non-question content."
        )
    if "requires at least one question" in error:
        return "Add the selected outbound question about its exact focus; do not add another topic."
    return "Correct only the stated contract violation and keep the same selected policy."


def _actor_retry_suffix(error: str, rejected_draft: str) -> str:
    return (
        f"\n\nREJECTED DRAFT FROM PREVIOUS ATTEMPT:\n{rejected_draft}\n"
        f"CONTRACT ERROR: {error}\n"
        f"REPAIR INSTRUCTION: {_actor_retry_instruction(error)}\n"
        "Return only the corrected message."
    )


def _actor_call(checkpoint: OperationCheckpoint, backend: Any, key: str, speaker: str, prompt: str, decision: dict[str, Any], raw_audit: Path, max_attempts: int, timeout: int) -> dict[str, Any]:
    feedback = {"error": "", "draft": ""}
    attempts = {"n": 0}
    def operation():
        attempts["n"] += 1
        suffix = _actor_retry_suffix(feedback["error"], feedback["draft"]) if feedback["error"] else ""
        system = ACTOR_SYSTEM_PROMPT.format(speaker=speaker)
        if feedback["error"]:
            system += "\nThis is a contract repair, not a new continuation. " + _actor_retry_instruction(feedback["error"])
        return _call_with_hard_timeout(lambda: backend.chat(system, prompt + suffix,
            temperature=0.0 if feedback["error"] else 0.6, top_p=0.9,
            max_tokens=DECODING["actor"], enable_thinking=False), timeout, key)
    def validate(result):
        with raw_audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"operation_key": key, "logical_attempt": attempts["n"], "model": result.model, "raw_response": result.content, "reasoning_content": result.reasoning_content, "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens, "response_id": result.response_id, "finish_reason": result.finish_reason, "recorded_at_utc": _now()}, ensure_ascii=False) + "\n")
        try:
            if result.finish_reason == "length":
                raise ValueError("actor output was truncated; finish the same plan within the output budget")
            message = _normalize_actor(result.content, speaker, decision)
        except Exception as exc:
            feedback["error"] = str(exc)
            feedback["draft"] = result.content.strip()
            raise
        return {"data": message, "audit": {"model": result.model, "logical_attempts": attempts["n"], "thinking_enabled": False, "response_id": result.response_id, "finish_reason": result.finish_reason}}
    return checkpoint.execute(key, operation, validate, max_attempts)


def _domain_prompt(item: dict[str, Any], stats: dict[str, Any]) -> str:
    ids = sorted(base.evidence_ids(item["reference_turns"], item["speaker"]))
    return f"TARGET SPEAKER: {item['speaker']}\nCOMPLETE Ca HISTORY:\n{base.format_evidence_turns(item['reference_turns'])}\n\nDETERMINISTIC OBSERVABLE STATISTICS:\n{_json(stats)}\n\nALLOWED TARGET EVIDENCE IDS:\n{_json(ids)}\n\nCOMPACT OUTPUT BUDGET: return at most 6 identity facts, 4 voice facts, 4 social facts, 4 uncertainties, and at most 4 evidence IDs per scene. Keep each value short and evidence-grounded. Empty unsupported scenes are preferred to verbose speculation."


def _user_prompt(speaker: str, partner: str, previous: dict[str, Any], completed: list[dict[str, Any]], allowed: set[str]) -> str:
    target_ids = sorted(base.evidence_ids(completed, speaker))
    return f"TARGET SPEAKER: {speaker}\nPARTNER: {partner}\nPREVIOUS USER DOMAIN:\n{_json(previous)}\nCOMPLETE FINISHED SESSION:\n{base.format_evidence_turns(completed)}\nALLOWED PARTNER EVIDENCE IDS (copy only these):\n{_json(sorted(allowed))}\nFORBIDDEN TARGET-SPEAKER EVIDENCE IDS (never cite these):\n{_json(target_ids)}\nThe output is a model of {partner}, not {speaker}. Every stored fact must be supported by one or more allowed partner IDs; target-speaker IDs are invalid even when the target message contains useful information."


def _decision_prompt(item: dict[str, Any], point: dict[str, Any], self_domain: dict[str, Any], user_domain: dict[str, Any], gate: dict[str, Any], stats: dict[str, Any]) -> str:
    visible = base.evidence_ids(point["context_turns"]) | base.evidence_ids(item["reference_turns"])
    return f"TARGET SPEAKER: {item['speaker']}\nPARTNER: {item['partner']}\n\nFIXED SELF DOMAIN:\n{_json(self_domain)}\n\nCURRENT FIVE-LAYER USER DOMAIN:\n{_json(user_domain)}\n\nCA BEHAVIOR STATISTICS:\n{_json(stats)}\n\nDETERMINISTIC CURRENT SCENE GATE:\n{_json(gate)}\n\nCOMPLETE REAL HISTORY BEFORE TARGET:\n{base.format_evidence_turns(point['context_turns'])}\n\nVISIBLE EVIDENCE IDS:\n{_json(sorted(visible))}\n\nCURRENT SESSION: {point['target_session']}"


def _actor_prompt(item: dict[str, Any], point: dict[str, Any], self_domain: dict[str, Any], decision: dict[str, Any], gate: dict[str, Any]) -> str:
    policy = decision["behavior_policy"]
    outbound_question_allowed = policy["outbound_question_mode"] != "none"
    turn_behavior_pattern = r"\b(ask|asks|question|questions|answer|answers|respond|responds|reply|replies|follow.?up|reflect|reflects|self.?disclos|support|supports|comfort|comforts|greet|greets|greeting|greetings|check.?in|check.?ins)\b"
    actor_self_view = {
        "identity_facts": self_domain["identity_facts"],
        "voice_profile": [
            fact for fact in self_domain["voice_profile"]
            if not _words(fact["value"], turn_behavior_pattern)
            and (outbound_question_allowed or "?" not in fact["value"] and "？" not in fact["value"])
        ],
    }
    question_contract = f"Selected partner-question slots to ANSWER: {len(policy['selected_question_slots'])}. Answer those slots; they never authorize a new question. Outbound question mode: {policy['outbound_question_mode']}. Outbound question focus: {policy['outbound_question_focus'] or 'none'}. "
    question_contract += (
        "You must include at least one natural question about the selected focus before ending; every question must serve the selected mode and focus."
        if outbound_question_allowed
        else "The message MUST contain no question mark and must not ask any question."
    )
    return f"CURRENT REAL HISTORY BEFORE TARGET:\n{base.format_evidence_turns(point['context_turns'])}\n\nPRIVATE SELF DOMAIN IDENTITY AND EXPRESSION VIEW:\n{_json(actor_self_view)}\n\nCURRENT USER STATE:\n{_json(decision['user_state'])}\n\nSELECTED BEHAVIOR POLICY:\n{_json(policy)}\n\nCURRENT SCENE GATE:\n{_json(gate)}\n\nHARD MESSAGE CONTRACT:\nThe selected policy overrides any general tendency for this turn. Execute the selected primary action and required content only. {question_contract} Do not add a greeting question, follow-up question, reflection, self-disclosure, or topic that is not selected.\n\nWrite only {item['speaker']}'s next message."


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(dataset_manifest: dict[str, Any], selected_ids: list[str], gate: int, parent_output: Path | None = None, selected_ids_file: str | None = None) -> dict[str, Any]:
    manifest = {"protocol": PROTOCOL, "mode": "cb", "gate": gate, "model": MODEL, "thinking_enabled_all_stages": False, "dataset": dataset_manifest, "selected_ids_sha256": _hash(selected_ids), "selected_ids_source": selected_ids_file, "prompt_hashes": _prompt_hashes(), "schema_hashes": {"self": _hash(SELF_DOMAIN_SCHEMA), "user": _hash(USER_DOMAIN_SCHEMA), "decision": _hash(DECISION_SCHEMA)}, "history_compression_enabled": False, "history_truncation_enabled": False, "generated_outputs_rolled_into_history": False, "omega_enabled": False, "future_user_state_enabled": False, "semantic_verification_or_candidate_search": False, "v2_outputs_read": False, "judge_labels_read": False, "ground_truth_read_by_generation": False}
    manifest["decoding"] = DECODING
    manifest["implementation_hashes"] = {name: _file_sha256(Path(__file__).with_name(name)) for name in
        ("realtalk_behavior_calibrated_v3.py", "realtalk_ours.py", "realtalk_evidence_schemas.py", "operation_checkpoint.py", "realtalk_evidence_conditioned.py", "exp1_protocol.py")}
    manifest["implementation_hashes"]["client.py"] = _file_sha256(Path(__file__).parent / "personaemp/client.py")
    manifest["diagnostic_only"] = bool(selected_ids_file)
    if parent_output is not None:
        manifest["parent_output"] = str(parent_output)
        manifest["parent_manifest_sha256"] = _file_sha256(parent_output / "manifest.json")
    return manifest


def _seed_parent_checkpoint(checkpoint: OperationCheckpoint, parent_output: Path, selected_ids: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_manifest_path = parent_output / "manifest.json"
    parent_checkpoint_path = parent_output / "checkpoint.json"
    if not parent_manifest_path.exists() or not parent_checkpoint_path.exists():
        raise ValueError("parent output must contain manifest.json and checkpoint.json")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    current = _manifest(parent_manifest["dataset"], [], 0)
    for field in ("prompt_hashes", "schema_hashes", "implementation_hashes", "decoding"):
        if parent_manifest.get(field) != current.get(field):
            raise ValueError(f"parent {field} mismatch; do not mix different implementations")
    if parent_manifest.get("protocol") != PROTOCOL or parent_manifest.get("status") != "generation_complete":
        raise ValueError("parent output is not a complete run of the same V3.1.1 protocol")
    if parent_manifest.get("unresolved_count", 1) != 0:
        raise ValueError("parent output contains unresolved errors")
    parent_checkpoint = json.loads(parent_checkpoint_path.read_text(encoding="utf-8"))
    parent_results = parent_checkpoint.get("results", {})
    if not set(parent_results).issubset(selected_ids):
        raise ValueError("parent results are not a subset of the requested gate")
    checkpoint.data["operations"].update(parent_checkpoint.get("operations", {}))
    checkpoint.data["results"].update(parent_results)
    checkpoint.save()
    self_path = parent_output / "self_domains.json"
    user_path = parent_output / "user_domains.json"
    self_domains = json.loads(self_path.read_text(encoding="utf-8")) if self_path.exists() else {}
    user_domains = json.loads(user_path.read_text(encoding="utf-8")) if user_path.exists() else {}
    return self_domains, user_domains


def _v3_gate_manifests(prepared: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build nested V3 gates with every speaker/session represented by gate 30."""
    by_speaker_session: list[list[list[str]]] = []
    for item in prepared:
        sessions = []
        for session in ("session_1", "session_2", "session_3"):
            sessions.append([
                point["result_id"] for point in item["points"]
                if point["target_session"] == session
            ])
        if any(not values for values in sessions):
            raise ValueError(f"V3 requires targets in all three sessions for {item['speaker']}")
        by_speaker_session.append(sessions)

    def spread(speaker_count: int, per_session: int) -> list[str]:
        return [
            result_id
            for sessions in by_speaker_session[:speaker_count]
            for points in sessions
            for result_id in points[:per_session]
        ]

    gate6 = (
        [by_speaker_session[i][0][0] for i in range(min(4, len(by_speaker_session)))]
        + [by_speaker_session[0][1][0], by_speaker_session[1][2][0]]
    )
    gate18 = spread(6, 1)
    gate30 = spread(len(by_speaker_session), 1)
    gate60 = spread(len(by_speaker_session), 2)
    gate120 = spread(len(by_speaker_session), 4)
    gate519 = [point["result_id"] for item in prepared for point in item["points"]]
    gates = {"6": gate6, "18": gate18, "30": gate30, "60": gate60, "120": gate120, "519": gate519}
    if any(len(values) != int(key) for key, values in gates.items()):
        raise ValueError({key: len(values) for key, values in gates.items()})
    if not all(set(gates[str(left)]) <= set(gates[str(right)]) for left, right in ((6, 18), (18, 30), (30, 60), (60, 120), (120, 519))):
        raise ValueError("V3 gate manifests are not nested")
    return gates


def run(config: Config, backend: Any | None = None) -> dict[str, Any]:
    if config.model != MODEL:
        raise ValueError(f"V3 model is fixed to {MODEL}")
    prepared, dataset_manifest = base.prepare_formal_cb(config.dataset_dir)
    gates = _v3_gate_manifests(prepared)
    if str(config.gate) not in gates:
        raise ValueError(f"V3 gate must be one of {sorted(gates)}")
    selected_ids = list(gates[str(config.gate)])
    if config.selected_ids_file:
        requested = json.loads(Path(config.selected_ids_file).read_text(encoding="utf-8"))
        if not isinstance(requested, list) or not requested or not all(isinstance(rid, str) for rid in requested) or len(requested) != len(set(requested)):
            raise ValueError("selected_ids_file must contain a non-empty list of unique result_id strings")
        available = {point["result_id"] for item in prepared for point in item["points"]}
        missing = [rid for rid in requested if rid not in available]
        if missing:
            raise ValueError(f"selected ids are not present in the canonical dataset: {missing[:3]}")
        selected_ids = list(requested)
    index = {point["result_id"]: (item, point) for item in prepared for point in item["points"]}
    selected_speakers = {index[rid][0]["speaker"] for rid in selected_ids}
    output = Path(config.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    parent_output = Path(config.parent_output).resolve() if config.parent_output else None
    if config.fresh:
        for name in ("checkpoint.json", "raw_responses.jsonl", "predictions.jsonl", "self_domains.json", "user_domains.json", "manifest.json", "unresolved_errors.jsonl", "GENERATION_COMPLETE"):
            (output/name).unlink(missing_ok=True)
    backend = backend or _backend_from_env(MODEL)
    if hasattr(backend, "client"):
        # Do not clone: older SDK wrappers close the shared HTTP transport when
        # the replaced client is garbage-collected.
        backend.client.max_retries = 0
        backend.max_attempts = 3
    if backend.model != MODEL:
        raise ValueError(f"backend model mismatch: {backend.model}")
    manifest = _manifest(dataset_manifest, selected_ids, config.gate, parent_output, config.selected_ids_file)
    signature = _hash({"manifest": manifest, "config": {k:v for k,v in asdict(config).items() if k not in {"output_dir", "fresh", "resume", "gate"}}})
    checkpoint = OperationCheckpoint(output/"checkpoint.json", signature)
    _write_json(output/"manifest.json", {**manifest, "status": "running", "implementation_signature": signature})
    _write_json(output/"selected_ids.json", selected_ids)
    if config.preflight_only:
        _write_json(output/"manifest.json", {**manifest, "status": "preflight_complete", "implementation_signature": signature})
        return {"status": "preflight_complete", "records": 0, "output_dir": str(output)}
    raw_audit = output/"raw_responses.jsonl"
    self_domains: dict[str, dict[str, Any]] = {}
    user_domains: dict[str, dict[str, dict[str, Any]]] = {}
    if parent_output is not None:
        self_domains, user_domains = _seed_parent_checkpoint(checkpoint, parent_output, set(selected_ids))
    for item in prepared:
        speaker = item["speaker"]
        if speaker not in selected_speakers:
            continue
        allowed_self = base.evidence_ids(item["reference_turns"], speaker)
        stats = _stats(item["reference_turns"], speaker)
        self_result = _structured_call(checkpoint=checkpoint, backend=backend, operation_key=f"v3:self:{base._safe_id(speaker)}", system_prompt=SELF_SYSTEM_PROMPT, user_prompt=_domain_prompt(item, stats), schema=SELF_DOMAIN_SCHEMA, normalizer=lambda value, allowed=allowed_self: _normalize_self(value, allowed), max_tokens=4096, max_attempts=config.operation_max_attempts, raw_audit=raw_audit, enable_thinking=False, hard_timeout_seconds=config.timeout_seconds)
        self_domains[speaker] = self_result["data"]
        _write_json(output/"self_domains.json", self_domains)
        by_session = {s:[t for t in item["current_turns"] if t["session_id"]==s] for s in ("session_1", "session_2", "session_3")}
        existing_domains = user_domains.get(speaker, {})
        domain = existing_domains.get("session_1", empty_user_domain())
        user_domains[speaker] = dict(existing_domains) if existing_domains else {"session_1": domain}
        allowed_partner: set[str] = set()
        needed = {index[rid][1]["target_session"] for rid in selected_ids if index[rid][0]["speaker"]==speaker}
        for previous_index, next_session in ((1,"session_2"),(2,"session_3")):
            if not any(int(s.split("_")[1]) >= int(next_session.split("_")[1]) for s in needed):
                continue
            completed = by_session[f"session_{previous_index}"]; allowed_partner |= base.evidence_ids(completed, item["partner"])
            user_result = _structured_call(checkpoint=checkpoint, backend=backend, operation_key=f"v3:user:{base._safe_id(speaker)}:after:{previous_index}", system_prompt=USER_SYSTEM_PROMPT, user_prompt=_user_prompt(speaker, item["partner"], domain, completed, allowed_partner), schema=USER_DOMAIN_SCHEMA, normalizer=lambda value, allowed=set(allowed_partner): _normalize_user_v3(value, allowed), max_tokens=8192, max_attempts=config.operation_max_attempts, raw_audit=raw_audit, enable_thinking=False, hard_timeout_seconds=config.timeout_seconds)
            domain = user_result["data"]; user_domains[speaker][next_session] = domain
            _write_json(output/"user_domains.json", user_domains)
    for rid in selected_ids:
        if rid in checkpoint.data["results"]:
            continue
        item, point = index[rid]; current = point["context_turns"]
        try:
            gate = behavior_calibrator(current, item["speaker"], point["target_session"], item["reference_turns"])
            stats = _stats(item["reference_turns"], item["speaker"])
            user_domain = user_domains[item["speaker"]][point["target_session"]]
            decision = _structured_call(checkpoint=checkpoint, backend=backend, operation_key=f"v3:decision:{rid}", system_prompt=DECISION_SYSTEM_PROMPT, user_prompt=_decision_prompt(item, point, self_domains[item["speaker"]], user_domain, gate, stats), schema=DECISION_SCHEMA, normalizer=lambda value, allowed=base.evidence_ids(current)|base.evidence_ids(item["reference_turns"]): _normalize_decision(value, allowed, gate), max_tokens=2048, max_attempts=config.operation_max_attempts, raw_audit=raw_audit, enable_thinking=False, hard_timeout_seconds=config.timeout_seconds)
            actor = _actor_call(checkpoint, backend, f"v3:actor:{rid}", item["speaker"], _actor_prompt(item, point, self_domains[item["speaker"]], decision["data"], gate), decision["data"], raw_audit, DECODING["actor_attempts"], config.timeout_seconds)
            checkpoint.store_result(rid, {"result_id": rid, "mode": "cb", "speaker": item["speaker"], "partner": item["partner"], "reference_file": item["reference_file"], "current_file": item["current_file"], "target_session": point["target_session"], "target_turn_id": point["target"]["turn_id"], "source_message_ids": point["target"]["dia_ids"], "history_hash": point["history_hash"], "visible_history_turns": len(current), "user_domain": user_domain, "scene_gate": gate, "self_domain": self_domains[item["speaker"]], "decision": decision["data"], "generated_message": actor["data"], "ground_truth": point["target_message"], "decision_audit": decision["audit"], "actor_audit": actor["audit"]})
        except Exception as exc:
            checkpoint.store_excluded_result(rid, {"status": "unresolved", "result_id": rid, "speaker": item["speaker"], "target_session": point["target_session"], "error_type": type(exc).__name__, "error": str(exc)[:500], "updated_at_utc": _now()})
            break
        checkpoint.data["failures"].pop(f"sample:{rid}", None)
        checkpoint.save()
        _write_jsonl(output/"predictions.jsonl", checkpoint.result_values())
    results = sorted(checkpoint.result_values(), key=lambda r:(r["speaker"].casefold(), r["target_session"], r["target_turn_id"]))
    _write_jsonl(output/"predictions.jsonl", results); _write_json(output/"self_domains.json", self_domains); _write_json(output/"user_domains.json", user_domains)
    unresolved = checkpoint.data.get("failures", {})
    _write_jsonl(output/"unresolved_errors.jsonl", [{"key": key, **value} for key, value in unresolved.items()])
    _write_json(output/"manifest.json", {**manifest, "status": "generation_complete" if len(results)==len(selected_ids) and not unresolved else "incomplete", "implementation_signature": signature, "completed_records": len(results), "unresolved_count": len(unresolved)})
    if len(results)==len(selected_ids) and not unresolved:
        (output/"GENERATION_COMPLETE").write_text(_now()+"\n", encoding="utf-8")
    return {"status": "generation_complete" if len(results)==len(selected_ids) and not unresolved else "incomplete", "records": len(results), "expected": len(selected_ids), "unresolved": len(unresolved), "output_dir": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/realtalk_behavior_calibrated_v3_gate6")
    parser.add_argument("--gate", type=int, default=6)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parent-output")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--selected-ids-file")
    args = parser.parse_args()
    result = run(Config(dataset_dir=args.dataset_dir, output_dir=args.output_dir, gate=args.gate, model=args.model, fresh=args.fresh, resume=args.resume, parent_output=args.parent_output, preflight_only=args.preflight_only, selected_ids_file=args.selected_ids_file))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

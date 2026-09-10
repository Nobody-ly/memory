"""Strict contracts for the REALTALK V15 behavior controller."""
from __future__ import annotations

import re
from typing import Any

PARTNER_ACTS = (
    "none",
    "greeting",
    "question",
    "self-disclosure",
    "opinion",
    "praise-or-encouragement",
    "explicit-affect",
    "support-request",
    "closing",
    "other",
)
CONVERSATIONAL_OBLIGATIONS = (
    "open",
    "respond",
    "acknowledge",
    "continue",
    "clarify",
    "close",
    "none",
)
ORIENTATIONS = ("self-led", "balanced", "partner-adaptive")
ADAPTATION_SOURCES = (
    "self-prior",
    "current-partner-turn",
    "visible-cb-pattern",
    "relevant-user-domain",
    "combined",
)
AFFECTED_DIMENSIONS = (
    "turn-composition",
    "content-selection",
    "relationship-register",
    "disclosure-depth",
    "message-scale",
)
TURN_ACTS = (
    "open",
    "answer",
    "acknowledge",
    "self-disclose",
    "explain-stance",
    "clarification-question",
    "follow-up-question",
    "reciprocal-question",
    "topic-shift-statement",
    "close",
)
QUESTION_ACTS = frozenset({
    "clarification-question",
    "follow-up-question",
    "reciprocal-question",
})
QUESTION_SLOT_PREFIXES = (
    "ask ",
    "clarify ",
    "find out ",
    "check whether ",
    "check if ",
)
QUESTION_WORDS = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "is", "are", "was", "were", "can", "could", "would",
    "will", "have", "has", "had",
})
DISCLOSURE_DEPTHS = ("none", "surface", "personal", "vulnerable")
RELATIONSHIP_REGISTERS = (
    "reserved",
    "casual",
    "casual-close",
    "warm",
    "playful",
    "intimate",
    "supportive",
)
LENGTH_BANDS = ("short", "typical", "extended")
TONES = (
    "neutral",
    "casual",
    "warm",
    "playful",
    "serious",
    "supportive",
    "reserved",
)


TURN_UNIT_SCHEMA = {
    "type": "object",
    "properties": {
        "act": {
            "type": "string",
            "enum": list(TURN_ACTS),
            "description": "Question acts have names ending in -question; all other acts are declarative.",
        },
        "content_slot": {
            "type": "string",
            "description": "Semantic instruction, not draft dialogue. For question acts begin with 'ask '.",
        },
        "question_target": {
            "type": "string",
            "description": "Concrete object for a question act; empty string for every non-question act.",
        },
    },
    "required": ["act", "content_slot", "question_target"],
    "additionalProperties": False,
}


DECISION_SCHEMA = {
    "name": "realtalk_ours_v15_cb_posterior_controller_v4",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "situation": {
                "type": "object",
                "properties": {
                    "partner_act": {"type": "string", "enum": list(PARTNER_ACTS)},
                    "conversational_obligation": {
                        "type": "string",
                        "enum": list(CONVERSATIONAL_OBLIGATIONS),
                    },
                    "topic": {"type": "string"},
                    "uncertainty": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": [
                    "partner_act",
                    "conversational_obligation",
                    "topic",
                    "uncertainty",
                ],
                "additionalProperties": False,
            },
            "relevant_user_domain": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "fact_id": {"type": "string"},
                    },
                    "required": ["fact_id"],
                    "additionalProperties": False,
                },
            },
            "alignment": {
                "type": "object",
                "properties": {
                    "orientation": {"type": "string", "enum": list(ORIENTATIONS)},
                    "lambda_trace": {"type": "number", "minimum": 0, "maximum": 1},
                    "adaptation_source": {
                        "type": "string",
                        "enum": list(ADAPTATION_SOURCES),
                    },
                    "affected_dimensions": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(AFFECTED_DIMENSIONS)},
                    },
                    "decision_basis": {"type": "string"},
                },
                "required": [
                    "orientation",
                    "lambda_trace",
                    "adaptation_source",
                    "affected_dimensions",
                    "decision_basis",
                ],
                "additionalProperties": False,
            },
            "turn_plan": {
                "type": "object",
                "properties": {
                    "turn_units": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 6,
                        "items": TURN_UNIT_SCHEMA,
                    },
                    "disclosure_depth": {
                        "type": "string",
                        "enum": list(DISCLOSURE_DEPTHS),
                    },
                    "relationship_register": {
                        "type": "string",
                        "enum": list(RELATIONSHIP_REGISTERS),
                    },
                    "length_band": {"type": "string", "enum": list(LENGTH_BANDS)},
                    "tone": {"type": "string", "enum": list(TONES)},
                },
                "required": [
                    "turn_units",
                    "disclosure_depth",
                    "relationship_register",
                    "length_band",
                    "tone",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["situation", "relevant_user_domain", "alignment", "turn_plan"],
        "additionalProperties": False,
    },
}


def normalize_v15_decision(value: Any) -> dict[str, Any]:
    schema = DECISION_SCHEMA["schema"]
    root = _exact_object(value, schema, "decision")
    situation = _exact_object(root["situation"], schema["properties"]["situation"], "situation")
    alignment = _exact_object(root["alignment"], schema["properties"]["alignment"], "alignment")
    plan = _exact_object(root["turn_plan"], schema["properties"]["turn_plan"], "turn_plan")

    relevant = root["relevant_user_domain"]
    if not isinstance(relevant, list) or len(relevant) > 2:
        raise ValueError("relevant_user_domain must contain at most two facts")
    normalized_relevant = []
    for index, fact in enumerate(relevant):
        item = _exact_object(
            fact,
            schema["properties"]["relevant_user_domain"]["items"],
            f"relevant_user_domain[{index}]",
        )
        normalized_relevant.append({
            "fact_id": _text(item["fact_id"], f"relevant_user_domain[{index}].fact_id"),
        })

    lambda_trace = _number(alignment["lambda_trace"], "alignment.lambda_trace")
    if not 0 <= lambda_trace <= 1:
        raise ValueError("alignment.lambda_trace must be in [0,1]")
    affected = alignment["affected_dimensions"]
    if not isinstance(affected, list) or len(affected) != len(set(affected)):
        raise ValueError("alignment.affected_dimensions must be a unique array")
    normalized_affected = [
        _enum(item, AFFECTED_DIMENSIONS, f"alignment.affected_dimensions[{index}]")
        for index, item in enumerate(affected)
    ]
    orientation = _enum(alignment["orientation"], ORIENTATIONS, "alignment.orientation")
    source = _enum(
        alignment["adaptation_source"], ADAPTATION_SOURCES, "alignment.adaptation_source"
    )
    if lambda_trace == 0 and normalized_affected:
        raise ValueError("zero lambda_trace requires no affected dimensions")
    if lambda_trace > 0 and not normalized_affected:
        raise ValueError("nonzero lambda_trace requires at least one affected dimension")
    if source == "self-prior" and orientation != "self-led":
        raise ValueError("self-prior adaptation source requires self-led orientation")

    units = plan["turn_units"]
    if not isinstance(units, list) or not 1 <= len(units) <= 6:
        raise ValueError("turn_plan.turn_units must contain 1 to 6 units")
    normalized_units = []
    for index, unit in enumerate(units):
        item = _exact_object(unit, TURN_UNIT_SCHEMA, f"turn_plan.turn_units[{index}]")
        act = _enum(item["act"], TURN_ACTS, f"turn_plan.turn_units[{index}].act")
        target = _text(
            item["question_target"],
            f"turn_plan.turn_units[{index}].question_target",
            allow_empty=True,
        )
        if act in QUESTION_ACTS and not target:
            raise ValueError(f"question act {act!r} requires question_target")
        content_slot = _text(
            item["content_slot"], f"turn_plan.turn_units[{index}].content_slot"
        )
        if act in QUESTION_ACTS and not _describes_information_question(content_slot):
            raise ValueError(
                f"question act {act!r} content_slot must describe exactly one information question"
            )
        if act in QUESTION_ACTS and re.search(
            r"\band\s+(?:if|whether|what|where|when|why|how|who|do|does|did|is|are|has|have|can|could|would|will)\b",
            content_slot,
            flags=re.I,
        ):
            raise ValueError(
                f"question act {act!r} content_slot combines multiple information questions"
            )
        if act not in QUESTION_ACTS and _describes_information_question(content_slot):
            raise ValueError(
                f"non-question act {act!r} content_slot must not describe an information question; "
                "remove the optional question, or put one necessary question in its own question-act unit"
            )
        if act not in QUESTION_ACTS and target:
            raise ValueError(f"non-question act {act!r} must have empty question_target")
        normalized_units.append({
            "act": act,
            "content_slot": content_slot,
            "question_target": target,
        })

    partner_act = _enum(situation["partner_act"], PARTNER_ACTS, "situation.partner_act")
    obligation = _enum(
        situation["conversational_obligation"],
        CONVERSATIONAL_OBLIGATIONS,
        "situation.conversational_obligation",
    )
    if partner_act == "praise-or-encouragement" and obligation == "acknowledge":
        if any(unit["act"] != "acknowledge" for unit in normalized_units):
            raise ValueError(
                "closing praise with acknowledge obligation permits only acknowledge turn units"
            )
        if any(_backdates_praise_suggestion(unit["content_slot"]) for unit in normalized_units):
            raise ValueError(
                "closing-praise acknowledgement must not answer an earlier suggestion by inventing "
                "prior consideration or plans"
            )

    return {
        "situation": {
            "partner_act": partner_act,
            "conversational_obligation": obligation,
            "topic": _text(situation["topic"], "situation.topic", allow_empty=True),
            "uncertainty": _enum(
                situation["uncertainty"], ("low", "medium", "high"), "situation.uncertainty"
            ),
        },
        "relevant_user_domain": normalized_relevant,
        "alignment": {
            "orientation": orientation,
            "lambda_trace": round(lambda_trace, 4),
            "adaptation_source": source,
            "affected_dimensions": normalized_affected,
            "decision_basis": _text(alignment["decision_basis"], "alignment.decision_basis"),
        },
        "turn_plan": {
            "turn_units": normalized_units,
            "disclosure_depth": _enum(
                plan["disclosure_depth"], DISCLOSURE_DEPTHS, "turn_plan.disclosure_depth"
            ),
            "relationship_register": _enum(
                plan["relationship_register"],
                RELATIONSHIP_REGISTERS,
                "turn_plan.relationship_register",
            ),
            "length_band": _enum(plan["length_band"], LENGTH_BANDS, "turn_plan.length_band"),
            "tone": _enum(plan["tone"], TONES, "turn_plan.tone"),
        },
    }


def _describes_information_question(value: str) -> bool:
    lowered = value.strip().casefold()
    if lowered.startswith(QUESTION_SLOT_PREFIXES):
        return True
    for fragment in re.findall(r"[^?]*\?", lowered):
        normalized = " ".join(fragment.split())
        if normalized.endswith(("you know?", "right?", "okay?", "ok?", "isn't it?", "aren't they?")):
            continue
        return True
    first = lowered.split(maxsplit=1)[0] if lowered else ""
    return first in QUESTION_WORDS


def _backdates_praise_suggestion(value: str) -> bool:
    return bool(
        re.search(
            r"\b(?:i've|i\s+have)\s+(?:already\s+)?(?:thought\s+about|considered|"
            r"planned|been\s+planning|started|tried)\b",
            value,
            re.I,
        )
        or (
            re.search(r"\b(?:before|for\s+a\s+while|already)\b", value, re.I)
            and re.search(r"\b(?:think|thought|consider|plan|start|try)\b", value, re.I)
        )
    )


def _exact_object(value: Any, schema: dict[str, Any], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    required = set(schema.get("required", []))
    keys = set(value)
    if keys != required:
        raise ValueError(
            f"{path} fields mismatch: missing={sorted(required - keys)} "
            f"extra={sorted(keys - required)}"
        )
    return value


def _text(value: Any, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    text = value.strip()
    if not text and not allow_empty:
        raise ValueError(f"{path} must not be empty")
    return text


def _enum(value: Any, allowed: tuple[str, ...], path: str) -> str:
    text = _text(value, path)
    if text not in allowed:
        raise ValueError(f"{path} must be one of {allowed}, got {text!r}")
    return text


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    return float(value)

"""Strict decision contract for the REALTALK V14 turn-bundle replay."""
from __future__ import annotations

from typing import Any

from .realtalk_ours_schemas import PROFILE_LAYERS


ORIENTATIONS = ("self-led", "balanced", "partner-adaptive")
PARTNER_ACTS = (
    "none",
    "greeting",
    "question",
    "self-disclosure",
    "opinion",
    "explicit-affect",
    "support-request",
    "closing",
    "other",
)
PRIMARY_MOVES = (
    "open",
    "answer",
    "acknowledge",
    "self-disclose",
    "follow-up",
    "topic-shift",
    "close",
)
SUPPORTING_MOVES = (
    "acknowledge",
    "answer",
    "self-disclose",
    "brief-reason",
    "close",
)
QUESTION_PLANS = ("none", "reciprocal", "follow-up", "clarify")
REFLECTION_DEPTHS = ("none", "surface", "brief")
RELATIONSHIP_REGISTERS = (
    "reserved",
    "casual",
    "warm",
    "playful",
    "intimate",
    "supportive",
)
LENGTH_BANDS = ("short", "typical", "extended")


DECISION_SCHEMA = {
    "name": "realtalk_ours_v14_turn_bundle_decision_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "situation": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "latest_partner_act": {
                        "type": "string",
                        "enum": list(PARTNER_ACTS),
                    },
                    "current_obligation": {"type": "string"},
                    "open_question": {"type": "boolean"},
                    "explicit_affect": {"type": "boolean"},
                    "support_request": {"type": "boolean"},
                    "uncertainty": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": [
                    "topic",
                    "latest_partner_act",
                    "current_obligation",
                    "open_question",
                    "explicit_affect",
                    "support_request",
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
                        "layer": {"type": "string", "enum": list(PROFILE_LAYERS)},
                        "value": {"type": "string"},
                    },
                    "required": ["layer", "value"],
                    "additionalProperties": False,
                },
            },
            "alignment": {
                "type": "object",
                "properties": {
                    "orientation": {
                        "type": "string",
                        "enum": list(ORIENTATIONS),
                    },
                    "lambda_trace": {"type": "number", "minimum": 0, "maximum": 1},
                    "decision_basis": {"type": "string"},
                },
                "required": ["orientation", "lambda_trace", "decision_basis"],
                "additionalProperties": False,
            },
            "message_plan": {
                "type": "object",
                "properties": {
                    "primary_move": {
                        "type": "string",
                        "enum": list(PRIMARY_MOVES),
                    },
                    "supporting_moves": {
                        "type": "array",
                        "maxItems": 2,
                        "items": {"type": "string", "enum": list(SUPPORTING_MOVES)},
                    },
                    "bubble_count": {"type": "integer", "minimum": 1, "maximum": 6},
                    "question_plan": {
                        "type": "string",
                        "enum": list(QUESTION_PLANS),
                    },
                    "reflection_depth": {
                        "type": "string",
                        "enum": list(REFLECTION_DEPTHS),
                    },
                    "relationship_register": {
                        "type": "string",
                        "enum": list(RELATIONSHIP_REGISTERS),
                    },
                    "length_band": {
                        "type": "string",
                        "enum": list(LENGTH_BANDS),
                    },
                    "content_direction": {"type": "string"},
                    "tone": {"type": "string"},
                },
                "required": [
                    "primary_move",
                    "supporting_moves",
                    "bubble_count",
                    "question_plan",
                    "reflection_depth",
                    "relationship_register",
                    "length_band",
                    "content_direction",
                    "tone",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "situation",
            "relevant_user_domain",
            "alignment",
            "message_plan",
        ],
        "additionalProperties": False,
    },
}


def normalize_v14_decision(value: Any) -> dict[str, Any]:
    schema = DECISION_SCHEMA["schema"]
    root = _exact_object(value, schema, "decision")
    situation = _exact_object(
        root["situation"], schema["properties"]["situation"], "situation"
    )
    alignment = _exact_object(
        root["alignment"], schema["properties"]["alignment"], "alignment"
    )
    plan = _exact_object(
        root["message_plan"], schema["properties"]["message_plan"], "message_plan"
    )

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
            "layer": _enum(item["layer"], PROFILE_LAYERS, f"relevant_user_domain[{index}].layer"),
            "value": _text(item["value"], f"relevant_user_domain[{index}].value"),
        })

    lambda_trace = _number(alignment["lambda_trace"], "alignment.lambda_trace")
    if not 0 <= lambda_trace <= 1:
        raise ValueError("alignment.lambda_trace must be in [0,1]")

    primary_move = _enum(plan["primary_move"], PRIMARY_MOVES, "message_plan.primary_move")
    supporting = plan["supporting_moves"]
    if not isinstance(supporting, list) or len(supporting) > 2:
        raise ValueError("message_plan.supporting_moves must contain at most two moves")
    supporting_moves = [
        _enum(item, SUPPORTING_MOVES, f"message_plan.supporting_moves[{index}]")
        for index, item in enumerate(supporting)
    ]
    if len(supporting_moves) != len(set(supporting_moves)):
        raise ValueError("message_plan.supporting_moves must not contain duplicates")

    question_plan = _enum(plan["question_plan"], QUESTION_PLANS, "message_plan.question_plan")
    if primary_move == "follow-up" and question_plan not in {"follow-up", "clarify"}:
        raise ValueError("follow-up primary_move requires follow-up or clarify question_plan")
    if question_plan in {"follow-up", "clarify"} and primary_move != "follow-up":
        raise ValueError("follow-up or clarify question_plan requires follow-up primary_move")
    if primary_move == "follow-up" and question_plan == "reciprocal":
        raise ValueError("follow-up primary_move cannot use reciprocal question_plan")

    bubble_count = _integer(plan["bubble_count"], "message_plan.bubble_count")
    if not 1 <= bubble_count <= 6:
        raise ValueError("message_plan.bubble_count must be in [1,6]")

    return {
        "situation": {
            "topic": _text(situation["topic"], "situation.topic", allow_empty=True),
            "latest_partner_act": _enum(
                situation["latest_partner_act"], PARTNER_ACTS, "situation.latest_partner_act"
            ),
            "current_obligation": _text(
                situation["current_obligation"], "situation.current_obligation", allow_empty=True
            ),
            "open_question": _boolean(situation["open_question"], "situation.open_question"),
            "explicit_affect": _boolean(situation["explicit_affect"], "situation.explicit_affect"),
            "support_request": _boolean(situation["support_request"], "situation.support_request"),
            "uncertainty": _enum(
                situation["uncertainty"], ("low", "medium", "high"), "situation.uncertainty"
            ),
        },
        "relevant_user_domain": normalized_relevant,
        "alignment": {
            "orientation": _enum(
                alignment["orientation"], ORIENTATIONS, "alignment.orientation"
            ),
            "lambda_trace": round(lambda_trace, 4),
            "decision_basis": _text(alignment["decision_basis"], "alignment.decision_basis"),
        },
        "message_plan": {
            "primary_move": primary_move,
            "supporting_moves": supporting_moves,
            "bubble_count": bubble_count,
            "question_plan": question_plan,
            "reflection_depth": _enum(
                plan["reflection_depth"], REFLECTION_DEPTHS, "message_plan.reflection_depth"
            ),
            "relationship_register": _enum(
                plan["relationship_register"],
                RELATIONSHIP_REGISTERS,
                "message_plan.relationship_register",
            ),
            "length_band": _enum(plan["length_band"], LENGTH_BANDS, "message_plan.length_band"),
            "content_direction": _text(plan["content_direction"], "message_plan.content_direction"),
            "tone": _text(plan["tone"], "message_plan.tone"),
        },
    }


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


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")
    return value

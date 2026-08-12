"""Strict contracts for the REALTALK Ours Agentic V2 pipeline."""
from __future__ import annotations

from typing import Any, Callable


CONFIDENCE = ("low", "medium", "high")
INTENSITY = ("low", "medium", "high")
PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")
ORIENTATIONS = ("self-led", "balanced", "partner-adaptive")
PRIMARY_MOVES = (
    "open",
    "self-disclose",
    "answer",
    "acknowledge",
    "follow-up",
    "topic-shift",
)


def _string_array_schema() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _fact_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "confidence": {"type": "string", "enum": list(CONFIDENCE)},
            "evidence_ids": _string_array_schema(),
        },
        "required": ["value", "confidence", "evidence_ids"],
        "additionalProperties": False,
    }


SELF_DOMAIN_SCHEMA = {
    "name": "realtalk_ours_agentic_self_domain_v4",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "identity_context": {
                "type": "object",
                "properties": {
                    "self_descriptions": _string_array_schema(),
                    "life_background": _string_array_schema(),
                    "relationships": _string_array_schema(),
                    "recurring_interests": _string_array_schema(),
                },
                "required": [
                    "self_descriptions",
                    "life_background",
                    "relationships",
                    "recurring_interests",
                ],
                "additionalProperties": False,
            },
            "communication_signature": {
                "type": "object",
                "properties": {
                    "tone": _string_array_schema(),
                    "vocabulary_and_phrasing": _string_array_schema(),
                    "information_density": {"type": "string"},
                    "typical_message_scale": {"type": "string"},
                    "expression_patterns": _string_array_schema(),
                },
                "required": [
                    "tone",
                    "vocabulary_and_phrasing",
                    "information_density",
                    "typical_message_scale",
                    "expression_patterns",
                ],
                "additionalProperties": False,
            },
            "interaction_policy_prior": {
                "type": "object",
                "properties": {
                    "initiative": {"type": "string"},
                    "self_disclosure": {"type": "string"},
                    "question_behavior": {"type": "string"},
                    "topic_continuation": {"type": "string"},
                    "topic_shift": {"type": "string"},
                    "advice_behavior": {"type": "string"},
                    "response_to_partner_emotion": {"type": "string"},
                },
                "required": [
                    "initiative",
                    "self_disclosure",
                    "question_behavior",
                    "topic_continuation",
                    "topic_shift",
                    "advice_behavior",
                    "response_to_partner_emotion",
                ],
                "additionalProperties": False,
            },
            "affective_social_signature": {
                "type": "object",
                "properties": {
                    "emotion_expression": {"type": "string"},
                    "sentiment_style": {"type": "string"},
                    "introspection_style": {"type": "string"},
                    "follow_up_style": {"type": "string"},
                    "warmth_style": {"type": "string"},
                    "closeness_style": {"type": "string"},
                },
                "required": [
                    "emotion_expression",
                    "sentiment_style",
                    "introspection_style",
                    "follow_up_style",
                    "warmth_style",
                    "closeness_style",
                ],
                "additionalProperties": False,
            },
            "boundaries_and_uncertainty": {
                "type": "object",
                "properties": {
                    "stable_boundaries": _string_array_schema(),
                    "uncertain_attributes": _string_array_schema(),
                },
                "required": ["stable_boundaries", "uncertain_attributes"],
                "additionalProperties": False,
            },
            "observable_statistics": {
                "type": "object",
                "properties": {
                    "target_message_count": {"type": "integer", "minimum": 1},
                    "mean_characters": {"type": "number", "minimum": 0},
                    "median_characters": {"type": "number", "minimum": 0},
                    "question_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    "first_person_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    "reflective_marker_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    "evaluative_opener_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    "median_merged_bubbles": {"type": "number", "minimum": 1},
                },
                "required": [
                    "target_message_count",
                    "mean_characters",
                    "median_characters",
                    "question_rate",
                    "first_person_rate",
                    "reflective_marker_rate",
                    "evaluative_opener_rate",
                    "median_merged_bubbles",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "identity_context",
            "communication_signature",
            "interaction_policy_prior",
            "affective_social_signature",
            "boundaries_and_uncertainty",
            "observable_statistics",
        ],
        "additionalProperties": False,
    },
}


USER_DOMAIN_SCHEMA = {
    "name": "realtalk_ours_agentic_user_domain_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            **{
                layer: {"type": "array", "items": _fact_schema()}
                for layer in PROFILE_LAYERS
            },
            "update_summary": {
                "type": "object",
                "properties": {
                    "added": _string_array_schema(),
                    "revised": _string_array_schema(),
                    "removed": _string_array_schema(),
                    "uncertainties": _string_array_schema(),
                },
                "required": ["added", "revised", "removed", "uncertainties"],
                "additionalProperties": False,
            },
        },
        "required": [*PROFILE_LAYERS, "update_summary"],
        "additionalProperties": False,
    },
}


ALIGNMENT_SCHEMA = {
    "name": "realtalk_ours_agentic_decision_v4",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "situation": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "partner_move": {"type": "string"},
                    "explicit_affect": {"type": "string"},
                    "affect_intensity": {"type": "string", "enum": list(INTENSITY)},
                    "support_request": {"type": "boolean"},
                    "open_question": {"type": "string"},
                    "partner_has_open_thread": {"type": "boolean"},
                    "missing_information": {"type": "string"},
                    "continuation_value": {
                        "type": "string",
                        "enum": ["none", "low", "medium", "high"],
                    },
                    "uncertainty": {"type": "string", "enum": list(CONFIDENCE)},
                },
                "required": [
                    "topic",
                    "partner_move",
                    "explicit_affect",
                    "affect_intensity",
                    "support_request",
                    "open_question",
                    "partner_has_open_thread",
                    "missing_information",
                    "continuation_value",
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
                    "orientation": {"type": "string", "enum": list(ORIENTATIONS)},
                    "lambda_trace": {"type": "number", "minimum": 0, "maximum": 1},
                    "decision_basis": {"type": "string"},
                },
                "required": ["orientation", "lambda_trace", "decision_basis"],
                "additionalProperties": False,
            },
            "next_action": {
                "type": "object",
                "properties": {
                    "communicative_intent": {"type": "string"},
                    "primary_move": {"type": "string", "enum": list(PRIMARY_MOVES)},
                    "content_direction": {"type": "string"},
                    "self_expression": {"type": "string"},
                    "partner_adaptation": {"type": "string"},
                    "tone": {"type": "string"},
                    "message_scale": {
                        "type": "string",
                        "enum": ["short", "typical", "extended"],
                    },
                    "question_mode": {
                        "type": "string",
                        "enum": ["none", "follow-up"],
                    },
                    "continuation_move": {
                        "type": "string",
                        "enum": ["none", "reciprocal-question"],
                    },
                },
                "required": [
                    "communicative_intent",
                    "primary_move",
                    "content_direction",
                    "self_expression",
                    "partner_adaptation",
                    "tone",
                    "message_scale",
                    "question_mode",
                    "continuation_move",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["situation", "relevant_user_domain", "alignment", "next_action"],
        "additionalProperties": False,
    },
}


def empty_user_domain() -> dict[str, Any]:
    return {
        **{layer: [] for layer in PROFILE_LAYERS},
        "update_summary": {
            "added": [],
            "revised": [],
            "removed": [],
            "uncertainties": ["No completed partner session has been observed yet."],
        },
    }


def normalize_self_domain(value: Any) -> dict[str, Any]:
    root = _exact_object(value, SELF_DOMAIN_SCHEMA["schema"], "self_domain")
    result: dict[str, Any] = {}
    for section in (
        "identity_context",
        "communication_signature",
        "interaction_policy_prior",
        "affective_social_signature",
        "boundaries_and_uncertainty",
    ):
        section_schema = SELF_DOMAIN_SCHEMA["schema"]["properties"][section]
        item = _exact_object(root[section], section_schema, section)
        result[section] = {
            key: (
                _strings(raw, f"{section}.{key}")
                if isinstance(raw, list)
                else _text(raw, f"{section}.{key}")
            )
            for key, raw in item.items()
        }
    stats_schema = SELF_DOMAIN_SCHEMA["schema"]["properties"]["observable_statistics"]
    stats = _exact_object(root["observable_statistics"], stats_schema, "observable_statistics")
    result["observable_statistics"] = {
        "target_message_count": _integer(stats["target_message_count"], "target_message_count", minimum=1),
        **{
            key: _number(stats[key], f"observable_statistics.{key}")
            for key in (
                "mean_characters",
                "median_characters",
                "question_rate",
                "first_person_rate",
                "reflective_marker_rate",
                "evaluative_opener_rate",
                "median_merged_bubbles",
            )
        },
    }
    for key in (
        "question_rate", "first_person_rate", "reflective_marker_rate",
        "evaluative_opener_rate",
    ):
        if not 0 <= result["observable_statistics"][key] <= 1:
            raise ValueError(f"observable_statistics.{key} must be in [0,1]")
    return result


def normalize_user_domain(value: Any) -> dict[str, Any]:
    root = _exact_object(value, USER_DOMAIN_SCHEMA["schema"], "user_domain")
    normalized: dict[str, Any] = {}
    for layer in PROFILE_LAYERS:
        facts = root[layer]
        if not isinstance(facts, list):
            raise ValueError(f"{layer} must be an array")
        seen: set[str] = set()
        normalized[layer] = []
        for index, fact in enumerate(facts):
            item = _exact_object(fact, _fact_schema(), f"{layer}[{index}]")
            fact_value = _text(item["value"], f"{layer}[{index}].value")
            if fact_value.casefold() in seen:
                raise ValueError(f"duplicate fact in {layer}: {fact_value}")
            seen.add(fact_value.casefold())
            normalized[layer].append({
                "value": fact_value,
                "confidence": _enum(item["confidence"], CONFIDENCE, f"{layer}[{index}].confidence"),
                "evidence_ids": _strings(item["evidence_ids"], f"{layer}[{index}].evidence_ids"),
            })
    summary_schema = USER_DOMAIN_SCHEMA["schema"]["properties"]["update_summary"]
    summary = _exact_object(root["update_summary"], summary_schema, "update_summary")
    normalized["update_summary"] = {
        key: _strings(summary[key], f"update_summary.{key}") for key in summary
    }
    return normalized


def normalize_alignment(value: Any) -> dict[str, Any]:
    schema = ALIGNMENT_SCHEMA["schema"]
    root = _exact_object(value, schema, "decision")
    situation_schema = schema["properties"]["situation"]
    situation = _exact_object(root["situation"], situation_schema, "situation")
    alignment_schema = schema["properties"]["alignment"]
    alignment = _exact_object(root["alignment"], alignment_schema, "alignment")
    action_schema = schema["properties"]["next_action"]
    action = _exact_object(root["next_action"], action_schema, "next_action")

    relevant = root["relevant_user_domain"]
    if not isinstance(relevant, list) or len(relevant) > 2:
        raise ValueError("relevant_user_domain must contain at most two facts")
    normalized_relevant = []
    fact_schema = schema["properties"]["relevant_user_domain"]["items"]
    for index, fact in enumerate(relevant):
        item = _exact_object(fact, fact_schema, f"relevant_user_domain[{index}]")
        normalized_relevant.append({
            "layer": _enum(item["layer"], PROFILE_LAYERS, f"relevant_user_domain[{index}].layer"),
            "value": _text(item["value"], f"relevant_user_domain[{index}].value"),
        })

    lambda_trace = _number(alignment["lambda_trace"], "alignment.lambda_trace")
    if not 0 <= lambda_trace <= 1:
        raise ValueError("alignment.lambda_trace must be in [0,1]")
    primary_move = _enum(action["primary_move"], PRIMARY_MOVES, "next_action.primary_move")
    question_mode = _enum(
        action["question_mode"],
        ("none", "follow-up"),
        "next_action.question_mode",
    )
    if question_mode == "follow-up" and primary_move != "follow-up":
        raise ValueError("follow-up question_mode requires follow-up primary_move")
    if primary_move == "follow-up" and question_mode != "follow-up":
        raise ValueError("follow-up primary_move requires follow-up question_mode")
    continuation_move = _enum(
        action["continuation_move"],
        ("none", "reciprocal-question"),
        "next_action.continuation_move",
    )
    if primary_move == "follow-up" and continuation_move != "none":
        raise ValueError("follow-up primary_move cannot add a continuation move")
    return {
        "situation": {
            "topic": _text(situation["topic"], "situation.topic", allow_empty=True),
            "partner_move": _text(situation["partner_move"], "situation.partner_move", allow_empty=True),
            "explicit_affect": _text(situation["explicit_affect"], "situation.explicit_affect", allow_empty=True),
            "affect_intensity": _enum(situation["affect_intensity"], INTENSITY, "situation.affect_intensity"),
            "support_request": _boolean(situation["support_request"], "situation.support_request"),
            "open_question": _text(situation["open_question"], "situation.open_question", allow_empty=True),
            "partner_has_open_thread": _boolean(
                situation["partner_has_open_thread"], "situation.partner_has_open_thread"
            ),
            "missing_information": _text(
                situation["missing_information"], "situation.missing_information", allow_empty=True
            ),
            "continuation_value": _enum(
                situation["continuation_value"],
                ("none", "low", "medium", "high"),
                "situation.continuation_value",
            ),
            "uncertainty": _enum(situation["uncertainty"], CONFIDENCE, "situation.uncertainty"),
        },
        "relevant_user_domain": normalized_relevant,
        "alignment": {
            "orientation": _enum(alignment["orientation"], ORIENTATIONS, "alignment.orientation"),
            "lambda_trace": round(lambda_trace, 4),
            "decision_basis": _text(alignment["decision_basis"], "alignment.decision_basis"),
        },
        "next_action": {
            "communicative_intent": _text(action["communicative_intent"], "next_action.communicative_intent"),
            "primary_move": primary_move,
            "content_direction": _text(action["content_direction"], "next_action.content_direction"),
            "self_expression": _text(action["self_expression"], "next_action.self_expression"),
            "partner_adaptation": _text(action["partner_adaptation"], "next_action.partner_adaptation", allow_empty=True),
            "tone": _text(action["tone"], "next_action.tone"),
            "message_scale": _enum(action["message_scale"], ("short", "typical", "extended"), "next_action.message_scale"),
            "question_mode": question_mode,
            "continuation_move": continuation_move,
        },
    }


def _exact_object(value: Any, schema: dict[str, Any], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    required = set(schema.get("required", []))
    keys = set(value)
    if keys != required:
        raise ValueError(
            f"{path} fields mismatch: missing={sorted(required - keys)} extra={sorted(keys - required)}"
        )
    return value


def _strings(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    result = []
    for index, item in enumerate(value):
        text = _text(item, f"{path}[{index}]")
        if text not in result:
            result.append(text)
    return result


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
    return round(float(value), 6)


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if value < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")
    return value


Normalizer = Callable[[Any], dict[str, Any]]

"""Strict interfaces for the evidence-conditioned REALTALK Ours protocol."""
from __future__ import annotations

from typing import Any


PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")
SELF_SECTION_LIMITS = {
    "self_claims": 12,
    "voice": 6,
    "social_dispositions": 6,
    "uncertainties": 8,
}


def _strings_schema(
    *, min_items: int = 0, max_items: int | None = None,
    item_max_length: int | None = None,
) -> dict[str, Any]:
    item_schema: dict[str, Any] = {"type": "string"}
    if item_max_length is not None:
        item_schema["maxLength"] = item_max_length
    schema: dict[str, Any] = {
        "type": "array", "items": item_schema, "minItems": min_items
    }
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _evidenced_item(properties: dict[str, Any]) -> dict[str, Any]:
    fields = {
        **properties,
        "evidence_ids": _strings_schema(min_items=1, max_items=4),
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


SELF_DOMAIN_SCHEMA = {
    "name": "realtalk_evidence_conditioned_self_domain_v1_3",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "self_claims": {
                "type": "array",
                "maxItems": SELF_SECTION_LIMITS["self_claims"],
                "items": _evidenced_item({
                    "value": {"type": "string", "maxLength": 240},
                    "temporal_scope": {"type": "string", "maxLength": 120},
                }),
            },
            "voice": {
                "type": "array",
                "maxItems": SELF_SECTION_LIMITS["voice"],
                "items": _evidenced_item({
                    "observation": {"type": "string", "maxLength": 240}
                }),
            },
            "social_dispositions": {
                "type": "array",
                "maxItems": SELF_SECTION_LIMITS["social_dispositions"],
                "items": _evidenced_item({
                    "observation": {"type": "string", "maxLength": 240},
                    "observed_context": {"type": "string", "maxLength": 160},
                }),
            },
            "uncertainties": {
                **_strings_schema(
                    max_items=SELF_SECTION_LIMITS["uncertainties"],
                    item_max_length=240,
                )
            },
        },
        "required": [
            "self_claims", "voice", "social_dispositions", "uncertainties"
        ],
        "additionalProperties": False,
    },
}


PROFILE_FACT_SCHEMA = _evidenced_item({"value": {"type": "string"}})


USER_DOMAIN_SCHEMA = {
    "name": "realtalk_evidence_conditioned_user_domain_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            **{
                layer: {"type": "array", "items": PROFILE_FACT_SCHEMA}
                for layer in PROFILE_LAYERS
            },
            "update_summary": {
                "type": "object",
                "properties": {
                    "added": _strings_schema(),
                    "revised": _strings_schema(),
                    "retained": _strings_schema(),
                    "withdrawn": _strings_schema(),
                },
                "required": ["added", "revised", "retained", "withdrawn"],
                "additionalProperties": False,
            },
        },
        "required": [*PROFILE_LAYERS, "update_summary"],
        "additionalProperties": False,
    },
}


DECISION_SCHEMA = {
    "name": "realtalk_evidence_conditioned_decision_v1_5",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "situation": {
                "type": "object",
                "properties": {
                    "partner_act": {
                        "type": "string",
                        "enum": ["question", "statement", "disclosure", "reaction", "closure", "unclear"],
                    },
                    "current_topic": {"type": "string", "maxLength": 240},
                    "conversational_obligation": {
                        "type": "string",
                        "enum": ["answer", "acknowledge", "react", "share", "ask", "close", "none"],
                    },
                    "uncertainty": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["partner_act", "current_topic", "conversational_obligation", "uncertainty"],
                "additionalProperties": False,
            },
            "alignment": {
                "type": "object",
                "properties": {
                    "orientation": {
                        "type": "string",
                        "enum": ["self-led", "balanced", "partner-adaptive"],
                    },
                    "lambda_trace": {"type": "number", "minimum": 0, "maximum": 1},
                    "basis": {"type": "string", "maxLength": 320},
                    "affected_dimensions": {
                        "type": "array",
                        "items": {"type": "string", "enum": [
                            "questioning", "length", "tone", "self_disclosure", "topic", "bubble_count"
                        ]},
                        "minItems": 1, "maxItems": 4,
                    },
                },
                "required": ["orientation", "lambda_trace", "basis", "affected_dimensions"],
                "additionalProperties": False,
            },
            "turn_plan": {
                "type": "object",
                "properties": {
                    "bubble_count": {"type": "integer", "minimum": 1, "maximum": 3},
                    "units": {
                        "type": "array", "minItems": 1, "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "act": {"type": "string", "enum": [
                                    "answer", "acknowledge", "react", "self_disclose", "explain",
                                    "follow_up", "topic_shift", "close"
                                ]},
                                "content_slot": {"type": "string", "maxLength": 240},
                                "question_allowed": {"type": "boolean"},
                                "self_disclosure_allowed": {"type": "boolean"},
                            },
                            "required": ["act", "content_slot", "question_allowed", "self_disclosure_allowed"],
                            "additionalProperties": False,
                        },
                    },
                    "relationship_tone": {"type": "string", "enum": ["casual", "warm", "close", "neutral", "playful", "serious"]},
                    "length_band": {"type": "string", "enum": ["short", "typical", "long"]},
                },
                "required": ["bubble_count", "units", "relationship_tone", "length_band"],
                "additionalProperties": False,
            },
            "evidence_ids": _strings_schema(max_items=8),
        },
        "required": ["situation", "alignment", "turn_plan", "evidence_ids"],
        "additionalProperties": False,
    },
}


def empty_user_domain() -> dict[str, Any]:
    return {
        **{layer: [] for layer in PROFILE_LAYERS},
        "update_summary": {
            "added": [], "revised": [], "retained": [], "withdrawn": []
        },
    }


def normalize_self_domain(value: Any) -> dict[str, Any]:
    root_schema = SELF_DOMAIN_SCHEMA["schema"]
    root = _exact(value, root_schema, "self_domain")
    result: dict[str, Any] = {
        "uncertainties": _strings_for_schema(
            root["uncertainties"], root_schema["properties"]["uncertainties"],
            "uncertainties",
        )
    }
    for section in ("self_claims", "voice", "social_dispositions"):
        if not isinstance(root[section], list):
            raise ValueError(f"{section} must be an array")
        if len(root[section]) > SELF_SECTION_LIMITS[section]:
            raise ValueError(f"{section} exceeds maxItems")
        schema = SELF_DOMAIN_SCHEMA["schema"]["properties"][section]["items"]
        result[section] = []
        for index, raw in enumerate(root[section]):
            item = _exact(raw, schema, f"{section}[{index}]")
            normalized = {}
            for key in schema["required"]:
                path = f"{section}[{index}].{key}"
                property_schema = schema["properties"][key]
                if key == "evidence_ids":
                    normalized[key] = _strings_for_schema(
                        item[key], property_schema, path
                    )
                elif key == "confidence":
                    normalized[key] = _confidence(item[key], path)
                else:
                    normalized[key] = _text_for_schema(
                        item[key], property_schema, path
                    )
            result[section].append(normalized)
    return result


def normalize_user_domain(value: Any) -> dict[str, Any]:
    root = _exact(value, USER_DOMAIN_SCHEMA["schema"], "user_domain")
    result: dict[str, Any] = {}
    for layer in PROFILE_LAYERS:
        if not isinstance(root[layer], list):
            raise ValueError(f"{layer} must be an array")
        result[layer] = []
        seen: set[str] = set()
        for index, raw in enumerate(root[layer]):
            item = _exact(raw, PROFILE_FACT_SCHEMA, f"{layer}[{index}]")
            fact = {
                "value": _text(item["value"], f"{layer}[{index}].value"),
                "evidence_ids": _strings_for_schema(
                    item["evidence_ids"],
                    PROFILE_FACT_SCHEMA["properties"]["evidence_ids"],
                    f"{layer}[{index}].evidence_ids",
                ),
                "confidence": _confidence(
                    item["confidence"], f"{layer}[{index}].confidence"
                ),
            }
            folded = fact["value"].casefold()
            if folded in seen:
                raise ValueError(f"duplicate value in {layer}: {fact['value']!r}")
            seen.add(folded)
            result[layer].append(fact)
    summary_schema = USER_DOMAIN_SCHEMA["schema"]["properties"]["update_summary"]
    summary = _exact(root["update_summary"], summary_schema, "update_summary")
    result["update_summary"] = {
        key: _strings(summary[key], f"update_summary.{key}")
        for key in summary_schema["required"]
    }
    return result


def normalize_decision(value: Any) -> dict[str, Any]:
    schema = DECISION_SCHEMA["schema"]
    root = _exact(value, schema, "decision")
    situation_schema = schema["properties"]["situation"]
    situation = _exact(root["situation"], situation_schema, "situation")
    alignment_schema = schema["properties"]["alignment"]
    alignment = _exact(root["alignment"], alignment_schema, "alignment")
    plan_schema = schema["properties"]["turn_plan"]
    plan = _exact(root["turn_plan"], plan_schema, "turn_plan")
    if not isinstance(plan["units"], list):
        raise ValueError("turn_plan.units must be an array")
    if len(plan["units"]) < 1 or len(plan["units"]) > 3:
        raise ValueError("turn_plan.units must contain 1-3 units")
    if plan["bubble_count"] != len(plan["units"]):
        raise ValueError("turn_plan.bubble_count must equal len(turn_plan.units)")
    units = []
    unit_schema = plan_schema["properties"]["units"]["items"]
    for index, raw in enumerate(plan["units"]):
        unit = _exact(raw, unit_schema, f"turn_plan.units[{index}]")
        if not isinstance(unit["question_allowed"], bool):
            raise ValueError(f"turn_plan.units[{index}].question_allowed must be boolean")
        if not isinstance(unit["self_disclosure_allowed"], bool):
            raise ValueError(f"turn_plan.units[{index}].self_disclosure_allowed must be boolean")
        units.append({
            "act": _enum(unit["act"], unit_schema["properties"]["act"]["enum"], f"turn_plan.units[{index}].act"),
            "content_slot": _text_for_schema(unit["content_slot"], unit_schema["properties"]["content_slot"], f"turn_plan.units[{index}].content_slot"),
            "question_allowed": unit["question_allowed"],
            "self_disclosure_allowed": unit["self_disclosure_allowed"],
        })
    return {
        "situation": {
            "partner_act": _enum(situation["partner_act"], situation_schema["properties"]["partner_act"]["enum"], "situation.partner_act"),
            "current_topic": _text_for_schema(situation["current_topic"], situation_schema["properties"]["current_topic"], "situation.current_topic"),
            "conversational_obligation": _enum(situation["conversational_obligation"], situation_schema["properties"]["conversational_obligation"]["enum"], "situation.conversational_obligation"),
            "uncertainty": _enum(situation["uncertainty"], situation_schema["properties"]["uncertainty"]["enum"], "situation.uncertainty"),
        },
        "alignment": {
            "orientation": _enum(alignment["orientation"], alignment_schema["properties"]["orientation"]["enum"], "alignment.orientation"),
            "lambda_trace": _confidence(
                alignment["lambda_trace"], "alignment.lambda_trace"
            ),
            "basis": _text_for_schema(alignment["basis"], alignment_schema["properties"]["basis"], "alignment.basis"),
            "affected_dimensions": [
                _enum(item, alignment_schema["properties"]["affected_dimensions"]["items"]["enum"], f"alignment.affected_dimensions[{index}]")
                for index, item in enumerate(_strings_for_schema(
                    alignment["affected_dimensions"],
                    alignment_schema["properties"]["affected_dimensions"],
                    "alignment.affected_dimensions",
                ))
            ],
        },
        "turn_plan": {
            "bubble_count": _integer(plan["bubble_count"], "turn_plan.bubble_count", 1, 3),
            "units": units,
            "relationship_tone": _enum(plan["relationship_tone"], plan_schema["properties"]["relationship_tone"]["enum"], "turn_plan.relationship_tone"),
            "length_band": _enum(plan["length_band"], plan_schema["properties"]["length_band"]["enum"], "turn_plan.length_band"),
        },
        "evidence_ids": _strings_for_schema(root["evidence_ids"], schema["properties"]["evidence_ids"], "evidence_ids"),
    }


def validate_evidence_ids(value: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    found: set[str] = set()
    for key, raw in value.items():
        if key in {"self_claims", "voice", "social_dispositions"}:
            for item in raw:
                found.update(item["evidence_ids"])
        elif key in PROFILE_LAYERS:
            for item in raw:
                found.update(item["evidence_ids"])
    invalid = found - allowed
    if invalid:
        raise ValueError(f"invalid evidence IDs: {sorted(invalid)}")
    return value


def validate_policy_evidence(
    decision: dict[str, Any], allowed: set[str]
) -> dict[str, Any]:
    invalid = set(decision["evidence_ids"]) - allowed
    if invalid:
        raise ValueError(f"policy cites invisible evidence IDs: {sorted(invalid)}")
    return decision


def _exact(value: Any, schema: dict[str, Any], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    expected = set(schema.get("required", []))
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{path} fields mismatch: missing={sorted(expected-actual)} "
            f"extra={sorted(actual-expected)}"
        )
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _enum(value: Any, choices: list[str], path: str) -> str:
    text = _text(value, path)
    if text not in choices:
        raise ValueError(f"{path} must be one of {choices}")
    return text


def _integer(value: Any, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{path} must be in [{minimum},{maximum}]")
    return value


def _nullable_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path)


def _strings(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _text(item, f"{path}[{index}]")
        if text not in result:
            result.append(text)
    return result


def _text_for_schema(value: Any, schema: dict[str, Any], path: str) -> str:
    text = _text(value, path)
    maximum = schema.get("maxLength")
    if maximum is not None and len(text) > maximum:
        raise ValueError(f"{path} exceeds maxLength={maximum}")
    return text


def _strings_for_schema(
    value: Any, schema: dict[str, Any], path: str
) -> list[str]:
    result = _strings(value, path)
    minimum = schema.get("minItems", 0)
    maximum = schema.get("maxItems")
    if len(result) < minimum:
        raise ValueError(f"{path} has fewer than minItems={minimum}")
    if maximum is not None and len(result) > maximum:
        raise ValueError(f"{path} exceeds maxItems={maximum}")
    item_schema = schema.get("items", {})
    for index, item in enumerate(result):
        _text_for_schema(item, item_schema, f"{path}[{index}]")
    return result


def _confidence(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f"{path} must be in [0,1]")
    return round(number, 6)

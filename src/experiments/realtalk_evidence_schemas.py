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
    "name": "realtalk_evidence_conditioned_decision_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scene": {
                "type": "object",
                "properties": {
                    "conversation_threads": _strings_schema(),
                    "target_current_context": {"type": ["string", "null"]},
                    "partner_current_state": {"type": ["string", "null"]},
                    "relationship_context": {"type": ["string", "null"]},
                    "uncertainty": {"type": ["string", "null"]},
                },
                "required": [
                    "conversation_threads", "target_current_context",
                    "partner_current_state", "relationship_context", "uncertainty",
                ],
                "additionalProperties": False,
            },
            "alignment": {
                "type": "object",
                "properties": {
                    "self_tendency": {"type": "string"},
                    "partner_expectation": {"type": ["string", "null"]},
                    "lambda_trace": {"type": "number", "minimum": 0, "maximum": 1},
                    "tradeoff": {"type": "string"},
                },
                "required": [
                    "self_tendency", "partner_expectation", "lambda_trace", "tradeoff"
                ],
                "additionalProperties": False,
            },
            "policy": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "evidence_ids": _strings_schema(),
                },
                "required": ["intent", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "required": ["scene", "alignment", "policy"],
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
    scene_schema = schema["properties"]["scene"]
    scene = _exact(root["scene"], scene_schema, "scene")
    alignment_schema = schema["properties"]["alignment"]
    alignment = _exact(root["alignment"], alignment_schema, "alignment")
    policy_schema = schema["properties"]["policy"]
    policy = _exact(root["policy"], policy_schema, "policy")
    return {
        "scene": {
            "conversation_threads": _strings(
                scene["conversation_threads"], "scene.conversation_threads"
            ),
            **{
                key: _nullable_text(scene[key], f"scene.{key}")
                for key in (
                    "target_current_context", "partner_current_state",
                    "relationship_context", "uncertainty",
                )
            },
        },
        "alignment": {
            "self_tendency": _text(
                alignment["self_tendency"], "alignment.self_tendency"
            ),
            "partner_expectation": _nullable_text(
                alignment["partner_expectation"], "alignment.partner_expectation"
            ),
            "lambda_trace": _confidence(
                alignment["lambda_trace"], "alignment.lambda_trace"
            ),
            "tradeoff": _text(alignment["tradeoff"], "alignment.tradeoff"),
        },
        "policy": {
            "intent": _text(policy["intent"], "policy.intent"),
            "evidence_ids": _strings(policy["evidence_ids"], "policy.evidence_ids"),
        },
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
    invalid = set(decision["policy"]["evidence_ids"]) - allowed
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

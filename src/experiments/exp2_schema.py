"""Strict structured-output contracts for Experiment 2."""
from __future__ import annotations

from typing import Any, Dict

from .exp1_schema import EMOTION_LABELS, SENTIMENT_LABELS


FUTURE_STATE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "name": "exp2_future_user_state",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "future_emotion": {
                "type": "string",
                "enum": list(EMOTION_LABELS),
                "description": "Dominant emotion expected in the next user message.",
            },
            "future_sentiment": {
                "type": "string",
                "enum": list(SENTIMENT_LABELS),
                "description": "Overall sentiment expected in the next user message.",
            },
            "future_intimacy": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": (
                    "Expected content-level intimacy: 0 routine greeting, "
                    "logistics, or impersonal facts; about 0.25 friendly "
                    "surface engagement; about 0.5 meaningful personal "
                    "experience or feelings; about 0.75 vulnerable/private "
                    "disclosure or strong trust; 1 only deeply intimate and "
                    "highly vulnerable content."
                ),
            },
            "future_topic": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
                "description": "Concise expected subject of the next user message.",
            },
        },
        "required": [
            "future_emotion",
            "future_sentiment",
            "future_intimacy",
            "future_topic",
        ],
        "additionalProperties": False,
    },
}


FRAMEWORK_STATE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "name": "exp2_framework_state",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "current_state": {
                "type": "object",
                "properties": {
                    "emotion": {"type": "string"},
                    "stress_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "motivation": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "energy": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "main_need": {"type": "string"},
                    "state_summary": {"type": "string"},
                },
                "required": [
                    "emotion",
                    "stress_level",
                    "motivation",
                    "energy",
                    "main_need",
                    "state_summary",
                ],
                "additionalProperties": False,
            },
            "projected_state": {
                "type": "object",
                "properties": {
                    "next_emotion_trend": {"type": "string"},
                    "possible_behavior": {"type": "string"},
                    "risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "recommended_intervention": {"type": "string"},
                },
                "required": [
                    "next_emotion_trend",
                    "possible_behavior",
                    "risk",
                    "recommended_intervention",
                ],
                "additionalProperties": False,
            },
            "activated_persona": {
                "type": "object",
                "properties": {
                    "empathy_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "teasing_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "warmth_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "guidance_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "activated_tone": {"type": "string"},
                },
                "required": [
                    "empathy_level",
                    "teasing_level",
                    "warmth_level",
                    "guidance_level",
                    "activated_tone",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["current_state", "projected_state", "activated_persona"],
        "additionalProperties": False,
    },
}


def normalize_future_state(value: Any) -> Dict[str, Any]:
    canonical = {"emotion", "sentiment", "intimacy", "topic"}
    if isinstance(value, dict) and set(value) == canonical:
        value = {
            "future_emotion": value["emotion"],
            "future_sentiment": value["sentiment"],
            "future_intimacy": value["intimacy"],
            "future_topic": value["topic"],
        }
    required = {
        "future_emotion",
        "future_sentiment",
        "future_intimacy",
        "future_topic",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"future-state result must contain exactly {sorted(required)}"
        )

    emotion = str(value["future_emotion"]).strip().lower()
    sentiment = str(value["future_sentiment"]).strip().lower()
    topic = str(value["future_topic"]).strip()
    if emotion not in EMOTION_LABELS:
        raise ValueError(f"unsupported future emotion label: {emotion}")
    if sentiment not in SENTIMENT_LABELS:
        raise ValueError(f"unsupported future sentiment label: {sentiment}")
    if not topic:
        raise ValueError("future topic must not be empty")

    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "intimacy": _bounded_number(value["future_intimacy"], "future intimacy"),
        "topic": topic,
    }


def normalize_framework_state(value: Any) -> Dict[str, Any]:
    required = {"current_state", "projected_state", "activated_persona"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"framework state must contain exactly {sorted(required)}"
        )
    schema = FRAMEWORK_STATE_RESPONSE_SCHEMA["schema"]["properties"]
    normalized: Dict[str, Any] = {}
    for block in required:
        content = value[block]
        expected = set(schema[block]["required"])
        if not isinstance(content, dict) or set(content) != expected:
            raise ValueError(
                f"{block} must contain exactly {sorted(expected)}"
            )
        normalized[block] = dict(content)
    return normalized


def _bounded_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not 0 <= normalized <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return normalized

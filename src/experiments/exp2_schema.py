"""Strict structured-output contract for Experiment 2 future-state prediction."""
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
            "future_reflective": {
                "type": "boolean",
                "description": "Whether the next user message is expected to be reflective.",
            },
            "future_grounding": {
                "type": "boolean",
                "description": "Whether the next user message is expected to use a grounding act.",
            },
            "future_empathy": {
                "type": "object",
                "properties": {
                    "emotional_reaction": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                    },
                    "interpretation": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                    },
                    "exploration": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                    },
                },
                "required": [
                    "emotional_reaction",
                    "interpretation",
                    "exploration",
                ],
                "additionalProperties": False,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence in the complete future-state prediction.",
            },
        },
        "required": [
            "future_emotion",
            "future_sentiment",
            "future_intimacy",
            "future_topic",
            "future_reflective",
            "future_grounding",
            "future_empathy",
            "confidence",
        ],
        "additionalProperties": False,
    },
}


def normalize_future_state(value: Any) -> Dict[str, Any]:
    canonical = {
        "emotion",
        "sentiment",
        "intimacy",
        "topic",
        "reflective",
        "grounding",
        "empathy",
        "confidence",
    }
    if isinstance(value, dict) and set(value) == canonical:
        value = {
            "future_emotion": value["emotion"],
            "future_sentiment": value["sentiment"],
            "future_intimacy": value["intimacy"],
            "future_topic": value["topic"],
            "future_reflective": value["reflective"],
            "future_grounding": value["grounding"],
            "future_empathy": value["empathy"],
            "confidence": value["confidence"],
        }
    required = {
        "future_emotion",
        "future_sentiment",
        "future_intimacy",
        "future_topic",
        "future_reflective",
        "future_grounding",
        "future_empathy",
        "confidence",
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

    reflective = value["future_reflective"]
    grounding = value["future_grounding"]
    if not isinstance(reflective, bool) or not isinstance(grounding, bool):
        raise ValueError("future reflective and grounding must be booleans")

    intimacy = _bounded_number(value["future_intimacy"], "future intimacy")
    confidence = _bounded_number(value["confidence"], "confidence")
    empathy = value["future_empathy"]
    empathy_fields = {"emotional_reaction", "interpretation", "exploration"}
    if not isinstance(empathy, dict) or set(empathy) != empathy_fields:
        raise ValueError("future empathy must contain all EPITOME components")
    normalized_empathy: Dict[str, int] = {}
    for field in empathy_fields:
        score = empathy[field]
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 2:
            raise ValueError(f"future empathy {field} must be an integer in [0, 2]")
        normalized_empathy[field] = score

    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "intimacy": intimacy,
        "topic": topic,
        "reflective": reflective,
        "grounding": grounding,
        "empathy": normalized_empathy,
        "confidence": confidence,
    }


def _bounded_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not 0 <= normalized <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return normalized

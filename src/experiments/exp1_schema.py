"""Shared structured output contract for Experiment 1."""
from __future__ import annotations

from typing import Any, Dict


EMOTION_LABELS = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "love",
    "optimism",
    "pessimism",
    "sadness",
    "surprise",
    "trust",
)
SENTIMENT_LABELS = ("positive", "negative", "neutral")

STATE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "name": "exp1_current_user_state",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": list(EMOTION_LABELS),
                "description": "Dominant emotion expressed by the current message.",
            },
            "sentiment": {
                "type": "string",
                "enum": list(SENTIMENT_LABELS),
                "description": "Overall sentiment expressed by the current message.",
            },
            "topic": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
                "description": "Concise main subject of the current message.",
            },
            "reflective": {
                "type": "boolean",
                "description": (
                    "Whether the speaker explicitly self-observes, takes "
                    "perspective, or explains motives behind their own state."
                ),
            },
            "grounding": {
                "type": "boolean",
                "description": (
                    "Whether the message builds mutual understanding through "
                    "clarification, confirmation, follow-up, or shared detail."
                ),
            },
            "intimacy": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Interpersonal intimacy expressed, from 0 to 1.",
            },
            "empathy": {
                "type": "object",
                "description": "EPITOME empathy component scores.",
                "properties": {
                    "emotional_reaction": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                        "description": "Emotional reaction, 0 absent to 2 explicit.",
                    },
                    "interpretation": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                        "description": "Interpretation, 0 absent to 2 explicit.",
                    },
                    "exploration": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                        "description": "Exploration, 0 absent to 2 explicit.",
                    },
                },
                "required": [
                    "emotional_reaction", "interpretation", "exploration"
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "emotion",
            "sentiment",
            "topic",
            "reflective",
            "grounding",
            "intimacy",
            "empathy",
        ],
        "additionalProperties": False,
    },
}

REFERENCE_JUDGMENT_SCHEMA: Dict[str, Any] = {
    "name": "exp1_realtalk_reference_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": list(EMOTION_LABELS),
                "description": "Dominant emotion expressed by the observed message.",
            },
            "sentiment": {
                "type": "string",
                "enum": list(SENTIMENT_LABELS),
                "description": "Overall sentiment expressed by the observed message.",
            },
            "topic": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
                "description": "Concise main subject of the observed message.",
            },
            "reflective": {
                "type": "boolean",
                "description": (
                    "Whether the speaker explicitly self-observes, takes "
                    "perspective, or explains motives behind their own state."
                ),
            },
            "grounding": {
                "type": "boolean",
                "description": (
                    "Whether the message builds mutual understanding through "
                    "clarification, confirmation, follow-up, or shared detail."
                ),
            },
            "intimacy": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Interpersonal intimacy expressed, from 0 to 1.",
            },
            "empathy": {
                "type": "object",
                "description": "EPITOME empathy component scores.",
                "properties": {
                    "emotional_reaction": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                        "description": "Emotional reaction, 0 absent to 2 explicit.",
                    },
                    "interpretation": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                        "description": "Interpretation, 0 absent to 2 explicit.",
                    },
                    "exploration": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                        "description": "Exploration, 0 absent to 2 explicit.",
                    },
                },
                "required": [
                    "emotional_reaction", "interpretation", "exploration"
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "emotion",
            "sentiment",
            "topic",
            "reflective",
            "grounding",
            "intimacy",
            "empathy",
        ],
        "additionalProperties": False,
    },
}


def normalize_state(value: Any) -> Dict[str, Any]:
    """Validate and normalize one schema-conforming state result."""
    required = {
        "emotion",
        "sentiment",
        "topic",
        "reflective",
        "grounding",
        "intimacy",
        "empathy",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"state result must contain exactly {sorted(required)}")
    emotion = str(value["emotion"]).strip().lower()
    sentiment = str(value["sentiment"]).strip().lower()
    topic = str(value["topic"]).strip()
    if emotion not in EMOTION_LABELS:
        raise ValueError(f"unsupported emotion label: {emotion}")
    if sentiment not in SENTIMENT_LABELS:
        raise ValueError(f"unsupported sentiment label: {sentiment}")
    if not topic:
        raise ValueError("topic must not be empty")
    reflective = value["reflective"]
    grounding = value["grounding"]
    if not isinstance(reflective, bool) or not isinstance(grounding, bool):
        raise ValueError("reflective and grounding must be booleans")
    intimacy = value["intimacy"]
    if isinstance(intimacy, bool) or not isinstance(intimacy, (int, float)):
        raise ValueError("intimacy must be numeric")
    intimacy = float(intimacy)
    if not 0 <= intimacy <= 1:
        raise ValueError("intimacy must be in [0, 1]")
    empathy = value["empathy"]
    empathy_fields = {"emotional_reaction", "interpretation", "exploration"}
    if not isinstance(empathy, dict) or set(empathy) != empathy_fields:
        raise ValueError("empathy must contain the three EPITOME components")
    normalized_empathy: Dict[str, int] = {}
    for field in empathy_fields:
        score = empathy[field]
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 2:
            raise ValueError(f"empathy {field} must be an integer in [0, 2]")
        normalized_empathy[field] = score
    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "topic": topic,
        "reflective": reflective,
        "grounding": grounding,
        "intimacy": intimacy,
        "empathy": normalized_empathy,
    }


def normalize_reference_judgment(value: Any) -> Dict[str, Any]:
    """Validate a complete Kimi-generated reference state."""
    return normalize_state(value)

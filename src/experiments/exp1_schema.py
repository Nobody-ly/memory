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
            "emotion": {"type": "string", "enum": list(EMOTION_LABELS)},
            "sentiment": {"type": "string", "enum": list(SENTIMENT_LABELS)},
            "topic": {"type": "string", "minLength": 1},
        },
        "required": ["emotion", "sentiment", "topic"],
        "additionalProperties": False,
    },
}


def normalize_state(value: Any) -> Dict[str, str]:
    """Validate and normalize one schema-conforming state result."""
    if not isinstance(value, dict) or set(value) != {"emotion", "sentiment", "topic"}:
        raise ValueError("state result must contain exactly emotion, sentiment, and topic")
    emotion = str(value["emotion"]).strip().lower()
    sentiment = str(value["sentiment"]).strip().lower()
    topic = str(value["topic"]).strip()
    if emotion not in EMOTION_LABELS:
        raise ValueError(f"unsupported emotion label: {emotion}")
    if sentiment not in SENTIMENT_LABELS:
        raise ValueError(f"unsupported sentiment label: {sentiment}")
    if not topic:
        raise ValueError("topic must not be empty")
    return {"emotion": emotion, "sentiment": sentiment, "topic": topic}

"""Future-state prediction module for Experiment 2 / RQ2."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .experiments.exp2_schema import (
    FUTURE_STATE_RESPONSE_SCHEMA,
    normalize_future_state,
)
from .llm_client import LLMClient


FUTURE_STATE_MAX_TOKENS = 4096


# The core task wording is retained from the original Experiment 2.
PREDICTION_SYSTEM_PROMPT = """You predict the user's next conversational state.

Given the dialogue (and optionally a user profile), infer the most likely state in the user's next turn.

Requirements:
- Predict exactly one emotion.
"""

_CONTEXT_BLOCKS = {
    "history": """CONVERSATION HISTORY ({n_turns} observed turns):
{conversation_history}

""",
    "profile": """USER BACKGROUND:
{user_profile}

""",
    "state": """CURRENT STATE DERIVED FROM OBSERVED HISTORY:
{current_state}

""",
}

_OUTPUT_INSTRUCTION = """USER'S LATEST OBSERVED MESSAGE:
"{user_message}"

Predict the user's state in their next message."""

_MODE_BLOCKS = {
    "llm_only": [],
    "dialogue_history": ["history"],
    "user_profile": ["history", "profile"],
    "full_framework": ["history", "profile", "state"],
}
_VALID_MODES = set(_MODE_BLOCKS)


def build_user_prompt(
    mode: str,
    user_message: str,
    conversation_history: str = "",
    n_turns: int = 0,
    user_profile: str = "",
    current_state: str = "",
) -> str:
    """Assemble the original four conditions without mode-specific hints."""
    parts = []
    for block_name in _MODE_BLOCKS[mode]:
        if block_name == "history" and not conversation_history:
            continue
        if block_name == "state" and not current_state:
            continue
        template = _CONTEXT_BLOCKS[block_name]
        parts.append(template.format(
            conversation_history=conversation_history,
            n_turns=n_turns,
            user_profile=user_profile,
            current_state=current_state,
        ))
    parts.append(_OUTPUT_INSTRUCTION.format(user_message=user_message))
    return "".join(parts)


class FutureStatePredictor:
    """Predict the next user state from caller-selected causal observations."""

    def __init__(self, llm: LLMClient, mode: str = "full_framework"):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown prediction mode: {mode}. Must be one of {_VALID_MODES}"
            )
        self.llm = llm
        self.mode = mode

    def predict(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        history = conversation_history or []
        history_text = "\n".join(
            f"{turn['speaker']}: {turn['content']}" for turn in history
        )
        profile_text = (
            json.dumps(user_profile, ensure_ascii=False, indent=2)
            if user_profile else ""
        )
        state_text = (
            json.dumps(current_state, ensure_ascii=False, indent=2)
            if current_state else ""
        )
        user_prompt = build_user_prompt(
            mode=self.mode,
            user_message=user_message,
            conversation_history=history_text,
            n_turns=len(history),
            user_profile=profile_text,
            current_state=state_text,
        )
        raw = self.llm.chat(
            PREDICTION_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
            max_tokens=FUTURE_STATE_MAX_TOKENS,
            response_schema=FUTURE_STATE_RESPONSE_SCHEMA,
        )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("strict future-state response was not valid JSON") from exc
        return normalize_future_state(result)


def compute_prediction_error(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, float]:
    """Compute REALTALK-style per-attribute prediction outcomes."""
    pred_words = set(str(prediction.get("topic", "")).lower().split())
    gt_words = set(str(ground_truth.get("topic", "")).lower().split())
    topic_overlap = (
        len(pred_words & gt_words) / len(gt_words)
        if pred_words and gt_words else 0.0
    )
    return {
        "emotion_accuracy": float(
            prediction.get("emotion") == ground_truth.get("emotion")
        ),
        "sentiment_accuracy": float(
            prediction.get("sentiment") == ground_truth.get("sentiment")
        ),
        "intimacy_absolute_difference": round(
            abs(
                float(prediction.get("intimacy", 0.0))
                - float(ground_truth.get("intimacy", 0.0))
            ),
            4,
        ),
        "topic_consistency": round(topic_overlap, 4),
        "reflectiveness_accuracy": float(
            prediction.get("reflective") == ground_truth.get("reflective")
        ),
        "grounding_accuracy": float(
            prediction.get("grounding") == ground_truth.get("grounding")
        ),
        "empathy_absolute_difference": float(
            abs(
                _empathy_total(prediction.get("empathy", {}))
                - _empathy_total(ground_truth.get("empathy", {}))
            )
        ),
    }


def _empathy_total(value: Dict[str, Any]) -> int:
    return sum(
        int(value.get(field, 0))
        for field in ("emotional_reaction", "interpretation", "exploration")
    )

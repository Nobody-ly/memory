"""Pinned message-level emotion and sentiment labels used by REALTALK."""
from __future__ import annotations

from typing import Any, Callable, Dict

from .exp1_schema import EMOTION_LABELS, SENTIMENT_LABELS


EMOTION_MODEL = "cardiffnlp/twitter-roberta-large-emotion-latest"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_REVISION = "1651a1393c6afeff010a924f1cbcc5985d0a86a7"
SENTIMENT_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"


class RealTalkLabelEvaluator:
    def __init__(self, pipeline_factory: Callable[..., Any] | None = None) -> None:
        self.pipeline_factory = pipeline_factory
        self._emotion_pipeline: Any = None
        self._sentiment_pipeline: Any = None

    def annotate(self, text: str) -> Dict[str, str]:
        self._ensure_loaded()
        emotion = _top_label(self._emotion_pipeline(text))
        sentiment = _top_label(self._sentiment_pipeline(text))
        if emotion not in EMOTION_LABELS:
            raise ValueError(f"unexpected REALTALK emotion label: {emotion}")
        if sentiment not in SENTIMENT_LABELS:
            raise ValueError(f"unexpected REALTALK sentiment label: {sentiment}")
        return {"emotion": emotion, "sentiment": sentiment}

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "pinned_huggingface_pipeline",
            "emotion_model": EMOTION_MODEL,
            "emotion_revision": EMOTION_REVISION,
            "sentiment_model": SENTIMENT_MODEL,
            "sentiment_revision": SENTIMENT_REVISION,
            "emotion_labels": list(EMOTION_LABELS),
            "sentiment_labels": list(SENTIMENT_LABELS),
            "top1_pipeline_semantics": True,
        }

    def _ensure_loaded(self) -> None:
        if self._emotion_pipeline is not None:
            return
        factory = self.pipeline_factory
        if factory is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "REALTALK evaluation dependencies are missing; run "
                    "`uv sync --extra realtalk-eval`"
                ) from exc
            factory = pipeline
        self._emotion_pipeline = factory(
            "text-classification",
            model=EMOTION_MODEL,
            tokenizer=EMOTION_MODEL,
            revision=EMOTION_REVISION,
        )
        self._sentiment_pipeline = factory(
            "text-classification",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            revision=SENTIMENT_REVISION,
        )


def _top_label(prediction: Any) -> str:
    if not isinstance(prediction, list) or not prediction:
        raise ValueError("unexpected Hugging Face pipeline output")
    first = prediction[0]
    if isinstance(first, list):
        if not first:
            raise ValueError("empty Hugging Face pipeline output")
        first = max(first, key=lambda item: float(item.get("score", 0)))
    if not isinstance(first, dict) or "label" not in first:
        raise ValueError("Hugging Face prediction has no label")
    return str(first["label"]).strip().lower()

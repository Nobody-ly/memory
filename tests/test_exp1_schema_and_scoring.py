import unittest

from src.experiments.exp1_schema import EMOTION_LABELS, normalize_state
from src.experiments.exp1_user_understanding import score_prediction


class Exp1SchemaAndScoringTests(unittest.TestCase):
    def test_schema_uses_realtalk_emotion_labels(self):
        self.assertEqual(
            set(EMOTION_LABELS),
            {
                "anger", "anticipation", "disgust", "fear", "joy", "love",
                "optimism", "pessimism", "sadness", "surprise", "trust",
            },
        )

    def test_schema_rejects_extra_keys_and_neutral_emotion(self):
        with self.assertRaises(ValueError):
            normalize_state({
                "emotion": "neutral", "sentiment": "neutral", "topic": "work"
            })
        with self.assertRaises(ValueError):
            normalize_state({
                "emotion": "joy", "sentiment": "positive", "topic": "work",
                "confidence": 0.9,
            })

    def test_scoring_uses_exact_labels(self):
        prediction = {"emotion": "optimism", "sentiment": "positive", "topic": "new job"}
        reference = {"emotion": "joy", "sentiment": "positive", "topic": "job"}
        scores = score_prediction(prediction, reference)
        self.assertEqual(scores["emotion_accuracy"], 0.0)
        self.assertEqual(scores["sentiment_accuracy"], 1.0)
        self.assertEqual(scores["topic_consistency"], 1.0)

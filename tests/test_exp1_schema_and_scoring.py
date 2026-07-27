import unittest

from src.experiments.exp1_schema import EMOTION_LABELS, normalize_state
from src.experiments.exp1_user_understanding import (
    REFERENCE_JUDGE_SYSTEM_PROMPT,
    score_prediction,
)


def _state(**overrides):
    value = {
        "emotion": "joy",
        "sentiment": "positive",
        "topic": "new job",
        "reflective": True,
        "grounding": False,
        "intimacy": 0.7,
        "empathy": {
            "emotional_reaction": 2,
            "interpretation": 1,
            "exploration": 1,
        },
    }
    value.update(overrides)
    return value


class Exp1SchemaAndScoringTests(unittest.TestCase):
    def test_reference_judge_has_content_level_intimacy_anchors(self):
        self.assertIn("routine greetings", REFERENCE_JUDGE_SYSTEM_PROMPT)
        self.assertIn("deeply intimate", REFERENCE_JUDGE_SYSTEM_PROMPT)
        self.assertIn("do not infer intimacy", REFERENCE_JUDGE_SYSTEM_PROMPT)

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
            normalize_state(_state(emotion="neutral"))
        with self.assertRaises(ValueError):
            normalize_state({**_state(), "confidence": 0.9})

    def test_scoring_uses_exact_labels_and_realtalk_differences(self):
        prediction = _state(
            emotion="optimism",
            intimacy=0.2,
            empathy={
                "emotional_reaction": 0,
                "interpretation": 1,
                "exploration": 0,
            },
        )
        reference = _state(topic="job", intimacy=0.7)
        scores = score_prediction(prediction, reference)
        self.assertEqual(scores["emotion_accuracy"], 0.0)
        self.assertEqual(scores["sentiment_accuracy"], 1.0)
        self.assertEqual(scores["topic_consistency"], 1.0)
        self.assertEqual(scores["reflectiveness_accuracy"], 1.0)
        self.assertEqual(scores["grounding_accuracy"], 1.0)
        self.assertEqual(scores["intimacy_absolute_difference"], 0.5)
        self.assertEqual(scores["empathy_absolute_difference"], 3)


if __name__ == "__main__":
    unittest.main()

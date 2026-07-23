import unittest

from src.experiments.exp1_metrics import (
    classification_report,
    paired_correctness_counts,
)
from src.experiments.exp1_schema import EMOTION_LABELS
from src.experiments.exp1_user_understanding import (
    aggregate_results,
    score_prediction,
)


def _state(emotion, sentiment, topic, **overrides):
    value = {
        "emotion": emotion,
        "sentiment": sentiment,
        "topic": topic,
        "reflective": True,
        "grounding": False,
        "intimacy": 0.5,
        "empathy": {
            "emotional_reaction": 1,
            "interpretation": 1,
            "exploration": 0,
        },
    }
    value.update(overrides)
    return value


def _method(prediction, reference):
    return {
        "prediction": prediction,
        "scores": score_prediction(prediction, reference),
    }


def _result(
    result_id,
    speaker,
    reference,
    self_prediction,
    flat_prediction,
    explicit_prediction,
):
    return {
        "result_id": result_id,
        "speaker": speaker,
        "train_chat_file": f"Train_{speaker}.json",
        "test_chat_file": f"Test_{speaker}.json",
        "chat_file": f"Test_{speaker}.json",
        "eval_id": result_id,
        "message_level_index": 0,
        "target_session": "session_1",
        "user_speaker": speaker,
        "target_dia_ids": [result_id],
        "reference": reference,
        "methods": {
            "self_model": _method(self_prediction, reference),
            "flat_profile": _method(flat_prediction, reference),
            "explicit_model": _method(explicit_prediction, reference),
        },
        "profile": {"explicit_portrait_entropy": 0.2},
    }


class Exp1MetricsTests(unittest.TestCase):
    def test_classification_report_keeps_fixed_confusion_matrix_and_support(self):
        report = classification_report(
            ["joy", "joy", "sadness", "sadness"],
            ["joy", "sadness", "sadness", "sadness"],
            EMOTION_LABELS,
        )

        self.assertEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["macro_f1"], 0.733333, places=6)
        self.assertEqual(report["per_class"]["joy"]["support"], 2)
        self.assertEqual(report["per_class"]["sadness"]["predicted"], 3)
        self.assertEqual(
            len(report["confusion_matrix"]["matrix"]), len(EMOTION_LABELS)
        )

    def test_aggregate_demotes_topic_and_preserves_paper_aligned_metrics(self):
        first_reference = _state("joy", "positive", "work")
        second_reference = _state(
            "sadness", "negative", "family", reflective=False, grounding=True
        )
        results = [
            _result(
                "sample-1",
                "Emi",
                first_reference,
                _state("sadness", "negative", "job"),
                _state("joy", "positive", "job"),
                _state("joy", "positive", "job"),
            ),
            _result(
                "sample-2",
                "Nicolas",
                second_reference,
                _state(
                    "sadness", "negative", "home",
                    reflective=False, grounding=True,
                ),
                _state("joy", "positive", "home"),
                _state(
                    "sadness", "neutral", "home",
                    reflective=False, grounding=True, intimacy=0.3,
                ),
            ),
        ]

        summary = aggregate_results(results)
        explicit = summary["comparison"]["explicit_model"]

        self.assertNotIn(
            "topic_consistency", summary["metric_protocol"]["primary_metrics"]
        )
        self.assertIn(
            "topic_consistency", summary["metric_protocol"]["extended_metrics"]
        )
        self.assertNotIn(
            "explicit_vs_self_model_topic_consistency",
            summary["improvement_speaker_macro"],
        )
        self.assertIn(
            "explicit_vs_self_model_topic_consistency",
            summary["extended_improvement_speaker_macro"],
        )
        self.assertEqual(explicit["micro"]["emotion_accuracy"], 1.0)
        self.assertEqual(explicit["speaker_macro"]["sentiment_macro_f1"], 0.5)
        self.assertEqual(explicit["micro"]["reflectiveness_accuracy"], 1.0)
        self.assertEqual(explicit["micro"]["grounding_accuracy"], 1.0)
        self.assertEqual(explicit["micro"]["intimacy_absolute_difference"], 0.1)
        self.assertIn(
            "confusion_matrix",
            explicit["classification_details"]["emotion"]["global"],
        )

        paired = paired_correctness_counts(
            results, "explicit_model", "self_model", "emotion"
        )
        self.assertEqual(paired["candidate_only_correct"], 1)
        self.assertEqual(paired["baseline_only_correct"], 0)


if __name__ == "__main__":
    unittest.main()

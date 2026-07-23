import unittest

from src.experiments.exp1_metrics import (
    classification_report,
    paired_correctness_counts,
)
from src.experiments.exp1_schema import EMOTION_LABELS
from src.experiments.exp1_user_understanding import aggregate_results


def _method(prediction, reference):
    return {
        "prediction": prediction,
        "scores": {
            "emotion_accuracy": float(
                prediction["emotion"] == reference["emotion"]
            ),
            "sentiment_accuracy": float(
                prediction["sentiment"] == reference["sentiment"]
            ),
            "topic_consistency": 0.25,
        },
    }


def _result(
    result_id,
    chat_file,
    reference,
    self_prediction,
    flat_prediction,
    explicit_prediction,
):
    return {
        "result_id": result_id,
        "chat_file": chat_file,
        "eval_id": result_id,
        "boundary_index": 2,
        "target_session": "session_3",
        "user_speaker": "User",
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

    def test_aggregate_demotes_topic_and_preserves_recomputable_details(self):
        first_reference = {
            "emotion": "joy", "sentiment": "positive", "topic": "work"
        }
        second_reference = {
            "emotion": "sadness", "sentiment": "negative", "topic": "family"
        }
        results = [
            _result(
                "sample-1",
                "Chat_1.json",
                first_reference,
                {"emotion": "sadness", "sentiment": "negative", "topic": "job"},
                {"emotion": "joy", "sentiment": "positive", "topic": "job"},
                {"emotion": "joy", "sentiment": "positive", "topic": "job"},
            ),
            _result(
                "sample-2",
                "Chat_2.json",
                second_reference,
                {"emotion": "sadness", "sentiment": "negative", "topic": "home"},
                {"emotion": "joy", "sentiment": "positive", "topic": "home"},
                {"emotion": "sadness", "sentiment": "neutral", "topic": "home"},
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
            summary["improvement_chat_macro"],
        )
        self.assertIn(
            "explicit_vs_self_model_topic_consistency",
            summary["extended_improvement_chat_macro"],
        )
        self.assertEqual(explicit["micro"]["emotion_accuracy"], 1.0)
        self.assertEqual(explicit["chat_macro"]["sentiment_macro_f1"], 0.5)
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

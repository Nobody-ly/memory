import unittest

from src.experiments.realtalk_evaluator import RealTalkLabelEvaluator


class RealTalkEvaluatorTests(unittest.TestCase):
    def test_uses_pinned_top1_labels(self):
        def factory(_task, model, tokenizer, revision):
            self.assertEqual(model, tokenizer)
            self.assertTrue(revision)
            label = "Joy" if "emotion" in model else "Positive"
            return lambda _text: [{"label": label, "score": 0.9}]

        evaluator = RealTalkLabelEvaluator(factory)
        self.assertEqual(
            evaluator.annotate("A lovely day"),
            {"emotion": "joy", "sentiment": "positive"},
        )

    def test_rejects_out_of_taxonomy_emotion(self):
        def factory(_task, model, tokenizer, revision):
            return lambda _text: [{"label": "neutral", "score": 1.0}]

        with self.assertRaisesRegex(ValueError, "emotion label"):
            RealTalkLabelEvaluator(factory).annotate("plain")


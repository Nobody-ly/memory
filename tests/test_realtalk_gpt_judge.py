import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.realtalk_gpt_judge import (
    EMPATHY_PROMPT,
    REFLECTIVENESS_PROMPT,
    GROUNDING_PROMPT,
    _contexts,
    _parse_bool,
    _parse_empathy,
    run,
)


class RealTalkGptJudgeTest(unittest.TestCase):
    def test_prompts_include_appendix_c_examples(self):
        self.assertIn("I did what I thought was best", REFLECTIVENESS_PROMPT)
        self.assertIn("Can you tell me more", GROUNDING_PROMPT)
        self.assertIn("underlying motivations or goals", REFLECTIVENESS_PROMPT)
        self.assertIn("preventing misunderstandings", GROUNDING_PROMPT)
        self.assertIn("potentially using multiple sub-categories", EMPATHY_PROMPT)

    def test_context_is_limited_to_target_session(self):
        rows = [{"speaker": "Akib", "result_id": "akib:message_16:session_2:turn_1"}]
        context = _contexts(Path("dataset"), rows)[rows[0]["result_id"]]
        self.assertNotIn("Good morning. How's it going?", context)

    def test_context_supports_multiword_speaker_ids(self):
        result_id = "fahim_khan:message_0:session_1:turn_1"
        contexts = _contexts(
            Path("dataset"),
            [{"speaker": "Fahim Khan", "result_id": result_id}],
        )
        self.assertIn(result_id, contexts)

    def test_boolean_parser(self):
        self.assertTrue(_parse_bool("True"))
        self.assertFalse(_parse_bool("'False'."))
        self.assertTrue(_parse_bool("Grounding: True"))

    def test_empathy_parser(self):
        self.assertEqual(
            _parse_empathy('{"emotional_reaction":1,"interpretation":2,"exploration":0}'),
            {"emotional_reaction": 1, "interpretation": 2, "exploration": 0},
        )

    def test_empathy_parser_rejects_range(self):
        with self.assertRaises(ValueError):
            _parse_empathy('{"emotional_reaction":3,"interpretation":0,"exploration":0}')

    def test_reference_checkpoint_reuses_three_labels(self):
        result_id = "akib:message_16:session_2:turn_1"
        source = {
            "judgments": {
                f"{result_id}:reference:reflectiveness": {"value": False, "audit": {}},
                f"{result_id}:reference:grounding": {"value": True, "audit": {}},
                f"{result_id}:reference:empathy": {
                    "value": {
                        "emotional_reaction": 0,
                        "interpretation": 1,
                        "exploration": 0,
                    },
                    "audit": {},
                },
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({
                "result_id": result_id,
                "speaker": "Akib",
                "ground_truth": "Reference",
                "generated_message": "Candidate",
            }) + "\n", encoding="utf-8")
            source_path = root / "source_checkpoint.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            responses = iter((
                ("False", {}),
                ("True", {}),
                ('{"emotional_reaction":0,"interpretation":1,"exploration":0}', {}),
            ))
            with patch.dict(os.environ, {
                "REALTALK_JUDGE_API_KEY": "test",
                "REALTALK_JUDGE_BASE_URL": "https://example.invalid/v1",
            }), patch(
                "src.experiments.realtalk_gpt_judge._chat",
                side_effect=lambda *_args: next(responses),
            ) as chat:
                summary = run(
                    predictions,
                    Path("dataset"),
                    root / "judge",
                    "gpt-4o-mini",
                    reference_checkpoint=source_path,
                )
            self.assertEqual(chat.call_count, 3)
            self.assertEqual(summary["reference_judgments_reused"], 3)
            self.assertEqual(summary["judgments_complete"], 6)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.exp1_user_understanding import Exp1Config, METHODS, run_exp1
from src.experiments.operation_checkpoint import CheckpointSignatureError


class FakeLLM:
    model = "fake-structured-model"
    enable_thinking = False
    max_retries = 6

    def __init__(self):
        self.calls = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def chat(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt, kwargs))
        self.token_usage["prompt_tokens"] += 10
        self.token_usage["completion_tokens"] += 2
        self.token_usage["calls"] += 1
        schema = kwargs.get("response_schema")
        if schema is not None:
            self.assert_state_prompt_has_no_format_rules(system_prompt, user_prompt)
            if schema["name"] == "exp1_realtalk_reference_judgment":
                return json.dumps({
                    "topic": "work",
                    "reflective": True,
                    "grounding": False,
                    "empathy": {
                        "emotional_reaction": 1,
                        "interpretation": 1,
                        "exploration": 0,
                    },
                })
            return json.dumps({
                "emotion": "joy",
                "sentiment": "positive",
                "topic": "work",
                "reflective": True,
                "grounding": False,
                "intimacy": 0.8,
                "empathy": {
                    "emotional_reaction": 1,
                    "interpretation": 1,
                    "exploration": 0,
                },
            })
        if "FLAT list" in system_prompt:
            return json.dumps({
                "work_preference": {
                    "value": "growth", "confidence": 0.8, "evidence": "I like work"
                }
            })
        return json.dumps({
            layer: {
                "item": {"value": layer, "confidence": 0.8, "evidence": "history"}
            }
            for layer in ("core", "regulation", "cognition", "identity", "behavior")
        })

    def assert_state_prompt_has_no_format_rules(self, system_prompt, user_prompt):
        combined = (system_prompt + "\n" + user_prompt).lower()
        assert "output json" not in combined
        assert "return only json" not in combined
        assert "one of:" not in combined


class FakeLabels:
    def annotate(self, _text):
        return {"emotion": "joy", "sentiment": "positive", "intimacy": 0.8}

    def metadata(self):
        return {"provider": "fake-pinned-labels"}


class AlternateFakeLabels(FakeLabels):
    def metadata(self):
        return {"provider": "different-label-revision"}


def _chat(first_speaker, second_speaker, prefix):
    chat = {"name": {"speaker_1": first_speaker, "speaker_2": second_speaker}}
    for index in range(1, 4):
        chat[f"session_{index}"] = [
            {
                "speaker": second_speaker,
                "clean_text": f"{prefix} partner {index}",
                "dia_id": f"{prefix}-p{index}",
            },
            {
                "speaker": first_speaker,
                "clean_text": f"{prefix} work update {index}",
                "dia_id": f"{prefix}-u{index}",
            },
        ]
    return chat


class Exp1MockRunTests(unittest.TestCase):
    def test_complete_triplets_and_resume_without_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "Chat_4_Emi_Paola.json").write_text(
                json.dumps(_chat("Emi", "Paola", "train")), encoding="utf-8"
            )
            (dataset / "Chat_1_Emi_Elise.json").write_text(
                json.dumps(_chat("Emi", "Elise", "test")), encoding="utf-8"
            )
            output = root / "output"
            config = Exp1Config(
                dataset_dir=str(dataset),
                output_dir=str(output),
                speaker_filter=["Emi"],
                max_eval_points_per_speaker=2,
            )
            llm = FakeLLM()
            summary = run_exp1(config, llm=llm, label_evaluator=FakeLabels())
            calls_after_first_run = len(llm.calls)
            resumed = run_exp1(config, llm=llm, label_evaluator=FakeLabels())

            self.assertEqual(summary["num_eval_points"], 2)
            self.assertEqual(summary["num_speakers"], 1)
            self.assertEqual(set(summary["comparison"]), set(METHODS))
            self.assertEqual(calls_after_first_run, 10)
            self.assertEqual(len(llm.calls), calls_after_first_run)
            prediction_calls = [
                kwargs for _, _, kwargs in llm.calls
                if kwargs.get("response_schema", {}).get("name")
                == "exp1_current_user_state"
            ]
            reference_calls = [
                kwargs for _, _, kwargs in llm.calls
                if kwargs.get("response_schema", {}).get("name")
                == "exp1_realtalk_reference_judgment"
            ]
            self.assertTrue(all(call["max_tokens"] == 2048 for call in prediction_calls))
            self.assertTrue(all(call["max_tokens"] == 1024 for call in reference_calls))
            self.assertTrue(all(
                kwargs["max_tokens"] == config.profile_max_tokens
                for _, _, kwargs in llm.calls
                if kwargs.get("response_schema") is None
            ))
            self.assertEqual(resumed["num_eval_points"], 2)
            lines = (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                result = json.loads(line)
                self.assertEqual(set(result["methods"]), set(METHODS))
                self.assertEqual(result["profile"]["train_sessions"], [
                    "session_1", "session_2", "session_3"
                ])
            metric_lines = (
                output / "metric_records.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(metric_lines), 2 * len(METHODS))
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "run_manifest.json").exists())

            with self.assertRaisesRegex(
                CheckpointSignatureError, "checkpoint does not match"
            ):
                run_exp1(config, llm=llm, label_evaluator=AlternateFakeLabels())


if __name__ == "__main__":
    unittest.main()

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
        if kwargs.get("response_schema") is not None:
            self.assert_state_prompt_has_no_format_rules(system_prompt, user_prompt)
            return json.dumps({
                "emotion": "joy", "sentiment": "positive", "topic": "work"
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
        return {"emotion": "joy", "sentiment": "positive"}

    def metadata(self):
        return {"provider": "fake-pinned-labels"}


class AlternateFakeLabels(FakeLabels):
    def metadata(self):
        return {"provider": "different-label-revision"}


class Exp1MockRunTests(unittest.TestCase):
    def test_complete_triplets_and_resume_without_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            persona_dir = dataset / "output" / "agent"
            persona_dir.mkdir(parents=True)
            (persona_dir / "agent_persona.json").write_text(
                json.dumps({"name": "Agent", "personality": "calm"}), encoding="utf-8"
            )
            chat = {"name": {"speaker_1": "User", "speaker_2": "Agent"}}
            for index in range(1, 5):
                chat[f"session_{index}"] = [
                    {"speaker": "Agent", "clean_text": f"hello {index}", "dia_id": f"a{index}"},
                    {"speaker": "User", "clean_text": f"work update {index}", "dia_id": f"u{index}"},
                ]
            (dataset / "Chat_1_User_Agent.json").write_text(
                json.dumps(chat), encoding="utf-8"
            )
            output = root / "output"
            config = Exp1Config(
                dataset_dir=str(dataset), output_dir=str(output),
                max_eval_points_per_chat=2,
            )
            llm = FakeLLM()
            summary = run_exp1(config, llm=llm, label_evaluator=FakeLabels())
            calls_after_first_run = len(llm.calls)
            resumed = run_exp1(config, llm=llm, label_evaluator=FakeLabels())

            self.assertEqual(summary["num_eval_points"], 2)
            self.assertEqual(set(summary["comparison"]), set(METHODS))
            self.assertEqual(calls_after_first_run, 12)
            self.assertEqual(len(llm.calls), calls_after_first_run)
            self.assertTrue(all(
                kwargs["max_tokens"] == 512
                for _, _, kwargs in llm.calls
                if kwargs.get("response_schema") is not None
            ))
            self.assertTrue(all(
                kwargs["max_tokens"] == 5000
                for _, _, kwargs in llm.calls
                if kwargs.get("response_schema") is None
            ))
            self.assertEqual(resumed["num_eval_points"], 2)
            lines = (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                self.assertEqual(set(json.loads(line)["methods"]), set(METHODS))
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "run_manifest.json").exists())

            with self.assertRaisesRegex(
                CheckpointSignatureError, "checkpoint does not match"
            ):
                run_exp1(config, llm=llm, label_evaluator=AlternateFakeLabels())

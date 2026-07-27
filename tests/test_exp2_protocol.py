import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.experiments.exp2_predictive_empathy import (
    Exp2Config,
    METHODS,
    run_exp2,
)
from src.experiments.exp2_schema import normalize_future_state


def _future_state():
    return {
        "future_emotion": "joy",
        "future_sentiment": "positive",
        "future_intimacy": 0.7,
        "future_topic": "work",
        "future_reflective": True,
        "future_grounding": False,
        "future_empathy": {
            "emotional_reaction": 1,
            "interpretation": 1,
            "exploration": 0,
        },
        "confidence": 0.8,
    }


def _reference_state():
    return {
        "emotion": "joy",
        "sentiment": "positive",
        "topic": "work",
        "reflective": True,
        "grounding": False,
        "intimacy": 0.7,
        "empathy": {
            "emotional_reaction": 1,
            "interpretation": 1,
            "exploration": 0,
        },
    }


class FakeExp2LLM:
    model = "fake-exp2-model"
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
        schema_name = (kwargs.get("response_schema") or {}).get("name")
        if schema_name == "exp2_future_user_state":
            return json.dumps(_future_state())
        if schema_name == "exp1_realtalk_reference_judgment":
            return json.dumps(_reference_state())
        return json.dumps({
            layer: {
                "item": {
                    "value": f"{layer}-value",
                    "confidence": 0.8,
                    "evidence": "ca history",
                }
            }
            for layer in ("core", "regulation", "cognition", "identity", "behavior")
        })


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


class Exp2ProtocolTests(unittest.TestCase):
    def test_schema_normalization_is_strict_and_idempotent(self):
        normalized = normalize_future_state(_future_state())
        self.assertEqual(normalized["emotion"], "joy")
        self.assertEqual(normalize_future_state(normalized), normalized)
        invalid = _future_state()
        invalid["future_emotion"] = "neutral"
        with self.assertRaisesRegex(ValueError, "unsupported future emotion"):
            normalize_future_state(invalid)

    def test_complete_quartets_are_causal_and_resumable(self):
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
            config = Exp2Config(
                dataset_dir=str(dataset),
                output_dir=str(output),
                speaker_filter=["Emi"],
                max_eval_points_per_speaker=2,
            )
            llm = FakeExp2LLM()
            summary = run_exp2(config, llm=llm)
            call_count = len(llm.calls)
            resumed = run_exp2(config, llm=llm)

            self.assertEqual(summary["num_eval_points"], 2)
            self.assertEqual(summary["num_speakers"], 1)
            self.assertEqual(set(summary["comparison"]), set(METHODS))
            self.assertEqual(call_count, 12)
            self.assertEqual(len(llm.calls), call_count)
            self.assertEqual(resumed["num_eval_points"], 2)

            future_prompts = [
                user_prompt
                for _, user_prompt, kwargs in llm.calls
                if (kwargs.get("response_schema") or {}).get("name")
                == "exp2_future_user_state"
            ]
            self.assertEqual(len(future_prompts), 2 * len(METHODS))
            self.assertEqual(future_prompts[2], future_prompts[3])
            self.assertTrue(all(
                "test work update 1" not in prompt
                for prompt in future_prompts[:4]
            ))
            self.assertTrue(all(
                "test work update 2" not in prompt
                for prompt in future_prompts[4:]
            ))
            self.assertNotIn("CONVERSATION HISTORY", future_prompts[0])
            self.assertIn("CONVERSATION HISTORY", future_prompts[5])
            self.assertIn('"core"', future_prompts[2])
            self.assertIn('"core"', future_prompts[3])
            self.assertNotIn(
                "CURRENT STATE DERIVED FROM OBSERVED HISTORY", future_prompts[2]
            )
            self.assertIn(
                "CURRENT STATE DERIVED FROM OBSERVED HISTORY", future_prompts[7]
            )

            results = [
                json.loads(line)
                for line in (output / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(results), 2)
            self.assertTrue(all(
                set(item["methods"]) == set(METHODS) for item in results
            ))
            self.assertTrue(all(
                item["context"]["target_visible_to_predictors"] is False
                for item in results
            ))
            self.assertTrue(all(item["context"]["turns"] is not None for item in results))
            self.assertTrue(all(item["ground_truth_response"] for item in results))
            self.assertEqual(
                results[0]["profile"]["train_sessions"],
                ["session_1", "session_2", "session_3"],
            )
            self.assertEqual(
                len(
                    (output / "metric_records.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                2 * len(METHODS),
            )

            expanded = run_exp2(
                replace(config, max_eval_points_per_speaker=3),
                llm=llm,
            )
            self.assertEqual(expanded["num_eval_points"], 3)
            self.assertEqual(len(llm.calls), call_count + 6)


if __name__ == "__main__":
    unittest.main()

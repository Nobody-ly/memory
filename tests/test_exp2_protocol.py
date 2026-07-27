import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.experiments.exp2_predictive_empathy import (
    Exp2Config,
    METHODS,
    _result_sort_key,
    run_exp2,
)
from src.experiments.exp2_schema import normalize_future_state


def _future_state():
    return {
        "future_emotion": "joy",
        "future_sentiment": "positive",
        "future_intimacy": 0.7,
        "future_topic": "work",
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


def _framework_state():
    return {
        "current_state": {
            "emotion": "encouraged",
            "stress_level": "low",
            "motivation": "high",
            "energy": "medium",
            "main_need": "encouragement",
            "state_summary": "The user is engaged and optimistic.",
        },
        "projected_state": {
            "next_emotion_trend": "stable positive",
            "possible_behavior": "share another update",
            "risk": "low",
            "recommended_intervention": "respond warmly",
        },
        "activated_persona": {
            "empathy_level": "medium",
            "teasing_level": "low",
            "warmth_level": "high",
            "guidance_level": "medium",
            "activated_tone": "warm and curious",
        },
    }


def _alignment():
    return {
        "understanding": {"self_domain": {}, "user_domain": {}},
        "prediction": {"projected_trend": "positive"},
        "exploration": {"decision": "balanced"},
        "alignment": {"empathy_adjustment": "moderate"},
        "empathy_state": {
            "empathy_level": "medium",
            "emotional_reaction": "1",
            "interpretation": "1",
            "exploration": "1",
            "activated_tone": "warm",
            "response_guidance": "acknowledge and ask one question",
        },
    }


class FakeLabelEvaluator:
    def __init__(self):
        self.messages = []

    def annotate(self, text):
        self.messages.append(text)
        return {
            "emotion": "joy",
            "sentiment": "positive",
            "intimacy": 0.7,
        }

    def metadata(self):
        return {"provider": "fake-pinned-realtalk"}


class FakeExp2LLM:
    model = "fake-exp2-model"
    enable_thinking = False
    max_retries = 6

    def __init__(self):
        self.calls = []
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
        }

    def chat(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt, kwargs))
        self.token_usage["prompt_tokens"] += 10
        self.token_usage["completion_tokens"] += 2
        self.token_usage["calls"] += 1
        schema_name = (kwargs.get("response_schema") or {}).get("name")
        if schema_name == "exp2_future_user_state":
            return json.dumps(_future_state())
        if schema_name == "exp2_framework_state":
            return json.dumps(_framework_state())
        if schema_name == "exp1_realtalk_reference_judgment":
            return json.dumps(_reference_state())
        if "extracting agent personas" in system_prompt:
            return json.dumps({
                "name": "elise",
                "personality": "warm and conversational",
                "tone": "friendly",
                "interaction_principles": ["listen", "ask relevant questions"],
                "expression_patterns": ["That sounds interesting"],
            })
        if "empathy alignment reasoning module" in system_prompt:
            return json.dumps(_alignment())
        if "personalized companion agent" in system_prompt:
            return "That sounds like a meaningful update. How did it feel?"
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
                "speaker": first_speaker,
                "clean_text": f"{prefix} work update {index}",
                "dia_id": f"{prefix}-u{index}",
            },
            {
                "speaker": second_speaker,
                "clean_text": f"{prefix} partner reply {index}",
                "dia_id": f"{prefix}-p{index}",
            },
        ]
    return chat


class Exp2ProtocolTests(unittest.TestCase):
    def test_result_sort_key_uses_numeric_message_order(self):
        results = [
            {
                "test_chat_file": "Chat_1.json",
                "speaker": "Emi",
                "message_level_index": index,
            }
            for index in (10, 2, 1)
        ]
        self.assertEqual(
            [
                item["message_level_index"]
                for item in sorted(results, key=_result_sort_key)
            ],
            [1, 2, 10],
        )

    def test_schema_normalization_is_strict_and_idempotent(self):
        normalized = normalize_future_state(_future_state())
        self.assertEqual(normalized["emotion"], "joy")
        self.assertEqual(normalize_future_state(normalized), normalized)
        invalid = _future_state()
        invalid["future_emotion"] = "neutral"
        with self.assertRaisesRegex(ValueError, "unsupported future emotion"):
            normalize_future_state(invalid)

    def test_prediction_and_generation_are_causal_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "Chat_4_Emi_Paola.json").write_text(
                json.dumps(_chat("Emi", "Paola", "profile")),
                encoding="utf-8",
            )
            (dataset / "Chat_2_Kevin_Elise.json").write_text(
                json.dumps(_chat("Kevin", "elise", "persona")),
                encoding="utf-8",
            )
            (dataset / "Chat_1_Emi_Elise.json").write_text(
                json.dumps(_chat("Emi", "elise", "test")),
                encoding="utf-8",
            )
            output = root / "output"
            config = Exp2Config(
                dataset_dir=str(dataset),
                output_dir=str(output),
                speaker_filter=["Emi"],
                max_eval_points_per_speaker=2,
            )
            llm = FakeExp2LLM()
            evaluator = FakeLabelEvaluator()
            summary = run_exp2(config, llm=llm, label_evaluator=evaluator)
            call_count = len(llm.calls)
            resumed = run_exp2(config, llm=llm, label_evaluator=evaluator)

            self.assertEqual(summary["num_eval_points"], 2)
            self.assertEqual(summary["num_speakers"], 1)
            self.assertEqual(set(summary["comparison"]), set(METHODS))
            self.assertEqual(len(llm.calls), call_count)
            self.assertEqual(resumed["num_eval_points"], 2)
            self.assertTrue(all(
                value["prediction"]["num_evaluations"] == 2
                and value["generation"]["num_evaluations"] == 2
                for value in summary["comparison"].values()
            ))

            future_prompts = [
                user_prompt
                for _, user_prompt, kwargs in llm.calls
                if (kwargs.get("response_schema") or {}).get("name")
                == "exp2_future_user_state"
            ]
            self.assertEqual(len(future_prompts), 6)
            self.assertTrue(all(
                "test work update 1" not in prompt
                for prompt in future_prompts[:2]
            ))
            self.assertTrue(all(
                "test work update 2" not in prompt
                for prompt in future_prompts[2:]
            ))
            second_point = future_prompts[-len(METHODS):]
            self.assertIn("test partner reply 1", second_point[0])
            self.assertNotIn("CONVERSATION HISTORY", second_point[0])
            self.assertEqual(second_point[1].count("test work update 1"), 1)
            self.assertEqual(second_point[1].count("test partner reply 1"), 1)
            self.assertIn('"core"', second_point[2])
            self.assertIn('"core"', second_point[3])
            self.assertNotIn("DEEP EMPATHY FRAMEWORK", second_point[2])
            self.assertIn("CURRENT STATE DERIVED FROM OBSERVED HISTORY", second_point[3])

            state_calls = [
                user_prompt
                for _, user_prompt, kwargs in llm.calls
                if (kwargs.get("response_schema") or {}).get("name")
                == "exp2_framework_state"
            ]
            self.assertEqual(len(state_calls), 1)
            self.assertIn("test work update 1", state_calls[0])
            self.assertIn("test partner reply 1", state_calls[0])
            self.assertNotIn("test work update 2", state_calls[0])

            generation_prompts = [
                user_prompt
                for system_prompt, user_prompt, kwargs in llm.calls
                if "personalized companion agent" in system_prompt
                and not kwargs.get("response_schema")
            ]
            self.assertEqual(len(generation_prompts), 2 * len(METHODS))
            self.assertTrue(all(
                "Do not invent a recent event or current activity" in prompt
                for prompt in generation_prompts
            ))
            self.assertTrue(any(
                "test work update 1" in prompt for prompt in generation_prompts
            ))
            self.assertTrue(any(
                "test work update 2" in prompt for prompt in generation_prompts
            ))

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
                set(item["methods"][method]["generation"])
                == {"response", "response_ei", "scores"}
                for item in results
                for method in METHODS
            ))
            self.assertEqual(results[0]["framework_state"], {})
            self.assertEqual(
                results[1]["framework_state"]["current_state"]["emotion"],
                "encouraged",
            )
            self.assertTrue(all(
                item["context"]["target_visible_to_predictors"] is False
                for item in results
            ))
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
                2 * len(METHODS) * 2,
            )
            alignment_prompts = [
                user_prompt
                for system_prompt, user_prompt, _ in llm.calls
                if "empathy alignment reasoning module" in system_prompt
            ]
            self.assertIn(
                "EPISTEMIC VALUE DECAY omega(t): 0.875",
                alignment_prompts[0],
            )
            self.assertIn(
                "EPISTEMIC VALUE DECAY omega(t): 0.8323",
                alignment_prompts[1],
            )

            expanded = run_exp2(
                replace(config, max_eval_points_per_speaker=3),
                llm=llm,
                label_evaluator=evaluator,
            )
            self.assertEqual(expanded["num_eval_points"], 3)
            self.assertGreater(len(llm.calls), call_count)


if __name__ == "__main__":
    unittest.main()

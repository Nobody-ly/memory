from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.experiments.personaemp.client import (
    ChatResult,
    OpenAICompatibleChatBackend,
    _is_retryable,
)
from src.experiments.personaemp.dataset import (
    PersonaEmpDataset,
    PersonaEmpDatasetError,
)
from src.experiments.personaemp.generation import (
    RESPONSE_MAX_TOKENS,
    PROFILE_MAX_TOKENS,
    PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
    BaseModelGenerator,
    DeepEmpathyGenerator,
    ProfileBuilder,
    ProfileCache,
    StageUsage,
    _parse_json_object,
)
from src.experiments.personaemp.runner import (
    PersonaEmpRunner,
    RunConfiguration,
)
from src.experiments.personaemp.official_eval import (
    summarize_official_results,
    validate_criteria_alignment,
    validate_prediction_alignment,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "personaemp_paper_case.json"


class FakeBackend:
    model = "fake-qwen3-8b"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, object] | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "system": system_prompt,
                "user": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_schema": response_schema,
            }
        )
        if "extracting user profiles" in system_prompt:
            content = json.dumps(
                {
                    "core": {
                        "social_values": {
                            "value": "deep friendships and autonomy",
                            "confidence": 0.9,
                        }
                    },
                    "regulation": {
                        "coping": {
                            "value": "quiet intellectual hobbies",
                            "confidence": 0.8,
                        }
                    },
                    "cognition": {
                        "communication": {
                            "value": "gentle and non-pressuring",
                            "confidence": 0.8,
                        }
                    },
                    "identity": {
                        "self_view": {
                            "value": "introverted",
                            "confidence": 0.9,
                        }
                    },
                    "behavior": {
                        "social_pattern": {
                            "value": "prefers a close friend group",
                            "confidence": 0.9,
                        }
                    },
                }
            )
        elif "empathy alignment reasoning module" in system_prompt:
            content = json.dumps(
                {
                    "understanding": {
                        "user_domain": {
                            "current_emotion": "conflicted",
                            "underlying_need": "permission to protect social energy",
                        }
                    },
                    "prediction": {
                        "projected_trend": "continued guilt",
                    },
                    "exploration": {
                        "decision": "balanced",
                        "exploration_focus": "preferred boundary wording",
                    },
                    "empathy_state": {
                        "empathy_level": "high",
                        "activated_tone": "warm and validating",
                    },
                }
            )
        else:
            content = (
                "It makes sense to protect the small circle that helps you recharge. "
                "Would a kind, honest boundary feel more comfortable than forcing yourself?"
            )
        return ChatResult(
            content=content,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            latency_seconds=0.01,
            attempts=1,
        )


class RecordingCompletions:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def create(self, **request: object) -> SimpleNamespace:
        self.request = request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="response")
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=2,
            ),
        )


class PersonaEmpClientTests(unittest.TestCase):
    def test_does_not_retry_authentication_errors(self) -> None:
        authentication_error = RuntimeError("invalid authentication")
        authentication_error.status_code = 401
        rate_limit_error = RuntimeError("rate limited")
        rate_limit_error.status_code = 429

        self.assertFalse(_is_retryable(authentication_error))
        self.assertTrue(_is_retryable(rate_limit_error))

    def test_kimi_k26_non_thinking_parameters_follow_official_api(self) -> None:
        completions = RecordingCompletions()
        backend = object.__new__(OpenAICompatibleChatBackend)
        backend.model = "kimi-k2.6"
        backend.max_attempts = 1
        backend.enable_thinking = False
        backend.is_kimi_k2 = True
        backend.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        backend.chat(
            "system",
            "user",
            temperature=0.2,
            max_tokens=100,
        )

        assert completions.request is not None
        self.assertEqual(completions.request["temperature"], 0.6)
        self.assertEqual(
            completions.request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )


class PersonaEmpDatasetTests(unittest.TestCase):
    def test_loads_official_shape(self) -> None:
        dataset = PersonaEmpDataset.load(FIXTURE)
        self.assertEqual(len(dataset.raw_sessions), 1)
        self.assertEqual(len(dataset.samples), 3)
        self.assertEqual(dataset.samples[0].category, "Emotional Support")
        self.assertEqual(len(dataset.fingerprint), 64)

    def test_rejects_duplicate_query_ids(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        raw[0]["queries"][1]["query_id"] = raw[0]["queries"][0]["query_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(PersonaEmpDatasetError):
                PersonaEmpDataset.load(path)


class DeepEmpathyGenerationTests(unittest.TestCase):
    def test_structured_parser_closes_only_unbalanced_eof_containers(self) -> None:
        parsed = _parse_json_object(
            '{"core":{},"regulation":{},"cognition":{},'
            '"identity":{},"behavior":{"patterns":[]}'
        )
        self.assertEqual(parsed["behavior"], {"patterns": []})

        with self.assertRaises(json.JSONDecodeError):
            _parse_json_object('{"core":{} "regulation":{}}')

    def test_stage_usage_includes_all_logical_schema_attempts(self) -> None:
        usage = StageUsage.combine(
            [
                ChatResult(
                    content="invalid",
                    model="fake-qwen3-8b",
                    prompt_tokens=7,
                    completion_tokens=2,
                    latency_seconds=0.1,
                    attempts=2,
                ),
                ChatResult(
                    content="valid",
                    model="fake-qwen3-8b",
                    prompt_tokens=8,
                    completion_tokens=3,
                    latency_seconds=0.2,
                    attempts=1,
                ),
            ]
        )

        self.assertEqual(usage.logical_calls, 2)
        self.assertEqual(usage.attempts, 3)
        self.assertEqual(usage.prompt_tokens, 15)
        self.assertEqual(usage.completion_tokens, 5)
        self.assertEqual(usage.latency_seconds, 0.3)

    def test_uses_shared_contract_and_only_allowed_raw_evidence(self) -> None:
        dataset = PersonaEmpDataset.load(FIXTURE)
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            builder = ProfileBuilder(
                backend,
                ProfileCache(Path(directory) / "profiles"),
            )
            generator = DeepEmpathyGenerator(backend, builder)
            sample = dataset.samples[0]
            output = generator.generate(sample)
            base_output = BaseModelGenerator(backend).generate(sample)

        self.assertEqual(output.method, "ours")
        self.assertTrue(output.response)
        self.assertIn(
            "five_layer_profile",
            output.to_record()["qualitative_artifacts"],
        )
        self.assertIn(
            "prediction",
            output.to_record()["qualitative_artifacts"],
        )
        self.assertIn(
            "exploration",
            output.to_record()["qualitative_artifacts"],
        )
        self.assertEqual(base_output.method, "base_model")
        ours_call = backend.calls[-2]
        base_call = backend.calls[-1]
        self.assertEqual(
            ours_call["system"],
            PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
        )
        self.assertEqual(
            base_call["system"],
            PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
        )
        self.assertIn(
            "exactly one paragraph containing 2 to 4",
            str(ours_call["system"]),
        )
        self.assertIn(
            "actionable suggestion or example phrase",
            str(base_call["system"]),
        )
        self.assertEqual(ours_call["max_tokens"], RESPONSE_MAX_TOKENS)
        self.assertEqual(base_call["max_tokens"], RESPONSE_MAX_TOKENS)
        self.assertEqual(ours_call["temperature"], base_call["temperature"])
        self.assertIn(sample.query, str(ours_call["user"]))
        self.assertIn(sample.query, str(base_call["user"]))
        self.assertIn("close friend group", str(ours_call["user"]))
        self.assertIn("close friend group", str(base_call["user"]))
        self.assertIn("empathy_state", str(ours_call["user"]))
        self.assertNotIn("empathy_state", str(base_call["user"]))
        profile_prompt = next(
            str(call["user"])
            for call in backend.calls
            if "extracting user profiles" in str(call["system"])
        )
        self.assertIn("Extracted long-term memory evidence", profile_prompt)
        profile_call = next(
            call
            for call in backend.calls
            if "extracting user profiles" in str(call["system"])
        )
        alignment_call = next(
            call
            for call in backend.calls
            if "empathy alignment reasoning module" in str(call["system"])
        )
        self.assertEqual(
            profile_call["response_schema"]["schema"]["required"],
            ["core", "regulation", "cognition", "identity", "behavior"],
        )
        self.assertEqual(profile_call["max_tokens"], PROFILE_MAX_TOKENS)
        self.assertIn(
            "empathy_state",
            alignment_call["response_schema"]["schema"]["required"],
        )
        forbidden_values = (
            sample.persona_text,
            sample.scenario,
            sample.category,
            sample.conversation[0]["text"],
        )
        for call in backend.calls:
            combined = f"{call['system']}\n{call['user']}"
            for forbidden in forbidden_values:
                self.assertNotIn(forbidden, combined)

    def test_runner_resumes_without_duplicate_calls(self) -> None:
        dataset = PersonaEmpDataset.load(FIXTURE)
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            builder = ProfileBuilder(
                backend,
                ProfileCache(output_dir / "cache" / "profiles"),
            )
            generator = DeepEmpathyGenerator(backend, builder)
            config = RunConfiguration(
                methods=("ours",),
                limit=1,
                dataset_provenance="paper_case_pilot",
                expected_table1_dataset_sha256=None,
                generator_model=backend.model,
                generator_base_url="https://example.invalid/v1",
                generator_enable_thinking=False,
            )
            runner = PersonaEmpRunner(
                repository_root=ROOT,
                dataset=dataset,
                output_dir=output_dir,
                config=config,
                generators={"ours": generator},
            )
            first_summary = runner.run()
            call_count = len(backend.calls)
            second_summary = runner.run()

            predictions = json.loads(
                (output_dir / "predictions" / "ours.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            cache_entries = list(
                (output_dir / "cache" / "profiles").glob("*.json")
            )
            cached_profile = json.loads(
                cache_entries[0].read_text(encoding="utf-8")
            )

        self.assertEqual(first_summary["successful_results"], 1)
        self.assertEqual(second_summary["successful_results"], 1)
        self.assertEqual(len(backend.calls), call_count)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(predictions[0]["responses"]), 1)
        self.assertFalse(
            manifest["dataset"]["table1_direct_comparison_allowed"]
        )
        self.assertEqual(cached_profile["format_version"], 2)
        self.assertEqual(
            summary["profile_preprocessing"]["profiles_with_usage"],
            1,
        )
        self.assertEqual(
            summary["online_inference"]["ours"]["included_stages"],
            ["alignment", "response"],
        )
        self.assertTrue(
            summary["online_inference"]["ours"][
                "profile_preprocessing_excluded"
            ]
        )


class OfficialEvaluationAdapterTests(unittest.TestCase):
    def test_rejects_misaligned_or_incomplete_criteria(self) -> None:
        dataset = PersonaEmpDataset.load(FIXTURE)
        criteria = [
            {
                "session_id": dataset.raw_sessions[0]["session_id"],
                "criterias": [
                    {
                        "query_id": sample.query_id,
                        "resonation": "criterion",
                        "expression": "criterion",
                        "reception": "criterion",
                    }
                    for sample in dataset.samples
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            criteria_path = Path(directory) / "criteria.json"
            criteria_path.write_text(json.dumps(criteria), encoding="utf-8")
            self.assertEqual(
                validate_criteria_alignment(FIXTURE, criteria_path),
                3,
            )

            criteria[0]["criterias"][1]["query_id"] = "wrong-query"
            criteria_path.write_text(json.dumps(criteria), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_criteria_alignment(FIXTURE, criteria_path)

    def test_validates_prediction_alignment_and_summarizes_scores(self) -> None:
        dataset = PersonaEmpDataset.load(FIXTURE)
        predictions = dataset.prediction_template()
        for session in predictions:
            for response in session["responses"]:
                response["response"] = "A non-empty empathetic response."

        official_results = [
            {
                "session_id": sample.session_id,
                "query_id": sample.query_id,
                "resonation": {"score": 4.0},
                "expression": {"score": 3.0},
                "reception": {"score": 5.0},
            }
            for sample in dataset.samples
        ]

        with tempfile.TemporaryDirectory() as directory:
            prediction_path = Path(directory) / "predictions.json"
            result_path = Path(directory) / "results.json"
            prediction_path.write_text(
                json.dumps(predictions),
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(official_results),
                encoding="utf-8",
            )
            sessions, queries = validate_prediction_alignment(
                FIXTURE,
                prediction_path,
            )
            summary = summarize_official_results(result_path)

        self.assertEqual(sessions, 1)
        self.assertEqual(queries, 3)
        self.assertEqual(summary["average_raw_1_to_5"], 4.0)
        self.assertEqual(summary["average_normalized_0_to_1"], 0.8)
        self.assertEqual(summary["invalid_scores"], [])


if __name__ == "__main__":
    unittest.main()

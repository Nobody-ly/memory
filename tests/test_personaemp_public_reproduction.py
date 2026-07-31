from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.experiments.personaemp.client import (
    ChatResult,
    OpenAICompatibleChatBackend,
)
from src.experiments.personaemp.dataset import PersonaEmpDataset
from src.experiments.personaemp.generation import (
    JsonCache,
    MemoryGenerator,
    MemorySummaryBuilder,
    RAGGenerator,
    RAGRetriever,
)
from src.experiments.personaemp.reconstruction import (
    IntentCache,
    IntentReconstructor,
    adapt_alpsbench,
    apply_official_model_compatibility,
)
from src.experiments.personaemp.report import build_report, paired_user_bootstrap
from src.experiments.personaemp.splitting import (
    TRAITS,
    build_ood_split,
    random_user_split,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "personaemp_paper_case.json"


class RecordingBackend:
    model = "fake-kimi-k2.6"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
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
        content = (
            "The user values close friendships and gentle, practical support."
            if "summarize user characteristics" in system_prompt
            else "That conflict makes sense; a gentle boundary can protect your energy."
        )
        return ChatResult(
            content=content,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            latency_seconds=0.01,
            attempts=1,
        )


class FixedEncoder:
    model_name = "fixed-test-encoder"

    def encode_query(self, query: str) -> list[float]:
        return [1.0, 0.0]

    def encode_memories(self, memories: tuple[str, ...]) -> list[list[float]]:
        return [
            [0.1, 0.9],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
        ][: len(memories)]


class StubIntentClassifier:
    def classify(self, record: dict[str, Any]) -> list[str]:
        return ["Personal Advice"]


class StructuredBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls += 1
        return ChatResult(
            content='{"intents":["Personal Advice"]}',
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            latency_seconds=0.01,
            attempts=1,
        )


class FakeCompletions:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def create(self, **request: Any) -> Any:
        self.request = request
        tool_name = request["tools"][0]["function"]["name"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name=tool_name,
                                    arguments='{"intents":["Personal Advice"]}',
                                )
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )


class PersonaEmpPublicReproductionTests(unittest.TestCase):
    def test_kimi_structured_output_uses_required_tool_schema(self) -> None:
        completions = FakeCompletions()
        backend = OpenAICompatibleChatBackend.__new__(
            OpenAICompatibleChatBackend
        )
        backend.model = "kimi-k2.6"
        backend.base_url = "https://api.moonshot.cn/v1"
        backend.max_attempts = 1
        backend.enable_thinking = False
        backend.is_kimi_k2 = True
        backend.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        schema = {
            "name": "intent_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "intents": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["intents"],
                "additionalProperties": False,
            },
        }

        result = backend.chat(
            "system",
            "user",
            temperature=0.0,
            max_tokens=100,
            response_schema=schema,
        )

        self.assertEqual(result.content, '{"intents":["Personal Advice"]}')
        self.assertIsNotNone(completions.request)
        assert completions.request is not None
        self.assertIn("tools", completions.request)
        self.assertIn("tool_choice", completions.request)
        self.assertNotIn("response_format", completions.request)

    def test_dashscope_qwen_structured_output_uses_required_tool_schema(
        self,
    ) -> None:
        completions = FakeCompletions()
        backend = OpenAICompatibleChatBackend.__new__(
            OpenAICompatibleChatBackend
        )
        backend.model = "qwen3-30b-a3b-instruct-2507"
        backend.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        backend.max_attempts = 1
        backend.enable_thinking = False
        backend.is_kimi_k2 = False
        backend.is_dashscope_qwen = True
        backend.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        schema = {
            "name": "intent_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "intents": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["intents"],
                "additionalProperties": False,
            },
        }

        result = backend.chat(
            "system",
            "user",
            temperature=0.0,
            max_tokens=100,
            response_schema=schema,
        )

        self.assertEqual(result.content, '{"intents":["Personal Advice"]}')
        assert completions.request is not None
        self.assertIn("tools", completions.request)
        self.assertIn("tool_choice", completions.request)
        self.assertNotIn("response_format", completions.request)

    def test_alpsbench_adapter_joins_gold_and_reconstructs_intent(self) -> None:
        input_row = {
            "benchmark_id": "bench-1",
            "task": "task1",
            "session_id": "session-1",
            "input": {
                "line_index": 7,
                "sessions": [
                    {
                        "session_id": "session-1",
                        "turns": [
                            {"role": "user", "text": "I feel stuck."},
                            {"role": "assistant", "text": "Tell me more."},
                        ],
                    }
                ],
                "dialogue": [{"role": "user", "text": "I feel stuck."}],
            },
        }
        reference_row = {
            "benchmark_id": "bench-1",
            "gold": {
                "memory_items": [
                    {
                        "memory_id": "m1",
                        "label": "States_Experiences/Mental_State",
                        "value": "Feels stuck about work",
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            reference_path = Path(directory) / "reference.jsonl"
            input_path.write_text(json.dumps(input_row) + "\n", encoding="utf-8")
            reference_path.write_text(
                json.dumps(reference_row) + "\n",
                encoding="utf-8",
            )
            records, stats = adapt_alpsbench(
                [(input_path, reference_path)],
                StubIntentClassifier(),  # type: ignore[arg-type]
            )

        self.assertEqual(stats.joined_rows, 1)
        self.assertEqual(stats.memory_items, 1)
        self.assertEqual(records[0]["intents_ranked"][0]["intent_category"], "Personal Advice")
        self.assertNotIn("persona", records[0])
        self.assertNotIn("situation", records[0])

    def test_intent_cache_invalidates_when_model_changes(self) -> None:
        record = {
            "benchmark_id": "bench-1",
            "input": {
                "dialogue": [{"role": "user", "text": "I need advice."}]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = IntentCache(Path(directory) / "intents.jsonl")
            first_backend = StructuredBackend("model-a")
            first = IntentReconstructor(first_backend, cache)
            first.classify(record)
            first.classify(record)
            second_backend = StructuredBackend("model-b")
            second = IntentReconstructor(second_backend, cache)
            second.classify(record)

        self.assertEqual(first_backend.calls, 1)
        self.assertEqual(second_backend.calls, 1)

    def test_kimi_official_compatibility_changes_transport_only(self) -> None:
        source = """resp = await client.chat.completions.create(
                        temperature=temperature,
                        # top_p=top_p,
                    )
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_dir = root / "prepare_dataset"
            prepare_dir.mkdir()
            api_client = prepare_dir / "api_call.py"
            api_client.write_text(source, encoding="utf-8")
            apply_official_model_compatibility(
                prepare_dir,
                model="kimi-k2.6",
                output_dir=root,
            )
            patched = api_client.read_text(encoding="utf-8")
            manifest = json.loads(
                (root / "model_compatibility.json").read_text(encoding="utf-8")
            )

        self.assertIn("PERSONAEMP_KIMI_NONTHINKING_COMPAT", patched)
        self.assertIn('"type": "disabled"', patched)
        self.assertNotIn("prompt", patched.lower())
        self.assertTrue(manifest["transport_only"])
        self.assertFalse(manifest["prompt_or_protocol_changed"])
        self.assertTrue(manifest["kimi_nonthinking_patch_active"])

    def test_dataset_normalizes_hei_and_keeps_relevant_memory_for_diagnostics(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        raw[0]["queries"][0]["category"] = "HEI"
        raw[0]["queries"][0]["relevant_mem"] = [2, "3", 2, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            dataset = PersonaEmpDataset.load(path)
        self.assertEqual(dataset.samples[0].category, "High-EQ Interaction")
        self.assertEqual(dataset.samples[0].relevant_memory_indices, (2, 3))

    def test_memory_baseline_uses_summary_without_hidden_metadata(self) -> None:
        dataset = PersonaEmpDataset.load(FIXTURE)
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as directory:
            generator = MemoryGenerator(
                backend,
                MemorySummaryBuilder(
                    backend,
                    JsonCache(Path(directory) / "summaries"),
                ),
            )
            output = generator.generate(dataset.samples[0])
        self.assertEqual(output.method, "memory")
        self.assertEqual(len(backend.calls), 2)
        response_prompt = str(backend.calls[-1]["user"])
        self.assertIn("close friendships", response_prompt)
        self.assertIn(dataset.samples[0].query, response_prompt)
        self.assertNotIn(dataset.samples[0].persona_text, response_prompt)
        self.assertNotIn(dataset.samples[0].scenario, response_prompt)

    def test_rag_retrieves_exactly_top_three_without_using_gold_indices(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        raw[0]["extracted_memory"].append(
            {"label": "Preference", "value": "Fourth memory"}
        )
        raw[0]["queries"][0]["relevant_mem"] = [1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            dataset = PersonaEmpDataset.load(path)
            backend = RecordingBackend()
            retriever = RAGRetriever(
                FixedEncoder(),
                JsonCache(Path(directory) / "embeddings"),
            )
            output = RAGGenerator(backend, retriever).generate(dataset.samples[0])
        artifacts = output.qualitative_artifacts or {}
        self.assertEqual(artifacts["retrieved_memory_indices"], [2, 3, 4])
        self.assertEqual(artifacts["recall_at_3"], 0.0)
        self.assertNotIn(dataset.samples[0].persona_text, str(backend.calls[-1]["user"]))

    def test_random_split_is_user_level_and_reproducible(self) -> None:
        users = [f"user-{index}" for index in range(20)]
        first = random_user_split(users)
        second = random_user_split(users)
        self.assertEqual(first, second)
        self.assertEqual(len(first[1]), 2)
        self.assertFalse(set(first[0]).intersection(first[1]))

    def test_paired_bootstrap_uses_user_level_differences(self) -> None:
        baseline = {
            ("u1", "q1"): {"average": 2.0},
            ("u1", "q2"): {"average": 4.0},
            ("u2", "q3"): {"average": 3.0},
        }
        ours = {
            ("u1", "q1"): {"average": 3.0},
            ("u1", "q2"): {"average": 5.0},
            ("u2", "q3"): {"average": 4.0},
        }
        result = paired_user_bootstrap(
            baseline,
            ours,
            "average",
            iterations=1000,
        )
        self.assertEqual(result["users"], 2)
        self.assertEqual(result["delta"], 1.0)
        self.assertEqual(result["ci95_low"], 1.0)
        self.assertEqual(result["ci95_high"], 1.0)

    def test_ood_split_uses_categorical_traits_and_keeps_users_disjoint(self) -> None:
        patterns = (
            ("low", "low", "low", "low", "low"),
            ("medium", "medium", "medium", "medium", "medium"),
            ("high", "high", "high", "high", "high"),
        )
        labels = {
            f"user-{pattern_index}-{copy_index}": dict(zip(TRAITS, pattern))
            for pattern_index, pattern in enumerate(patterns)
            for copy_index in range(4)
        }
        split = build_ood_split(labels, k_min=2, k_max=5)
        self.assertGreaterEqual(split.selected_k, 2)
        self.assertTrue(split.train_users)
        self.assertTrue(split.test_users)
        self.assertFalse(set(split.train_users).intersection(split.test_users))
        self.assertEqual(
            set(split.train_users).union(split.test_users),
            set(labels),
        )

    def test_balanced_selection_fails_when_a_category_is_short(self) -> None:
        dataset = PersonaEmpDataset.load(
            ROOT / "tests" / "fixtures" / "personaemp_kimi_12_case.json"
        )
        with self.assertRaisesRegex(ValueError, "Social Strategy"):
            list(dataset.iter_balanced(4))

    def test_report_builds_aligned_four_method_artifacts(self) -> None:
        def official_rows(offset: float) -> list[dict[str, Any]]:
            return [
                {
                    "session_id": user,
                    "query_id": query,
                    "resonation": {"score": 2.0 + offset},
                    "expression": {"score": 3.0 + offset},
                    "reception": {"score": 4.0 + offset},
                }
                for user, query in (("u1", "q1"), ("u1", "q2"), ("u2", "q3"))
            ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = []
            offsets = {
                "base_model": 0.0,
                "memory": 0.2,
                "rag": 0.1,
                "ours": 0.5,
            }
            for method, offset in offsets.items():
                path = root / f"{method}.json"
                path.write_text(
                    json.dumps(official_rows(offset)),
                    encoding="utf-8",
                )
                specs.append(("random", "qwen3", method, path))
            output_dir = root / "report"
            report = build_report(specs, output_dir)

            self.assertEqual(
                report["settings"]["random:qwen3"]["ours_deltas"]["base_model"][
                    "average"
                ]["delta"],
                0.5,
            )
            self.assertTrue(
                (output_dir / "controlled_reproduction_report_zh.md").is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "charts"
                    / "random"
                    / "qwen3"
                    / "personaemp_metrics.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "official_table1_training_free_reference.csv"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_dir / "controlled_reproduction_summary.csv"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "controlled_reproduction_user_metrics.csv"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()

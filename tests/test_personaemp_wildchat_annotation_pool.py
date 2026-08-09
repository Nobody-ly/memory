from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.personaemp.client import ChatResult
from src.experiments.personaemp.alpsbench_two_stage import (
    ALPSBENCH_SOURCE_COMMIT,
    AlpsBenchTwoStageExtractor,
)
from src.experiments.personaemp.wildchat_annotation_pool import (
    AnnotationCheckpoint,
    AnnotationPoolExtractor,
    DEFAULT_INTENT_CATEGORIES,
    DEFAULT_INTENT_SUBTYPES,
    PAPER_MEMORY_MODEL,
    _annotation_subset,
    _checkpoint_identity,
    annotate_sample,
    scan_and_sample,
    write_by_label,
)


def _row(identifier: str, messages: int, language: str = "English") -> dict:
    return {
        "conversation_hash": identifier,
        "language": language,
        "hashed_ip": f"ip-{identifier}",
        "header": {"user-agent": "test", "accept-language": "en"},
        "conversation": [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"{identifier}-{index}",
            }
            for index in range(messages)
        ],
    }


class FixedBackend:
    model = PAPER_MEMORY_MODEL

    def chat(self, *args, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        return ChatResult(
            content=json.dumps(
                {
                    "intents_ranked": [
                        {
                            "intent_category": "Personal Interaction Intent",
                            "intent_subtype": "Personal Advice",
                        }
                    ],
                    "memory_items": [
                        {
                            "type": "direct",
                            "label": "States_Experiences/Mental_State",
                            "label_suggestion": "",
                            "value": "Feels worried about a recurring problem",
                            "reasoning": "The user states this directly.",
                            "evidence_turn_index": 0,
                            "confidence": 0.9,
                            "time_scope": "ongoing",
                            "emotion": "worry",
                            "preference_attitude": "",
                        }
                    ],
                }
            ),
            model=self.model,
            prompt_tokens=1,
            completion_tokens=1,
            latency_seconds=0.0,
            attempts=1,
        )


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Completion:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _SequentialCompletions:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.requests: list[dict] = []

    def create(self, **request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return _Completion(json.dumps(self.payloads.pop(0)))


class _SequentialClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _SequentialCompletions(payloads)


class OfficialBackend:
    model = PAPER_MEMORY_MODEL

    def __init__(self) -> None:
        item = {
            "memory_id": "m1",
            "type": "direct",
            "label": "Preferences/Food",
            "label_suggestion": None,
            "value": "Prefers tea",
            "reasoning": "The user states this preference.",
            "evidence": {
                "session_id": "placeholder",
                "utterance_index": 0,
                "text": "one-0",
            },
            "confidence": 0.95,
            "time_scope": "long_term",
            "emotion": None,
            "preference_attitude": "like",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        self.client = _SequentialClient(
            [
                {
                    "intents_ranked": [
                        {
                            "intent_category": "Personal Interaction Intent",
                            "intent_subtype": "Personal Advice",
                            "probability": 1.0,
                            "reasoning": "test",
                            "evidence": [],
                        }
                    ]
                },
                {"memory_items": [item]},
                {"memory_items": [item]},
            ]
        )


class AnnotationPoolTests(unittest.TestCase):
    def test_scan_uses_normalized_message_count_and_all_languages(self) -> None:
        malformed = _row("malformed", 6)
        malformed["conversation"].append({"role": "tool", "content": "ignored"})
        rows = [
            ("part.parquet", 0, _row("en", 6, "English")),
            ("part.parquet", 1, _row("zh", 7, "Chinese")),
            ("part.parquet", 2, _row("short", 5, "Russian")),
            ("part.parquet", 3, malformed),
        ]
        sample, stats = scan_and_sample(rows, sample_size=3, seed="fixed")
        self.assertEqual(stats.raw_rows, 4)
        self.assertEqual(stats.eligible_normalized_messages, 3)
        self.assertEqual(stats.eligible_raw_messages, 3)
        self.assertEqual(stats.language_counts["English"], 2)
        self.assertEqual(stats.language_counts["Chinese"], 1)
        self.assertEqual(stats.joint_stratum_counts["English|006-011"], 2)
        self.assertEqual(stats.joint_stratum_counts["Chinese|006-011"], 1)
        self.assertEqual(sum(stats.sample_stratum_counts.values()), 3)
        self.assertEqual(len(sample), 3)

    def test_sampling_is_stable(self) -> None:
        rows = [("part.parquet", index, _row(str(index), 6 + index % 10)) for index in range(50)]
        first, _ = scan_and_sample(rows, sample_size=12, seed="fixed")
        second, _ = scan_and_sample(rows, sample_size=12, seed="fixed")
        self.assertEqual(
            [row["source_key"] for row in first],
            [row["source_key"] for row in second],
        )

    def test_extractor_matches_personaemp_filter_contract(self) -> None:
        source, _ = scan_and_sample(
            [("part.parquet", 0, _row("one", 6))], sample_size=1, seed="fixed"
        )
        extractor = AnnotationPoolExtractor(
            FixedBackend(), DEFAULT_INTENT_CATEGORIES, DEFAULT_INTENT_SUBTYPES
        )
        record = extractor.extract(source[0])
        self.assertEqual(
            record["intents_ranked"][0]["intent_subtype"], "Personal Advice"
        )
        self.assertEqual(
            record["memory_items"][0]["label"],
            "States_Experiences/Mental_State",
        )
        self.assertEqual(record["memory_items"][0]["type"], "direct")

    def test_by_label_assigns_each_record_once(self) -> None:
        source, _ = scan_and_sample(
            [("part.parquet", 0, _row("one", 6))], sample_size=1, seed="fixed"
        )
        record = AnnotationPoolExtractor(
            FixedBackend(), DEFAULT_INTENT_CATEGORIES, DEFAULT_INTENT_SUBTYPES
        ).extract(source[0])
        with tempfile.TemporaryDirectory() as directory:
            counts = write_by_label(Path(directory), [record])
            files = list((Path(directory) / "by_label_json").glob("*.json"))
        self.assertEqual(sum(counts.values()), 1)
        self.assertEqual(len(files), 1)

    def test_annotation_subset_is_stable_and_bounded(self) -> None:
        sample, _ = scan_and_sample(
            [("part.parquet", index, _row(str(index), 6)) for index in range(20)],
            sample_size=20,
            seed="fixed",
        )
        first = _annotation_subset(sample, 7, "fixed")
        second = _annotation_subset(sample, 7, "fixed")
        self.assertEqual(
            [row["source_key"] for row in first],
            [row["source_key"] for row in second],
        )
        self.assertEqual(len(first), 7)

    def test_annotation_checkpoint_resumes_without_duplicate_calls(self) -> None:
        sample, _ = scan_and_sample(
            [("part.parquet", index, _row(str(index), 6)) for index in range(3)],
            sample_size=3,
            seed="fixed",
        )
        backend = FixedBackend()
        extractor = AnnotationPoolExtractor(
            backend, DEFAULT_INTENT_CATEGORIES, DEFAULT_INTENT_SUBTYPES
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = _checkpoint_identity(sample, extractor)
            checkpoint = AnnotationCheckpoint(
                root / "success.jsonl", root / "identity.json", identity
            )
            first, failures, cached = annotate_sample(
                sample, extractor, checkpoint
            )
            resumed = AnnotationCheckpoint(
                root / "success.jsonl", root / "identity.json", identity
            )
            second, second_failures, second_cached = annotate_sample(
                sample, extractor, resumed
            )
            lines = (root / "success.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(first), 3)
        self.assertEqual(failures, [])
        self.assertEqual(cached, 0)
        self.assertEqual(second, first)
        self.assertEqual(second_failures, [])
        self.assertEqual(second_cached, 3)
        self.assertEqual(len(lines), 3)

    def test_annotation_checkpoint_rejects_identity_change(self) -> None:
        sample, _ = scan_and_sample(
            [("part.parquet", 0, _row("one", 6))], sample_size=1, seed="fixed"
        )
        extractor = AnnotationPoolExtractor(
            FixedBackend(), DEFAULT_INTENT_CATEGORIES, DEFAULT_INTENT_SUBTYPES
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            AnnotationCheckpoint(
                root / "success.jsonl",
                root / "identity.json",
                _checkpoint_identity(sample, extractor),
            )
            with self.assertRaisesRegex(RuntimeError, "different sample"):
                AnnotationCheckpoint(
                    root / "success.jsonl",
                    root / "identity.json",
                    {"different": True},
                )

    def test_official_two_stage_adapter_preserves_three_call_protocol(self) -> None:
        sample, _ = scan_and_sample(
            [("part.parquet", 0, _row("one", 6))], sample_size=1, seed="fixed"
        )
        backend = OfficialBackend()
        extractor = AlpsBenchTwoStageExtractor(backend)  # type: ignore[arg-type]
        record = extractor.extract(sample[0])
        requests = backend.client.chat.completions.requests
        self.assertEqual(len(requests), 3)
        self.assertEqual([row["temperature"] for row in requests], [0.1, 0.2, 0.0])
        self.assertEqual(
            [row["response_format"] for row in requests],
            [{"type": "json_object"}] * 3,
        )
        self.assertEqual(record["memory_items"][0]["value"], "Prefers tea")
        self.assertEqual(len(record["memory_stage1_candidates"]), 1)
        self.assertEqual(
            record["reconstruction_metadata"]["upstream_commit"],
            ALPSBENCH_SOURCE_COMMIT,
        )


if __name__ == "__main__":
    unittest.main()
    _annotation_subset,
    _checkpoint_identity,
    annotate_sample,

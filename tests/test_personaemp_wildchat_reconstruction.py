from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.personaemp.client import ChatResult
from src.experiments.personaemp.wildchat_reconstruction import (
    MemoryExtractionCache,
    PAPER_MEMORY_MODEL,
    PaperMemoryExtractor,
    compare_memory_sets,
    extract_memories,
    paper_style_curation,
    select_long_dialogues,
)
from src.experiments.personaemp.wildchat_full_reconstruction import (
    _build_reconstructed_splits,
)
from src.experiments.personaemp.splitting import PAPER_BIG_FIVE_MODEL


ROOT = Path(__file__).resolve().parents[1]
PERSONAEMP_FIXTURE = ROOT / "tests" / "fixtures" / "personaemp_paper_case.json"


class FixedMemoryBackend:
    model = PAPER_MEMORY_MODEL

    def chat(self, *args, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        return ChatResult(
            content=json.dumps(
                {
                    "intents": ["Personal Advice"],
                    "memory_items": [
                        {
                            "type": "direct",
                            "label": "Preferences/Food",
                            "label_suggestion": "",
                            "value": "Prefers spicy noodles",
                            "reasoning": "Explicit preference",
                            "evidence_turn_index": 0,
                            "evidence_text": "I like spicy noodles.",
                            "confidence": 0.95,
                            "time_scope": "long_term",
                            "emotion": "",
                            "preference_attitude": "like",
                        },
                        {
                            "type": "implicit",
                            "label": "States_Experiences/Mental_State",
                            "label_suggestion": "",
                            "value": "Feels isolated after repeated social rejection",
                            "reasoning": "Repeated personal account",
                            "evidence_turn_index": 2,
                            "evidence_text": "People keep ignoring me and I feel alone.",
                            "confidence": 0.7,
                            "time_scope": "ongoing",
                            "emotion": "loneliness",
                            "preference_attitude": "",
                        },
                    ],
                }
            ),
            model=self.model,
            prompt_tokens=1,
            completion_tokens=1,
            latency_seconds=0.0,
            attempts=1,
        )


class DuplicateIntentBackend(FixedMemoryBackend):
    def chat(self, *args, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        result = super().chat(*args, **kwargs)
        payload = json.loads(result.content)
        payload["intents"] = ["Personal Advice", "Personal Advice"]
        return ChatResult(
            content=json.dumps(payload),
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_seconds=result.latency_seconds,
            attempts=result.attempts,
        )


class WrongModelBackend(FixedMemoryBackend):
    model = "qwen3-8b"


class BigFiveBackend:
    model = PAPER_BIG_FIVE_MODEL


class FlakyMemoryBackend(FixedMemoryBackend):
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *args, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary provider failure")
        return super().chat(*args, **kwargs)


class ContentRejectedBackend(FixedMemoryBackend):
    def chat(self, *args, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        raise RuntimeError(
            "Error code: 400; code=data_inspection_failed; "
            "Input data may contain inappropriate content"
        )


class WildChatReconstructionTests(unittest.TestCase):
    def test_full_reconstruction_builds_split_stage_after_dataset(self) -> None:
        expected = {
            "dataset_sha256": "fingerprint",
            "random": {"test_users": ["u1"]},
            "ood": {"test_users": ["u2"]},
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "src.experiments.personaemp.wildchat_full_reconstruction."
            "build_split_artifacts",
            return_value=expected,
        ) as build:
            result = _build_reconstructed_splits(
                PERSONAEMP_FIXTURE,
                Path(directory),
                BigFiveBackend(),  # type: ignore[arg-type]
            )
        self.assertEqual(result, expected)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(
            build.call_args.args[2].backend.model,
            PAPER_BIG_FIVE_MODEL,
        )

    def test_turn_filter_uses_paper_range(self) -> None:
        accepted = {
            "conversation": [
                {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
                for index in range(6)
            ],
            "conversation_id": "accepted",
        }
        rejected = {
            "conversation": [
                {"role": "user", "content": str(index)} for index in range(5)
            ],
            "conversation_id": "rejected",
        }
        records, stats = select_long_dialogues(
            [("source.jsonl", 0, accepted), ("source.jsonl", 1, rejected)]
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_conversation_id"], "accepted")
        self.assertEqual(stats.rejected_turn_range, 1)

    def test_memory_extraction_matches_personaemp_input_contract(self) -> None:
        record = {
            "session_id": "wc_test",
            "turns": [
                {"role": "user", "text": "I like spicy noodles."},
                {"role": "assistant", "text": "Noted."},
                {"role": "user", "text": "People keep ignoring me and I feel alone."},
                {"role": "assistant", "text": "That sounds painful."},
                {"role": "user", "text": "I want to find better friends."},
                {"role": "assistant", "text": "We can think about it."},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            extractor = PaperMemoryExtractor(
                FixedMemoryBackend(),
                MemoryExtractionCache(Path(directory) / "cache.jsonl"),
            )
            extracted, cached = extractor.extract(record)
            repeat, repeat_cached = extractor.extract(record)
        self.assertFalse(cached)
        self.assertTrue(repeat_cached)
        self.assertEqual(extracted, repeat)
        self.assertEqual(extracted["intents_ranked"][0]["intent_category"], "Personal Advice")
        self.assertEqual(extracted["memory_items"][0]["evidence"]["utterance_index"], 0)
        self.assertEqual(extracted["memory_items"][1]["type"], "implicit")

    def test_duplicate_intents_are_removed_after_structured_output(self) -> None:
        record = {
            "session_id": "wc_duplicate_intent",
            "turns": [
                {"role": "user", "text": "I like spicy noodles."},
                {"role": "assistant", "text": "Noted."},
                {"role": "user", "text": "I feel lonely sometimes."},
                {"role": "assistant", "text": "That sounds hard."},
                {"role": "user", "text": "I want more supportive friends."},
                {"role": "assistant", "text": "We can think about that."},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            extractor = PaperMemoryExtractor(
                DuplicateIntentBackend(),
                MemoryExtractionCache(Path(directory) / "cache.jsonl"),
            )
            extracted, _ = extractor.extract(record)
        self.assertEqual(
            extracted["intents_ranked"],
            [{"intent_category": "Personal Advice", "intent_subtype": ""}],
        )

    def test_memory_model_is_paper_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, PAPER_MEMORY_MODEL):
                PaperMemoryExtractor(
                    WrongModelBackend(),
                    MemoryExtractionCache(Path(directory) / "cache.jsonl"),
                )

    def test_transient_extraction_failure_is_not_cached(self) -> None:
        record = {
            "session_id": "wc_retry",
            "turns": [
                {"role": "user", "text": "I like spicy noodles."},
                {"role": "assistant", "text": "Noted."},
                {"role": "user", "text": "I feel lonely sometimes."},
                {"role": "assistant", "text": "That sounds hard."},
                {"role": "user", "text": "I want more supportive friends."},
                {"role": "assistant", "text": "We can think about that."},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = MemoryExtractionCache(Path(directory) / "cache.jsonl")
            backend = FlakyMemoryBackend()
            extractor = PaperMemoryExtractor(backend, cache)
            with self.assertRaisesRegex(RuntimeError, "temporary provider failure"):
                extractor.extract(record)
            self.assertFalse(cache.values)
            extracted, cached = extractor.extract(record)
        self.assertFalse(cached)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(extracted["session_id"], "wc_retry")

    def test_provider_content_rejection_is_terminal_and_auditable(self) -> None:
        record = {
            "session_id": "wc_content_rejected",
            "turns": [
                {"role": "user", "text": "Sensitive source text."},
                {"role": "assistant", "text": "Response."},
                {"role": "user", "text": "More context."},
                {"role": "assistant", "text": "Response."},
                {"role": "user", "text": "More context."},
                {"role": "assistant", "text": "Response."},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            extractor = PaperMemoryExtractor(
                ContentRejectedBackend(),
                MemoryExtractionCache(Path(directory) / "cache.jsonl"),
            )
            results, stats, failures, exclusions = extract_memories(
                [record], extractor
            )
        self.assertEqual(results, [])
        self.assertEqual(failures, [])
        self.assertEqual(stats.content_rejected, 1)
        self.assertEqual(stats.failed, 0)
        self.assertEqual(
            exclusions,
            [
                {
                    "session_id": "wc_content_rejected",
                    "status": "content_rejected",
                    "reason_code": "provider_data_inspection_failed",
                }
            ],
        )

    def test_curation_retains_implicit_memory_and_is_explicit_about_skipped_dedup(self) -> None:
        direct_only = {"session_id": "a", "memory_items": [{"type": "direct", "label": "Preferences/Food", "value": "tea"}]}
        implicit = {"session_id": "b", "memory_items": [{"type": "implicit", "label": "Preferences/Food", "value": "coffee"}]}
        selected, stats = paper_style_curation(
            [direct_only, implicit], category_cap=10, skip_semantic_dedup=True
        )
        self.assertEqual([record["session_id"] for record in selected], ["b"])
        self.assertEqual(stats["rejected_without_implicit_memory"], 1)
        self.assertEqual(stats["semantic_deduplication"]["status"], "skipped_by_flag")

    def test_curation_does_not_enable_unpublished_semantic_dedup_by_default(self) -> None:
        implicit = {
            "session_id": "a",
            "memory_items": [
                {"type": "implicit", "label": "Preferences/Food", "value": "coffee"}
            ],
        }
        selected, stats = paper_style_curation([implicit], category_cap=10)
        self.assertEqual([record["session_id"] for record in selected], ["a"])
        self.assertEqual(stats["semantic_deduplication"]["status"], "skipped_by_flag")

    def test_gold_comparison_rewards_identical_memory(self) -> None:
        item = {"type": "direct", "label": "Preferences/Food", "value": "likes tea"}
        comparison = compare_memory_sets([item], [item])
        self.assertEqual(comparison["f1"], 1.0)
        self.assertEqual(comparison["type_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

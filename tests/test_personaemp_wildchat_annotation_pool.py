from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.personaemp.client import ChatResult
from src.experiments.personaemp.wildchat_annotation_pool import (
    AnnotationPoolExtractor,
    DEFAULT_INTENT_CATEGORIES,
    DEFAULT_INTENT_SUBTYPES,
    PAPER_MEMORY_MODEL,
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


if __name__ == "__main__":
    unittest.main()

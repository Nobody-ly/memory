from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.realtalk_v15_report import build_paired_report


def _local(result_id: str, speaker: str, candidate_intimacy: float, rouge: float) -> dict:
    return {
        "result_id": result_id,
        "speaker": speaker,
        "local_labels": {
            "reference": {"sentiment": "positive", "emotion": "joy", "intimacy": 0.5},
            "candidate": {
                "sentiment": "positive",
                "emotion": "joy",
                "intimacy": candidate_intimacy,
            },
        },
        "local_metrics": {
            "rouge_l": rouge,
            "bertscore_f1": 0.8,
            "sentiment_accuracy": 1.0,
            "emotion_accuracy": 1.0,
            "intimacy_absolute_difference": abs(0.5 - candidate_intimacy),
        },
    }


def _gpt(result_id: str, speaker: str, reflective: bool, grounding: bool) -> dict:
    return {
        "result_id": result_id,
        "speaker": speaker,
        "reference": {
            "reflectiveness": True,
            "grounding": True,
            "empathy": {"emotional_reaction": 1, "interpretation": 1, "exploration": 1},
        },
        "candidate": {
            "reflectiveness": reflective,
            "grounding": grounding,
            "empathy": {"emotional_reaction": 1, "interpretation": 1, "exploration": 1},
        },
        "metrics": {
            "reflectiveness_accuracy": float(reflective),
            "grounding_accuracy": float(grounding),
            "empathy_absolute_difference": 0.0,
        },
    }


class RealTalkV15ReportTests(unittest.TestCase):
    def test_report_aligns_ids_and_reverses_ad_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: root / f"{name}.jsonl"
                for name in ("v9_local", "v15_local", "v9_gpt", "v15_gpt")
            }
            v9_local = [_local("a", "A", 0.8, 0.1), _local("b", "B", 0.7, 0.2)]
            v15_local = [_local("a", "A", 0.6, 0.2), _local("b", "B", 0.6, 0.3)]
            v9_gpt = [_gpt("a", "A", False, True), _gpt("b", "B", False, False)]
            v15_gpt = [_gpt("a", "A", True, True), _gpt("b", "B", True, True)]
            for path, rows in (
                (paths["v9_local"], v9_local),
                (paths["v15_local"], v15_local),
                (paths["v9_gpt"], v9_gpt),
                (paths["v15_gpt"], v15_gpt),
            ):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            report = build_paired_report(
                v9_local=paths["v9_local"],
                v15_local=paths["v15_local"],
                v9_gpt=paths["v9_gpt"],
                v15_gpt=paths["v15_gpt"],
                output_dir=root / "report",
                bootstrap_draws=100,
            )
            self.assertGreater(
                report["speaker_macro"]["intimacy_absolute_difference"]["signed_improvement"],
                0,
            )
            self.assertGreater(
                report["speaker_macro"]["reflectiveness_accuracy"]["signed_improvement"],
                0,
            )
            self.assertTrue((root / "report" / "paired_report.md").exists())

    def test_report_rejects_unaligned_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (
                ("v9_local", [_local("a", "A", 0.5, 0.1)]),
                ("v15_local", [_local("b", "A", 0.5, 0.1)]),
                ("v9_gpt", [_gpt("a", "A", True, True)]),
                ("v15_gpt", [_gpt("b", "A", True, True)]),
            ):
                (root / f"{name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "not identical"):
                build_paired_report(
                    v9_local=root / "v9_local.jsonl",
                    v15_local=root / "v15_local.jsonl",
                    v9_gpt=root / "v9_gpt.jsonl",
                    v15_gpt=root / "v15_gpt.jsonl",
                    output_dir=root / "report",
                    bootstrap_draws=10,
                )


if __name__ == "__main__":
    unittest.main()

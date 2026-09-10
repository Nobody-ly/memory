"""Paired V9/V15 REALTALK reporting with speaker-cluster bootstrap."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .realtalk_ours import PAPER_TABLE2_ROWS


LOCAL_METRICS = (
    "rouge_l",
    "bertscore_f1",
    "sentiment_accuracy",
    "emotion_accuracy",
    "intimacy_absolute_difference",
)
GPT_METRICS = (
    "reflectiveness_accuracy",
    "grounding_accuracy",
    "empathy_absolute_difference",
)
LOWER_IS_BETTER = frozenset({
    "intimacy_absolute_difference",
    "empathy_absolute_difference",
})
TARGET_METRICS = (
    "reflectiveness_accuracy",
    "grounding_accuracy",
    "intimacy_absolute_difference",
)


def build_paired_report(
    *,
    v9_local: Path,
    v15_local: Path,
    v9_gpt: Path,
    v15_gpt: Path,
    output_dir: Path,
    seed: int = 20260909,
    bootstrap_draws: int = 10000,
) -> dict[str, Any]:
    v9 = _merge_scores(v9_local, v9_gpt)
    v15 = _merge_scores(v15_local, v15_gpt)
    if set(v9) != set(v15):
        raise ValueError("V9 and V15 scored result IDs are not identical")
    ids = list(v9)
    if not ids:
        raise ValueError("paired report has no records")
    for result_id in ids:
        if v9[result_id]["speaker"] != v15[result_id]["speaker"]:
            raise ValueError(f"speaker mismatch for {result_id}")

    speakers = list(dict.fromkeys(v9[result_id]["speaker"] for result_id in ids))
    per_speaker = {}
    for speaker in speakers:
        speaker_ids = [result_id for result_id in ids if v9[result_id]["speaker"] == speaker]
        per_speaker[speaker] = _speaker_comparison(speaker_ids, v9, v15)

    metric_names = LOCAL_METRICS + GPT_METRICS
    macro = {}
    for metric in metric_names:
        v9_values = [per_speaker[speaker]["v9"][metric] for speaker in speakers]
        v15_values = [per_speaker[speaker]["v15"][metric] for speaker in speakers]
        signed = [per_speaker[speaker]["signed_improvement"][metric] for speaker in speakers]
        ci = _cluster_bootstrap_ci(signed, seed=seed, draws=bootstrap_draws)
        macro[metric] = {
            "v9": round(statistics.mean(v9_values), 6),
            "v15": round(statistics.mean(v15_values), 6),
            "signed_improvement": round(statistics.mean(signed), 6),
            "bootstrap_95ci": ci,
            "higher_signed_improvement_is_always_better": True,
        }

    diagnostics = _classification_diagnostics(ids, v9, v15)
    gate = _gate_decision(len(ids), macro, per_speaker)
    report = {
        "status": "complete",
        "records": len(ids),
        "speakers": speakers,
        "aggregation": "speaker_macro",
        "bootstrap": {
            "unit": "speaker",
            "seed": seed,
            "draws": bootstrap_draws,
            "interval": "percentile_95",
        },
        "speaker_macro": macro,
        "by_speaker": per_speaker,
        "diagnostics": diagnostics,
        "gate_decision": gate,
        "paper_table2": PAPER_TABLE2_ROWS,
        "comparison_status": "protocol_aligned_exploratory_comparison",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "paired_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def _merge_scores(local_path: Path, gpt_path: Path) -> dict[str, dict[str, Any]]:
    local_rows = _jsonl(local_path)
    gpt_rows = _jsonl(gpt_path)
    gpt_index = {row["result_id"]: row for row in gpt_rows}
    if len(gpt_index) != len(gpt_rows):
        raise ValueError(f"duplicate GPT result IDs in {gpt_path}")
    merged = {}
    for row in local_rows:
        result_id = row["result_id"]
        if result_id in merged:
            raise ValueError(f"duplicate local result ID {result_id}")
        if result_id not in gpt_index:
            raise ValueError(f"missing GPT scores for {result_id}")
        gpt = gpt_index[result_id]
        metrics = {name: float(row["local_metrics"][name]) for name in LOCAL_METRICS}
        metrics.update({name: float(gpt["metrics"][name]) for name in GPT_METRICS})
        labels = row["local_labels"]
        merged[result_id] = {
            "speaker": row["speaker"],
            "metrics": metrics,
            "local_labels": labels,
            "gpt_labels": {
                "reference": gpt["reference"],
                "candidate": gpt["candidate"],
            },
        }
    if set(merged) != set(gpt_index):
        raise ValueError("local and GPT result IDs do not align")
    return merged


def _speaker_comparison(
    ids: list[str],
    v9: dict[str, dict[str, Any]],
    v15: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    v9_means = {
        metric: statistics.mean(v9[result_id]["metrics"][metric] for result_id in ids)
        for metric in LOCAL_METRICS + GPT_METRICS
    }
    v15_means = {
        metric: statistics.mean(v15[result_id]["metrics"][metric] for result_id in ids)
        for metric in LOCAL_METRICS + GPT_METRICS
    }
    signed = {
        metric: (
            v9_means[metric] - v15_means[metric]
            if metric in LOWER_IS_BETTER
            else v15_means[metric] - v9_means[metric]
        )
        for metric in LOCAL_METRICS + GPT_METRICS
    }
    return {
        "records": len(ids),
        "v9": {key: round(value, 6) for key, value in v9_means.items()},
        "v15": {key: round(value, 6) for key, value in v15_means.items()},
        "signed_improvement": {key: round(value, 6) for key, value in signed.items()},
        "target_metrics_improved": sum(signed[metric] > 0 for metric in TARGET_METRICS),
    }


def _cluster_bootstrap_ci(values: list[float], *, seed: int, draws: int) -> list[float]:
    if not values:
        raise ValueError("bootstrap requires values")
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(rng.choice(values) for _ in values)
        for _ in range(draws)
    )
    low = samples[int(0.025 * (draws - 1))]
    high = samples[int(0.975 * (draws - 1))]
    return [round(low, 6), round(high, 6)]


def _classification_diagnostics(
    ids: list[str],
    v9: dict[str, dict[str, Any]],
    v15: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output = {}
    for metric in ("reflectiveness", "grounding"):
        output[metric] = {
            "v9": _binary_stats(ids, v9, metric),
            "v15": _binary_stats(ids, v15, metric),
        }
    for label, rows in (("v9", v9), ("v15", v15)):
        errors = [
            rows[result_id]["local_labels"]["candidate"]["intimacy"]
            - rows[result_id]["local_labels"]["reference"]["intimacy"]
            for result_id in ids
        ]
        output.setdefault("intimacy", {})[label] = {
            "mean_signed_error": round(statistics.mean(errors), 6),
            "mean_absolute_error": round(statistics.mean(abs(value) for value in errors), 6),
        }
    return output


def _binary_stats(
    ids: list[str], rows: dict[str, dict[str, Any]], metric: str
) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for result_id in ids:
        reference = bool(rows[result_id]["gpt_labels"]["reference"][metric])
        candidate = bool(rows[result_id]["gpt_labels"]["candidate"][metric])
        if reference and candidate:
            tp += 1
        elif candidate:
            fp += 1
        elif reference:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _gate_decision(
    records: int,
    macro: dict[str, dict[str, Any]],
    per_speaker: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    signed = {metric: values["signed_improvement"] for metric, values in macro.items()}
    target_improved = sum(signed[metric] > 0 for metric in TARGET_METRICS)
    speakers_two_of_three = sum(
        values["target_metrics_improved"] >= 2 for values in per_speaker.values()
    )
    checks: dict[str, bool] = {}
    if records == 30:
        checks = {
            "at_least_two_target_metrics_improve": target_improved >= 2,
            "reflectiveness_not_worse_0_03": signed["reflectiveness_accuracy"] >= -0.03,
            "grounding_not_worse_0_03": signed["grounding_accuracy"] >= -0.03,
            "intimacy_not_worse_0_01": signed["intimacy_absolute_difference"] >= -0.01,
        }
    elif records == 60:
        checks = {
            "at_least_two_target_metrics_improve": target_improved >= 2,
            "reflectiveness_not_worse_0_03": signed["reflectiveness_accuracy"] >= -0.03,
            "grounding_not_worse_0_03": signed["grounding_accuracy"] >= -0.03,
            "intimacy_not_worse_0_01": signed["intimacy_absolute_difference"] >= -0.01,
            "rouge_noninferior": signed["rouge_l"] >= -0.02,
            "bertscore_noninferior": signed["bertscore_f1"] >= -0.02,
            "sentiment_noninferior": signed["sentiment_accuracy"] >= -0.02,
            "emotion_noninferior": signed["emotion_accuracy"] >= -0.02,
            "empathy_noninferior": signed["empathy_absolute_difference"] >= -0.15,
        }
    elif records == 120:
        checks = {
            "all_target_metrics_improve": target_improved == 3,
            "reflectiveness_or_grounding_plus_0_03": max(
                signed["reflectiveness_accuracy"], signed["grounding_accuracy"]
            ) >= 0.03,
            "intimacy_improves_0_005": signed["intimacy_absolute_difference"] >= 0.005,
            "seven_speakers_improve_two_targets": speakers_two_of_three >= 7,
            "rouge_noninferior": signed["rouge_l"] >= -0.02,
            "bertscore_noninferior": signed["bertscore_f1"] >= -0.02,
            "sentiment_noninferior": signed["sentiment_accuracy"] >= -0.02,
            "emotion_noninferior": signed["emotion_accuracy"] >= -0.02,
            "empathy_noninferior": signed["empathy_absolute_difference"] >= -0.15,
        }
    return {
        "records": records,
        "automatic_gate_applicable": bool(checks),
        "checks": checks,
        "passed": all(checks.values()) if checks else None,
        "target_metrics_improved": target_improved,
        "speakers_improving_at_least_two_targets": speakers_two_of_three,
    }


def _markdown(report: dict[str, Any]) -> str:
    labels = {
        "rouge_l": "ROUGE",
        "bertscore_f1": "BERTScore",
        "reflectiveness_accuracy": "Reflectiveness",
        "grounding_accuracy": "Grounding",
        "sentiment_accuracy": "Sentiment",
        "emotion_accuracy": "Emotion",
        "intimacy_absolute_difference": "Intimacy AD",
        "empathy_absolute_difference": "Empathy AD",
    }
    table_keys = (
        "rouge_l",
        "bertscore_f1",
        "reflectiveness_accuracy",
        "grounding_accuracy",
        "sentiment_accuracy",
        "emotion_accuracy",
        "intimacy_absolute_difference",
        "empathy_absolute_difference",
    )
    paper_keys = (
        "lexical", "semantic", "reflective", "grounding",
        "sentiment", "emotion", "intimacy", "empathy",
    )
    lines = [
        "# REALTALK V9 / V15 Paired Report",
        "",
        f"Records: {report['records']}; speakers: {len(report['speakers'])}.",
        "",
        "| Metric | V9 | V15 | Signed improvement | 95% speaker bootstrap CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in report["speaker_macro"].items():
        ci = values["bootstrap_95ci"]
        lines.append(
            f"| {labels[metric]} | {values['v9']:.3f} | {values['v15']:.3f} | "
            f"{values['signed_improvement']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines.extend([
        "",
        "## Table 2 Context",
        "",
        "| Method | ROUGE | BERTScore | Reflectiveness | Grounding | Sentiment | Emotion | Intimacy AD | Empathy AD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method, row in report["paper_table2"].items():
        lines.append(
            f"| Paper {method} | "
            + " | ".join(row[key] for key in paper_keys)
            + " |"
        )
    for method, side in (("V9 matched gate", "v9"), ("V15 matched gate", "v15")):
        lines.append(
            f"| {method} | "
            + " | ".join(f"{report['speaker_macro'][key][side]:.3f}" for key in table_keys)
            + " |"
        )
    lines.extend([
        "",
        f"Gate passed: {report['gate_decision']['passed']}",
        "",
        "Positive signed improvement always favors V15; AD metrics are direction-reversed.",
        "This is a protocol-aligned exploratory comparison.",
        "Paper rows are the published full-test mean +/- population standard deviation; V9/V15 rows are matched gate speaker-macro means.",
    ])
    return "\n".join(lines) + "\n"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-local", type=Path, required=True)
    parser.add_argument("--v15-local", type=Path, required=True)
    parser.add_argument("--v9-gpt", type=Path, required=True)
    parser.add_argument("--v15-gpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_paired_report(
        v9_local=args.v9_local,
        v15_local=args.v15_local,
        v9_gpt=args.v9_gpt,
        v15_gpt=args.v15_gpt,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["gate_decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

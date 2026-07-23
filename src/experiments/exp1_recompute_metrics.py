"""Recompute Exp1 metrics from saved results without model or API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .exp1_metrics import build_metric_records
from .exp1_user_understanding import METHODS, aggregate_results


def load_results(path: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            results.append(value)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute Exp1 summary metrics from results.jsonl"
    )
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to recomputed_metrics.json beside the input file",
    )
    parser.add_argument(
        "--records-output",
        type=Path,
        help="Defaults to recomputed_metric_records.jsonl beside the input file",
    )
    args = parser.parse_args()

    output = args.output or args.results.with_name("recomputed_metrics.json")
    records_output = args.records_output or args.results.with_name(
        "recomputed_metric_records.jsonl"
    )
    results = load_results(args.results)
    summary = aggregate_results(results)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metric_records = build_metric_records(results, METHODS)
    records_output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in metric_records
        ),
        encoding="utf-8",
    )
    print(f"[Exp1 metrics] wrote {output}")
    print(f"[Exp1 metrics] wrote {records_output}")


if __name__ == "__main__":
    main()

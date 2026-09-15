"""Frozen hard-window diagnostic. No data selection by candidate score or restart loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import realtalk_behavior_calibrated_v3 as v3
from src.experiments import realtalk_gpt_judge as judge
from src.experiments.realtalk_local_metrics import run as local_metrics

BASE = Path("/amax/xidian_ty/Ly/personaemp-exp2")
V9 = BASE / "runs/realtalk-ours-v9-full519-evidencefix-flash-5927bbf/predictions.jsonl"
V9_JUDGE = BASE / "runs/realtalk-ours-v9-full519-judge-resume-v1"
HARD = BASE / "runs/realtalk-ours-v14-12-v9-worst-contiguous60-v1-8683497/predictions.jsonl"
V9_SHA = "ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")


def build_selection(dataset):
    if sha(V9) != V9_SHA:
        raise ValueError("canonical V9 hash mismatch")
    old = read_rows(V9)
    original = {(r["speaker"].casefold(), r["target_turn_id"]): r for r in old}
    prepared, manifest = v3.base.prepare_formal_cb(dataset)
    mapping = {}
    for item in prepared:
        for point in item["points"]:
            key = (item["speaker"].casefold(), point["target"]["turn_id"])
            row = original[key]
            if row["ground_truth"] != point["target_message"] or row["context_hash"] != point["history_hash"]:
                raise ValueError(f"V9 target/history mismatch: {key}")
            mapping[key] = point["result_id"]
    if len(mapping) != 519 or len(original) != 519:
        raise ValueError("expected 519 unique canonical targets")
    hard = read_rows(HARD)
    ids = [mapping[(r["speaker"].casefold(), r["target_turn_id"])] for r in hard]
    if len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError("hard window must have 60 unique targets")
    # Check contiguity in source order, not lexical sorting of turn numbers.
    vanessa = next(item for item in prepared if item["speaker"].casefold() == "vanessa")
    ordered = [p["result_id"] for p in vanessa["points"]]
    positions = [ordered.index(rid) for rid in ids]
    if positions != list(range(positions[0], positions[0] + 60)):
        raise ValueError("hard window is not source-contiguous")
    return ids, original, {"all519_history_and_gt_match": True, "dataset": manifest,
        "v9_sha256": sha(V9), "hard_source_sha256": sha(HARD), "full_window": ids,
        "preflight_indices": [0, 7, 23, 31, 32, 59],
        "selection_scope": "retrospective Vanessa contiguous difficulty diagnostic, not independent test"}


def evaluate(predictions, output, original, dataset, *, local=False):
    rows = read_rows(predictions)
    baseline = []
    candidate = []
    for row in rows:
        old = original[(row["speaker"].casefold(), row["target_turn_id"])]
        if row["ground_truth"] != old["ground_truth"] or row["history_hash"] != old["context_hash"]:
            raise ValueError("paired history or answer mismatch")
        baseline.append(old)
        candidate.append({**row, "original_result_id": row["result_id"], "result_id": old["result_id"]})
    candidate_file = output / "candidate_aligned.jsonl"
    write_rows(candidate_file, candidate)
    write_rows(output / "v9_matched.jsonl", baseline)
    wanted = {r["result_id"] for r in baseline}
    baseline_scores = [r for r in read_rows(V9_JUDGE / "scored.jsonl") if r["result_id"] in wanted]
    if len(baseline_scores) != len(candidate):
        raise ValueError("missing canonical V9 judgments")
    write_rows(output / "v9_scored.jsonl", baseline_scores)
    summary = judge.run(candidate_file, dataset, output / "judge", "gpt-4o-mini", V9_JUDGE / "checkpoint.json")
    if summary["status"] != "complete":
        raise RuntimeError("judge incomplete; stop, do not expand")
    new_scores = read_rows(output / "judge/scored.jsonl")
    old_scores = {r["result_id"]: r for r in baseline_scores}
    if any(r["reference"] != old_scores[r["result_id"]]["reference"] for r in new_scores):
        raise ValueError("reference judge labels differ")
    comparison = {k: {"v9": statistics.mean(r["metrics"][k] for r in baseline_scores),
                      "candidate": statistics.mean(r["metrics"][k] for r in new_scores)}
                  for k in baseline_scores[0]["metrics"]}
    write(output / "paired_gpt.json", comparison)
    print(json.dumps({"paired_gpt": comparison}, ensure_ascii=False), flush=True)
    if local:
        local_metrics(output / "v9_matched.jsonl", output / "v9_local")
        local_metrics(candidate_file, output / "candidate_local")
        old_local = read_rows(output / "v9_local/results_with_local_metrics.jsonl")
        new_local = read_rows(output / "candidate_local/results_with_local_metrics.jsonl")
        for k in old_local[0]["local_metrics"]:
            comparison[k] = {"v9": statistics.mean(r["local_metrics"][k] for r in old_local),
                             "candidate": statistics.mean(r["local_metrics"][k] for r in new_local)}
    write(output / "paired_all_available.json", comparison)
    lines = ["# V3.2 vs canonical V9", "", f"Retrospective Vanessa diagnostic: {len(rows)} records. Not a paper main result.", "",
             "| Metric | V9 | V3.2 | New minus old |", "|---|---:|---:|---:|"]
    for k, value in comparison.items():
        lines.append(f"| {k} | {value['v9']:.6f} | {value['candidate']:.6f} | {value['candidate'] - value['v9']:+.6f} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--count", type=int, choices=[6, 60], required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    ids, original, selection = build_selection(dataset)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("immutable selection changed")
    write(selection_path, selection)
    selected = [ids[i] for i in selection["preflight_indices"]] if args.count == 6 else ids
    ids_path = root / f"ids{args.count}.json"
    write(ids_path, selected)
    output = root / f"generation{args.count}"
    try:
        if not args.evaluate_only:
            if output.exists():
                raise ValueError("generation directory exists; no automatic rerun or overwrite")
            result = v3.run(v3.Config(dataset_dir=str(dataset), output_dir=str(output), gate=args.count,
                selected_ids_file=str(ids_path), parent_output=str(root / "generation6") if args.count == 60 else None))
            write(root / f"generation{args.count}_status.json", result)
            if result["status"] != "generation_complete":
                raise RuntimeError("generation unresolved; stop without expanding")
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["status"] != "generation_complete" or manifest["unresolved_count"]:
            raise ValueError("only complete generation may enter paired evaluation")
        evaluate(output / "predictions.jsonl", root / f"evaluation{args.count}", original, dataset, local=args.count == 60)
        write(root / f"stage{args.count}_status.json", {"status": "complete", "count": args.count})
    except Exception as exc:
        write(root / f"stage{args.count}_status.json", {"status": "stopped", "type": type(exc).__name__, "error": str(exc)[:500]})
        raise


if __name__ == "__main__":
    main()

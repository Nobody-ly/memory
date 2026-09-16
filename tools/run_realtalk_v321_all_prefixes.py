"""Complete all ten fixed 20-target prefixes without rerunning the first three."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_realtalk_v32_diagnostic import BASE, build_selection, evaluate, read_rows, sha, v3, write, write_rows
from run_realtalk_v321_other_windows import FROZEN_RUN, freeze_json, paired_report

PREVIOUS = BASE / "runs/realtalk-v321-other3-prefix60-0f1625c"
SCOPE = "All ten speakers, first 20 causal targets each; exploratory early-prefix comparison, not full Table 2"
PAIRED_FILES = ("v9_scored.jsonl", "judge/scored.jsonl",
                "v9_local/results_with_local_metrics.jsonl", "candidate_local/results_with_local_metrics.jsonl",
                "v9_matched.jsonl", "candidate_aligned.jsonl")


def all_prefixes(prepared):
    windows = {item["speaker"]: [p["result_id"] for p in item["points"][:20]] for item in prepared}
    ids = [rid for window in windows.values() for rid in window]
    if len(windows) != 10 or any(len(w) != 20 for w in windows.values()) or len(set(ids)) != 200:
        raise ValueError("expected ten nonduplicated 20-target source prefixes")
    return windows


def require_frozen(manifest, frozen):
    for field in ("protocol", "model", "prompt_hashes", "schema_hashes", "implementation_hashes", "decoding"):
        if manifest[field] != frozen[field]:
            raise ValueError(f"frozen generation mismatch: {field}")


def validate_merged(rows, expected):
    ids = [r["result_id"] for r in rows]
    if len(ids) != 200 or len(set(ids)) != 200 or set(ids) != expected:
        raise ValueError("missing, duplicated or unexpected merged IDs")
    if sorted(Counter(r["speaker"] for r in rows).values()) != [20] * 10:
        raise ValueError("merged speaker counts are not ten times twenty")


def merge_evaluations(directories, output, expected, speakers):
    provenance = []
    for name in PAIRED_FILES:
        rows = []
        for directory in directories:
            path = directory / name
            rows.extend(read_rows(path))
            provenance.append({"path": str(path), "sha256": sha(path)})
        validate_merged(rows, expected)
        write_rows(output / name, rows)
    write(output / "source_files.json", provenance)
    paired_report(output, speakers=speakers, count=200, scope=SCOPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    root, dataset = args.output_root.resolve(), args.dataset.resolve()
    hard_ids, original, source = build_selection(dataset)
    prepared, data_manifest = v3.base.prepare_formal_cb(dataset)
    windows = all_prefixes(prepared)
    selected = [rid for window in windows.values() for rid in window]
    if set(selected) & set(hard_ids):
        raise ValueError("unexpected overlap with retrospective Vanessa hard window")
    frozen = json.loads((FROZEN_RUN / "manifest.json").read_text())
    require_frozen(v3._manifest(data_manifest, selected, 60), frozen)
    prior_manifest = json.loads((PREVIOUS / "generation60/manifest.json").read_text())
    require_frozen(prior_manifest, frozen)
    if json.loads((PREVIOUS / "stage_status.json").read_text())["status"] != "complete":
        raise ValueError("previous generation/evaluation not complete")
    prior = read_rows(PREVIOUS / "generation60/predictions.jsonl")
    prior_speakers = {r["speaker"] for r in prior}
    wanted_prior = {rid for speaker in prior_speakers for rid in windows[speaker]}
    if len(prior) != 60 or {r["result_id"] for r in prior} != wanted_prior:
        raise ValueError("previous 60 are not exact three-person prefixes")
    expected = {original[(item["speaker"].casefold(), p["target"]["turn_id"])]["result_id"]
                for item in prepared for p in item["points"][:20]}
    selection = {"scope": SCOPE, "windows": windows, "ids": selected, "source": source,
                 "reuse_prediction_count": 60, "new_prediction_count": 140,
                 "prior_predictions_sha256": sha(PREVIOUS / "generation60/predictions.jsonl"),
                 "prior_judge_sha256": sha(PREVIOUS / "evaluation60/judge/checkpoint.json"),
                 "frozen_manifest_sha256": sha(FROZEN_RUN / "manifest.json"),
                 "selected_without_scores": True, "hard_window_included": False,
                 "vanessa_self_domain": "new same-configuration realization; hard-window profiles not reused",
                 "gate_config": "legacy gate=60 compatibility selector; actual per-job selected IDs=20"}
    freeze_json(root / "selection.json", selection)
    evaluations, statuses = [PREVIOUS / "evaluation60"], {}
    for speaker, ids in windows.items():
        if speaker in prior_speakers:
            statuses[speaker] = {"status": "reused", "count": 20}
            continue
        person = root / v3.base._safe_id(speaker)
        ids_file = person / "ids20.json"
        freeze_json(ids_file, ids)
        generation, evaluation = person / "generation20", person / "evaluation20"
        try:
            if not generation.exists():
                result = v3.run(v3.Config(dataset_dir=str(dataset), output_dir=str(generation), gate=60,
                                         selected_ids_file=str(ids_file)))
                write(person / "generation_status.json", result)
            manifest = json.loads((generation / "manifest.json").read_text())
            require_frozen(manifest, frozen)
            if manifest["status"] != "generation_complete" or manifest["unresolved_count"]:
                raise ValueError("unresolved generation; no automatic retry beyond fixed limits")
            rows = read_rows(generation / "predictions.jsonl")
            if len(rows) != 20 or {r["result_id"] for r in rows} != set(ids):
                raise ValueError("generated prefix ID mismatch")
            # Completed jobs are reused unchanged if the orchestration is resumed.
            if not (person / "COMPLETE.json").exists():
                evaluate(generation / "predictions.jsonl", evaluation, original, dataset, local=True,
                         scope=f"Frozen {speaker} first 20 targets")
                write(person / "COMPLETE.json", {"status": "complete", "count": 20,
                       "predictions_sha256": sha(generation / "predictions.jsonl")})
            evaluations.append(evaluation)
            statuses[speaker] = {"status": "complete", "count": 20}
        except Exception as exc:
            statuses[speaker] = {"status": "stopped", "error": str(exc)[:500], "type": type(exc).__name__}
        write(root / "progress.json", statuses)
        print(json.dumps({"speaker": speaker, **statuses[speaker]}, ensure_ascii=False), flush=True)
    if any(x["status"] == "stopped" for x in statuses.values()):
        write(root / "stage_status.json", {"status": "incomplete", "speakers": statuses,
              "note": "No all-person aggregate: failed people are not silently omitted"})
        raise RuntimeError("at least one speaker incomplete; preserved all results without a partial main table")
    merge_evaluations(evaluations, root / "evaluation200", expected, list(windows))
    write(root / "stage_status.json", {"status": "complete", "count": 200, "new_count": 140, "reused_count": 60})


if __name__ == "__main__":
    main()

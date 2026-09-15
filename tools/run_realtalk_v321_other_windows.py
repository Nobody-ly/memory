"""Frozen cross-person continuation: first 20 Cb targets of three fixed speakers."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from run_realtalk_v32_diagnostic import (
    BASE, build_selection, evaluate, read_rows, sha, v3, write,
)

SPEAKERS = ("Emi", "Nicolas", "Kevin")
FROZEN_RUN = BASE / "runs/realtalk-v321-hard60-4302989/generation60"
SCOPE = "Three-person sequential-prefix diagnostic; not an independent confirmation or paper main result"


def choose_windows(prepared):
    by_name = {item["speaker"]: item for item in prepared}
    chosen, windows = [], []
    for speaker in SPEAKERS:
        points = by_name[speaker]["points"]
        if len(points) < 20:
            raise ValueError(f"not enough source targets: {speaker}")
        window = [point["result_id"] for point in points[:20]]
        chosen.extend(window)
        windows.append({"speaker": speaker, "source_start": 0, "count": 20,
                        "first_id": window[0], "last_id": window[-1],
                        "sessions": {s: sum(p["target_session"] == s for p in points[:20])
                                     for s in sorted({p["target_session"] for p in points[:20]})}})
    if len(chosen) != 60 or len(set(chosen)) != 60:
        raise ValueError("expected 60 unique IDs")
    return chosen, windows


def freeze_json(path, value):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"immutable file mismatch: {path.name}")
    write(path, value)


def summarize(rows, field):
    speakers = sorted({row["speaker"] for row in rows})
    per = {speaker: {key: statistics.mean(r[field][key] for r in rows if r["speaker"] == speaker)
                     for key in rows[0][field]} for speaker in speakers}
    macro = {key: statistics.mean(per[speaker][key] for speaker in speakers) for key in rows[0][field]}
    return per, macro


def paired_report(output):
    details, summary = {s: {} for s in SPEAKERS}, {}
    for field, old_file, new_file in (
        ("metrics", "v9_scored.jsonl", "judge/scored.jsonl"),
        ("local_metrics", "v9_local/results_with_local_metrics.jsonl", "candidate_local/results_with_local_metrics.jsonl"),
    ):
        old, new = read_rows(output / old_file), read_rows(output / new_file)
        if {r["result_id"] for r in old} != {r["result_id"] for r in new} or len(old) != 60 or len(new) != 60:
            raise ValueError("paired scoring IDs/count mismatch")
        # Judge rows retain speaker identity in the standard evaluation output.
        old_per, old_macro = summarize(old, field)
        new_per, new_macro = summarize(new, field)
        for key in old_macro:
            summary[key] = {"v9": old_macro[key], "candidate": new_macro[key],
                            "delta": new_macro[key] - old_macro[key]}
            for speaker in SPEAKERS:
                details[speaker][key] = {"v9": old_per[speaker][key], "candidate": new_per[speaker][key],
                                        "delta": new_per[speaker][key] - old_per[speaker][key]}
    write(output / "speaker_macro_comparison.json", {"scope": SCOPE, "macro": summary, "per_speaker": details})
    lines = ["# V3.2.1 Other-Person Sequential Windows", "", SCOPE,
             "", "Emi, Nicolas, Kevin: first 20 target messages each. Full causal history; frozen generation code.",
             "Speaker macro mean; AD lower is better. Historical V9 prediction labels, shared reference labels."]
    for title, values in [("Macro (60)", summary), *details.items()]:
        lines += ["", f"## {title}", "", "| Metric | V9 | V3.2.1 | Delta |", "|---|---:|---:|---:|"]
        lines += [f"| {k} | {v['v9']:.6f} | {v['candidate']:.6f} | {v['delta']:+.6f} |" for k, v in values.items()]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    dataset, root = args.dataset.resolve(), args.output_root.resolve()
    hard_ids, original, source = build_selection(dataset)
    prepared, dataset_manifest = v3.base.prepare_formal_cb(dataset)
    selected, windows = choose_windows(prepared)
    if set(selected) & set(hard_ids):
        raise ValueError("overlap with previous Vanessa window")
    frozen = json.loads((FROZEN_RUN / "manifest.json").read_text())
    current = v3._manifest(dataset_manifest, selected, 60)
    for field in ("protocol", "model", "prompt_hashes", "schema_hashes", "implementation_hashes", "decoding"):
        if current[field] != frozen[field]:
            raise ValueError(f"frozen generation identity changed: {field}")
    freeze_json(root / "selection.json", {"scope": SCOPE, "windows": windows, "ids": selected,
                "selected_ids_sha256": v3._hash(selected), "source": source,
                "frozen_manifest_sha256": sha(FROZEN_RUN / "manifest.json"),
                "selected_without_scores": True, "overlap_with_v321_vanessa": 0})
    ids_file = root / "ids60.json"
    freeze_json(ids_file, selected)
    output = root / "generation60"
    try:
        if not args.evaluate_only:
            if output.exists():
                raise ValueError("generation directory exists; refusing automatic regeneration")
            result = v3.run(v3.Config(dataset_dir=str(dataset), output_dir=str(output), gate=60,
                                     selected_ids_file=str(ids_file)))
            write(root / "generation_status.json", result)
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["status"] != "generation_complete" or manifest["unresolved_count"]:
            raise ValueError("generation incomplete; stop without expansion")
        evaluate(output / "predictions.jsonl", root / "evaluation60", original, dataset, local=True, scope=SCOPE)
        paired_report(root / "evaluation60")
        write(root / "stage_status.json", {"status": "complete", "count": 60})
    except Exception as exc:
        write(root / "stage_status.json", {"status": "stopped", "error": str(exc)[:500], "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()

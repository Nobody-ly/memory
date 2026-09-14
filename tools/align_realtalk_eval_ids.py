"""Align REALTALK prediction IDs for paired evaluation without changing source runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_rows = rows(args.reference)
    candidate_rows = rows(args.candidate)
    reference_by_turn = {
        (row["speaker"].casefold(), row["target_turn_id"]): row
        for row in reference_rows
    }
    if len(reference_by_turn) != len(reference_rows):
        raise ValueError("reference contains duplicate speaker/target_turn_id pairs")

    aligned: list[dict] = []
    mapping: list[dict] = []
    for row in candidate_rows:
        key = (row["speaker"].casefold(), row["target_turn_id"])
        reference = reference_by_turn.get(key)
        if reference is None:
            raise ValueError(f"reference is missing {key}")
        if row["ground_truth"] != reference["ground_truth"]:
            raise ValueError(f"ground truth mismatch for {key}")
        aligned_row = dict(row)
        aligned_row["result_id"] = reference["result_id"]
        aligned.append(aligned_row)
        mapping.append({"candidate_result_id": row["result_id"], "reference_result_id": reference["result_id"], "speaker": row["speaker"], "target_turn_id": row["target_turn_id"]})

    ids = [row["result_id"] for row in aligned]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate maps to duplicate reference IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in aligned), encoding="utf-8")
    map_path = args.output.with_name(args.output.stem + "_id_map.json")
    map_path.write_text(json.dumps({"count": len(mapping), "mapping": mapping}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(mapping), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "id_map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

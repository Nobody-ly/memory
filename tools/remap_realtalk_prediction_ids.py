"""Align REALTALK prediction IDs by canonical test-chat and target location."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def location(row: dict) -> tuple[str, str, str]:
    return (
        row.get("test_chat") or row.get("current_file") or "",
        row["target_session"],
        row["target_turn_id"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = rows(Path(args.baseline))
    candidate = rows(Path(args.candidate))
    index = {location(row): row for row in baseline}
    if len(index) != len(baseline):
        raise ValueError("baseline contains duplicate canonical locations")
    mapped: list[dict] = []
    seen: set[str] = set()
    for row in candidate:
        key = location(row)
        reference = index.get(key)
        if reference is None:
            raise ValueError(f"candidate location is absent from baseline: {key}")
        if row.get("ground_truth") != reference.get("ground_truth"):
            raise ValueError(f"ground truth mismatch at {key}")
        canonical_id = reference["result_id"]
        if canonical_id in seen:
            raise ValueError(f"candidate maps to duplicate baseline ID: {canonical_id}")
        seen.add(canonical_id)
        value = dict(row)
        value["v2_result_id"] = row["result_id"]
        value["result_id"] = canonical_id
        mapped.append(value)
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mapped),
        encoding="utf-8",
    )
    print(json.dumps({"candidate_rows": len(candidate), "mapped_rows": len(mapped)}))


if __name__ == "__main__":
    main()

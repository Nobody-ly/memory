"""Replay only REALTALK Actor calls selected by an existing Decision trace."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .exp1_protocol import protocol_turns, stable_hash
from .operation_checkpoint import OperationCheckpoint
from .realtalk_ours import (
    EXPECTED_MODEL,
    GENERATION_SYSTEM_TEMPLATE,
    GENERATION_USER_TEMPLATE,
    _action_contract,
    _backend_from_env,
    _behavioral_self_domain,
    _target_spoke_in_session,
    _text_call,
    _turns_with_session_boundaries,
)


def run(source_dir: Path, dataset_dir: Path, output_dir: Path) -> dict:
    rows = [
        json.loads(line)
        for line in (source_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    self_domains = json.loads((source_dir / "self_domains.json").read_text(encoding="utf-8"))
    backend = _backend_from_env()
    if backend.model != EXPECTED_MODEL:
        raise ValueError(f"actor replay requires {EXPECTED_MODEL}, got {backend.model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = stable_hash({
        "protocol": "realtalk_actor_continuation_replay_v1",
        "source_predictions_sha256": _sha256(source_dir / "predictions.jsonl"),
        "self_domains_sha256": _sha256(source_dir / "self_domains.json"),
        "generation_system": stable_hash(GENERATION_SYSTEM_TEMPLATE),
        "generation_user": stable_hash(GENERATION_USER_TEMPLATE),
        "model": backend.model,
    })
    checkpoint = OperationCheckpoint(output_dir / "checkpoint.json", signature)
    raw_audit = output_dir / "raw_responses.jsonl"
    chat_cache: dict[str, dict[str, dict]] = {}
    replayed = 0
    output_rows = []
    for source in rows:
        row = dict(source)
        if row["next_action"]["continuation_move"] == "reciprocal-question":
            test_chat = row["test_chat"]
            if test_chat not in chat_cache:
                chat = json.loads((dataset_dir / test_chat).read_text(encoding="utf-8"))
                chat_cache[test_chat] = {
                    turn["turn_id"]: turn
                    for turn in protocol_turns(chat, merge_adjacent_bubbles=True)
                }
            by_id = chat_cache[test_chat]
            context = [by_id[turn_id] for turn_id in row["context_turn_ids"]]
            envelope = _text_call(
                checkpoint=checkpoint,
                backend=backend,
                operation_key=f"actor_replay:{row['result_id']}",
                system_prompt=GENERATION_SYSTEM_TEMPLATE.format(speaker=row["speaker"]),
                user_prompt=GENERATION_USER_TEMPLATE.format(
                    history=_turns_with_session_boundaries(context),
                    current_session=row["target_session"],
                    target_spoke_in_current_session=_target_spoke_in_session(
                        context, row["speaker"], row["target_session"]
                    ),
                    behavioral_self_domain=json.dumps(
                        _behavioral_self_domain(self_domains[row["speaker"]]),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    situation=json.dumps(row["situation"], ensure_ascii=False, indent=2),
                    next_action=json.dumps(row["next_action"], ensure_ascii=False, indent=2),
                    action_contract=_action_contract(
                        row["next_action"]["primary_move"],
                        row["next_action"]["continuation_move"],
                    ),
                ),
                speaker=row["speaker"],
                max_attempts=3,
                raw_audit=raw_audit,
                enable_thinking=False,
            )
            row["generated_message"] = envelope["data"]
            row["actor_replay_audit"] = envelope["audit"]
            replayed += 1
        output_rows.append(row)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "complete",
        "protocol": "realtalk_actor_continuation_replay_v1",
        "source_dir": str(source_dir.resolve()),
        "source_predictions_sha256": _sha256(source_dir / "predictions.jsonl"),
        "output_predictions_sha256": _sha256(output_dir / "predictions.jsonl"),
        "records": len(output_rows),
        "replayed_records": replayed,
        "preserved_records": len(output_rows) - replayed,
        "selection": "next_action.continuation_move == reciprocal-question",
        "decision_and_domains_regenerated": False,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "run_signature": signature,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.dataset_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

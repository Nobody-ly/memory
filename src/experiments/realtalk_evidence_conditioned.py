"""Evidence-conditioned, from-zero Ours implementation for REALTALK Task 1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .exp1_protocol import (
    REALTALK_PERSONA_SPLITS,
    build_message_level_points,
    build_profile_corpus,
    canonical_speaker,
    message_speakers,
    protocol_turns,
    session_keys,
    stable_hash,
)
from .operation_checkpoint import OperationCheckpoint
from .personaemp.client import ChatBackend, ChatResult
from .realtalk_evidence_schemas import (
    DECISION_SCHEMA,
    SELF_DOMAIN_SCHEMA,
    USER_DOMAIN_SCHEMA,
    empty_user_domain,
    normalize_decision,
    normalize_self_domain,
    normalize_user_domain,
    validate_evidence_ids,
    validate_policy_evidence,
)
from .realtalk_ours import _backend_from_env, _structured_call


PROTOCOL = "realtalk_task1_ours_evidence_conditioned_v1_6"
MODEL = "deepseek-v4-flash"
OFFICIAL_REALTALK_COMMIT = "b903e06a9770bf4e5fe9018c3e132889666d3b4a"
EXPECTED_RAW_MESSAGES = 8944
EXPECTED_SESSIONS = 219
EXPECTED_FORMAL_TARGETS = 519
EXPECTED_FORMAL_RAW_TARGET_BUBBLES = 1076
DEV_FILES = (
    ("Chat_4_Emi_Paola.json", "Emi", "Paola"),
    ("Chat_4_Emi_Paola.json", "Paola", "Emi"),
    ("Chat_5_Nicolas_Nebraas.json", "Nicolas", "Nebraas"),
    ("Chat_5_Nicolas_Nebraas.json", "Nebraas", "Nicolas"),
    ("Chat_10_Fahim_Muhhamed.json", "Fahim Khan", "Muhhamed"),
    ("Chat_10_Fahim_Muhhamed.json", "Muhhamed", "Fahim Khan"),
)
DEV_GATES = (6, 24, 68)
FORMAL_GATES = (10, 30, 90, 519)


SELF_SYSTEM_PROMPT = """You compile a private Self Domain for persona simulation.
Describe the target speaker as observed in the supplied conversation, not as an ideal assistant.
Separate the target's own claims from the partner's words. Record identity claims as sourced
self-descriptions, not independently verified truth. Describe voice and interaction tendencies only when
the target's messages support them. A tendency observed with this one partner is context-bound unless the
evidence itself supports a broader claim. Leave unsupported traits out and preserve uncertainty.
Keep this as a compact transferable model, not a transcript inventory. Consolidate related evidence and
prioritize identity/background that remains useful beyond one moment: at most 12 self claims, 6 voice
observations, 6 social dispositions, and 8 uncertainties. Put one-off activities in temporal scope rather
than expanding every conversational detail into its own claim. Cite at most four representative source IDs
per item; evidence lists are not exhaustive indexes.
Return only the strict JSON schema. Do not draft a reply or invent example quotations."""

SELF_USER_TEMPLATE = """TARGET SPEAKER: {speaker}
SOURCE PARTNER: {partner}
REFERENCE SCOPE: {reference_scope}

COMPLETE OBSERVED CONVERSATION WITH SOURCE IDS:
{reference_history}

Build the target speaker's evidence-grounded Self Domain. Every evidence_ids entry must be copied from
the target-speaker evidence whitelist below.

TARGET-SPEAKER EVIDENCE ID WHITELIST:
{allowed_evidence_ids}"""

USER_SYSTEM_PROMPT = """You update a private five-layer model of the current conversation partner.
The fixed layers are core, regulation, cognition, identity, and behavior. Store only relatively durable
evidence that can help future interaction. Current mood and one-off events belong to online scene
understanding, not the stable profile. Empty layers are valid. Do not fill the structure with unsupported
psychology. Facts about the target speaker must never become partner facts. Revise or withdraw old facts
when new evidence conflicts. Return only the strict JSON schema."""

USER_USER_TEMPLATE = """TARGET SPEAKER: {speaker}
PARTNER BEING MODELED: {partner}

PREVIOUS FIVE-LAYER PARTNER MODEL:
{previous_domain}

NEWLY COMPLETED CONVERSATION SESSION WITH SOURCE IDS:
{completed_session}

CUMULATIVE PARTNER EVIDENCE ID WHITELIST:
{allowed_evidence_ids}

Return the updated complete five-layer partner model. Every evidence ID must be copied from the
whitelist. Preserve well-supported prior facts when they remain consistent."""

DECISION_SYSTEM_PROMPT = """You are the private situation and alignment controller for persona simulation.
This is a next-utterance prediction task: infer what the target person would most naturally say now,
not what would improve the relationship, maximize empathy, or satisfy an evaluation metric. Use the
private Self Domain as the default identity and voice. Use the five-layer User Domain only when a relevant
partner fact genuinely changes the current response. Read the complete real history before the target
turn and distinguish the partner's move, the target's conversational obligation, and uncertain details.

A direct question normally needs a direct answer. Do not add a follow-up question by default. Do not force
self-disclosure, reflection, emotion labeling, praise, advice, warmth, or a multi-part response. The target
may answer, react, share, ask, close, or say nothing depending on the actual exchange and observed person.
The lambda_trace is an auditable soft tradeoff between the target's default identity and adaptation to the
current exchange. It is not a score, quota, reward, or formula that mechanically changes the text. Its
orientation and affected_dimensions must be reflected in response_guidance. Return only the strict JSON
schema; never draft the final message or reconstruct the reference answer."""

DECISION_FORMAL_TEMPLATE = """TARGET SPEAKER: {speaker}
CURRENT PARTNER: {partner}

REFERENCE CONVERSATION WITH {reference_partner} ({reference_scope}):
{reference_history}

PRIVATE SELF DOMAIN:
{self_domain}

CURRENT FIVE-LAYER PARTNER MODEL:
{user_domain}

CURRENT CONVERSATION TO CONTINUE (REAL HISTORY BEFORE THE TARGET TURN):
{current_history}

CURRENT CONVERSATION POSITION (KNOWN BEFORE THE TARGET TEXT):
{conversation_position}

VISIBLE SOURCE ID WHITELIST FOR POLICY EVIDENCE:
{visible_evidence_ids}

DETERMINISTIC TARGET-SPEAKER BEHAVIOR STATISTICS FROM THE REFERENCE SCOPE:
{behavior_statistics}

Infer the situation, alignment and soft response guidance for {speaker}. Evidence IDs may be empty;
otherwise copy only visible IDs. Answer a direct partner question before any optional continuation."""

DECISION_DEV_TEMPLATE = """TARGET SPEAKER: {speaker}
CURRENT PARTNER: {partner}
REFERENCE SCOPE WITHIN THIS HISTORY: {reference_scope}
CURRENT HISTORY SCOPE: all text below, ending strictly before the target turn

COMPLETE OBSERVED CONVERSATION, PROVIDED ONCE:
{current_history}

CURRENT CONVERSATION POSITION (KNOWN BEFORE THE TARGET TEXT):
{conversation_position}

PRIVATE SELF DOMAIN COMPILED FROM THE REFERENCE SCOPE:
{self_domain}

CURRENT FIVE-LAYER PARTNER MODEL:
{user_domain}

VISIBLE SOURCE ID WHITELIST FOR POLICY EVIDENCE:
{visible_evidence_ids}

DETERMINISTIC TARGET-SPEAKER BEHAVIOR STATISTICS FROM THE REFERENCE SCOPE:
{behavior_statistics}

Infer the situation, alignment and soft response guidance for {speaker}. Evidence IDs may be empty;
otherwise copy only visible IDs. Answer a direct partner question before any optional continuation."""

ACTOR_SYSTEM_TEMPLATE = """You are {speaker}. Continue the conversation.
Act as the person represented by the private Self Domain and use the private response guidance softly.
Write the most likely natural message in the person's observed voice. Answer a direct question when the
situation requires it. Do not force disclosure, reflection, emotion, praise, advice, or warmth. Follow
question guidance, but do not manufacture a question when none is warranted. One or more natural message
bubbles are allowed; there is no fixed bubble count or fixed length. Output only the message, not the
speaker name, JSON, internal reasoning, or private guidance."""

ACTOR_FORMAL_TEMPLATE = """REFERENCE CONVERSATION WITH {reference_partner} ({reference_scope}):
{reference_history}

PRIVATE SELF DOMAIN:
{self_domain}

CURRENT CONVERSATION TO CONTINUE (REAL HISTORY BEFORE YOUR NEXT TURN):
{current_history}

CURRENT CONVERSATION POSITION (KNOWN BEFORE YOUR NEXT TEXT):
{conversation_position}

PRIVATE RESPONSE GUIDANCE:
{response_guidance}

Use this as soft guidance rather than a script. Do not include JSON, speaker labels, or internal reasoning.

Continue naturally as {speaker}."""

ACTOR_DEV_TEMPLATE = """REFERENCE SCOPE WITHIN THIS HISTORY: {reference_scope}
CURRENT HISTORY SCOPE: all text below, ending strictly before your next turn

COMPLETE OBSERVED CONVERSATION, PROVIDED ONCE:
{current_history}

CURRENT CONVERSATION POSITION (KNOWN BEFORE YOUR NEXT TEXT):
{conversation_position}

PRIVATE SELF DOMAIN COMPILED FROM THE REFERENCE SCOPE:
{self_domain}

PRIVATE RESPONSE GUIDANCE:
{response_guidance}

Use this as soft guidance rather than a script. Do not include JSON, speaker labels, or internal reasoning.

Continue naturally as {speaker}."""


@dataclass(frozen=True)
class EvidenceConditionedConfig:
    dataset_dir: str = "dataset"
    output_dir: str = "data/realtalk_evidence_conditioned_dev6"
    mode: str = "ca-dev"
    gate: int = 6
    model: str = MODEL
    operation_max_attempts: int = 3
    model_call_timeout_seconds: int = 240
    fresh: bool = False
    resume: bool = False
    preflight_only: bool = False
    profile_source_dir: str | None = None
    contiguous_start: int | None = None
    contiguous_count: int | None = None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def select_result_ids(
    gate_manifests: dict[str, list[str]],
    gate: int,
    mode: str,
    contiguous_start: int | None,
    contiguous_count: int | None,
) -> list[str]:
    selected = list(gate_manifests[str(gate)])
    if contiguous_start is None and contiguous_count is None:
        return selected
    if contiguous_start is None or contiguous_count is None:
        raise ValueError("contiguous_start and contiguous_count must be set together")
    if mode != "cb" or gate != EXPECTED_FORMAL_TARGETS:
        raise ValueError("contiguous selection requires cb mode with gate 519")
    if contiguous_start < 1 or contiguous_count < 1:
        raise ValueError("contiguous selection values must be positive")
    begin = contiguous_start - 1
    end = begin + contiguous_count
    if end > len(selected):
        raise ValueError("contiguous selection exceeds the canonical result order")
    return selected[begin:end]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_profile_source(
    source_dir: Path,
    prepared: list[dict[str, Any]],
    selected_speakers: set[str],
    dataset_manifest: dict[str, Any],
    mode: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    manifest_path = source_dir / "manifest.json"
    self_path = source_dir / "self_domains.json"
    user_path = source_dir / "user_domains.json"
    for path in (manifest_path, self_path, user_path):
        if not path.is_file():
            raise ValueError(f"profile source is missing {path.name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "generation_complete":
        raise ValueError("profile source must be generation_complete")
    if manifest.get("mode") != mode or manifest.get("model") != MODEL:
        raise ValueError("profile source mode or model does not match")
    if stable_hash(manifest.get("dataset")) != stable_hash(dataset_manifest):
        raise ValueError("profile source dataset manifest does not match")

    raw_self = json.loads(self_path.read_text(encoding="utf-8"))
    raw_user = json.loads(user_path.read_text(encoding="utf-8"))
    self_domains: dict[str, dict[str, Any]] = {}
    user_domains: dict[str, dict[str, dict[str, Any]]] = {}
    items = {item["speaker"]: item for item in prepared if item["speaker"] in selected_speakers}
    for speaker in sorted(selected_speakers):
        if speaker not in raw_self or speaker not in raw_user or speaker not in items:
            raise ValueError(f"profile source is missing selected speaker {speaker!r}")
        item = items[speaker]
        self_domains[speaker] = validate_evidence_ids(
            normalize_self_domain(raw_self[speaker]),
            evidence_ids(item["reference_turns"], speaker),
        )
        allowed_partner_ids: set[str] = set()
        user_domains[speaker] = {}
        by_session = {
            session_id: [
                turn for turn in item["current_turns"]
                if turn["session_id"] == session_id
            ]
            for session_id in ("session_1", "session_2", "session_3")
        }
        for session_index, session_id in enumerate(
            ("session_1", "session_2", "session_3"), start=1
        ):
            if session_id not in raw_user[speaker]:
                raise ValueError(
                    f"profile source is missing {speaker!r} {session_id!r}"
                )
            if session_index > 1:
                completed = by_session[f"session_{session_index - 1}"]
                allowed_partner_ids |= evidence_ids(completed, item["partner"])
            user_domains[speaker][session_id] = validate_evidence_ids(
                normalize_user_domain(raw_user[speaker][session_id]),
                allowed_partner_ids,
            )

    metadata = {
        "source_protocol": manifest.get("protocol"),
        "source_run_signature": manifest.get("run_signature"),
        "self_domains_sha256": hashlib.sha256(self_path.read_bytes()).hexdigest(),
        "user_domains_sha256": hashlib.sha256(user_path.read_bytes()).hexdigest(),
        "selected_speakers": sorted(selected_speakers),
    }
    return self_domains, user_domains, metadata


def format_evidence_turns(turns: Iterable[dict[str, Any]]) -> str:
    blocks: list[str] = []
    current_session = ""
    for turn in turns:
        if turn["session_id"] != current_session:
            current_session = turn["session_id"]
            blocks.append(f"[SESSION {current_session}]")
        if "source_id" not in turn:
            raise ValueError("evidence turn is missing a globally unique source_id")
        raw_ids = ",".join(turn["dia_ids"]) or "none"
        blocks.append(
            f"[{turn['source_id']} | source_message_ids={raw_ids}] "
            f"{turn['speaker']}: {turn['content']}"
        )
    return "\n".join(blocks)


def evidence_ids(turns: Iterable[dict[str, Any]], speaker: str | None = None) -> set[str]:
    result: set[str] = set()
    for turn in turns:
        if speaker is not None and turn["speaker"].casefold() != speaker.casefold():
            continue
        if "source_id" not in turn:
            raise ValueError("evidence turn is missing a globally unique source_id")
        result.add(turn["source_id"])
    return result


def tag_source(turns: Iterable[dict[str, Any]], file_id: str) -> list[dict[str, Any]]:
    """Attach the file namespace required because D1:1-style IDs repeat by chat."""
    return [
        {
            **turn,
            "file_id": file_id,
            "source_id": f"{file_id}::{turn['turn_id']}",
            "source_message_ids": list(turn["dia_ids"]),
        }
        for turn in turns
    ]


def tag_point_source(point: dict[str, Any], file_id: str) -> dict[str, Any]:
    return {
        **point,
        "context_turns": tag_source(point["context_turns"], file_id),
        "target": tag_source([point["target"]], file_id)[0],
    }


def prepare_ca_dev(dataset_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = Path(dataset_dir)
    prepared: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for filename, requested_speaker, requested_partner in DEV_FILES:
        path = dataset / filename
        chat = json.loads(path.read_text(encoding="utf-8"))
        speaker = canonical_speaker(chat, requested_speaker)
        partner = canonical_speaker(chat, requested_partner)
        all_turns = tag_source(
            protocol_turns(chat, merge_adjacent_bubbles=True), filename
        )
        reference_sessions = session_keys(chat)[:2]
        reference_turns = [
            turn for turn in all_turns if turn["session_id"] in set(reference_sessions)
        ]
        session_three = session_keys(chat)[2]
        points = [
            tag_point_source(point, filename) for point in build_message_level_points(
                chat, speaker, test_sessions=3, max_context_chars=0,
                merge_adjacent_bubbles=True,
            ) if point["target_session"] == session_three
        ]
        for point in points:
            point["result_id"] = (
                f"ca_dev:{filename}:{speaker}:{point['target']['turn_id']}"
            )
        source_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared.append({
            "mode": "ca-dev", "speaker": speaker, "partner": partner,
            "reference_file": filename, "current_file": filename,
            "reference_scope": f"{reference_sessions[0]}..{reference_sessions[-1]}",
            "reference_sessions": reference_sessions,
            "reference_turns": reference_turns,
            "current_turns": [
                turn for turn in all_turns if turn["session_id"] in set(session_keys(chat)[:3])
            ],
            "points": points,
        })
    manifest = _dataset_manifest(dataset, source_hashes, prepared, "ca-dev")
    if sum(len(item["points"]) for item in prepared) != 68:
        raise ValueError("Ca development set must contain exactly 68 targets")
    return prepared, manifest


def prepare_formal_cb(dataset_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = Path(dataset_dir)
    prepared: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for split in REALTALK_PERSONA_SPLITS:
        reference_path = dataset / split["train_chat"]
        current_path = dataset / split["test_chat"]
        reference_chat = json.loads(reference_path.read_text(encoding="utf-8"))
        current_chat = json.loads(current_path.read_text(encoding="utf-8"))
        speaker = canonical_speaker(current_chat, split["speaker"])
        partner = next(
            value for value in message_speakers(current_chat)
            if value.casefold() != speaker.casefold()
        )
        reference_partner = next(
            value for value in message_speakers(reference_chat)
            if value.casefold() != speaker.casefold()
        )
        reference = build_profile_corpus(
            reference_chat, speaker, profile_sessions=3,
            merge_adjacent_bubbles=True,
        )
        points = [
            tag_point_source(point, split["test_chat"])
            for point in build_message_level_points(
                current_chat, speaker, test_sessions=3, max_context_chars=0,
                merge_adjacent_bubbles=True,
            )
        ]
        for point in points:
            point["result_id"] = f"cb:{speaker}:{point['target']['turn_id']}"
        current_sessions = session_keys(current_chat)[:3]
        current_turns = [
            turn for turn in tag_source(
                protocol_turns(current_chat, merge_adjacent_bubbles=True),
                split["test_chat"],
            )
            if turn["session_id"] in set(current_sessions)
        ]
        for path in (reference_path, current_path):
            source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared.append({
            "mode": "cb", "speaker": speaker, "partner": partner,
            "reference_partner": reference_partner,
            "reference_file": split["train_chat"], "current_file": split["test_chat"],
            "reference_scope": "session_1..session_3",
            "reference_sessions": reference["sessions"],
            "reference_turns": tag_source(reference["turns"], split["train_chat"]),
            "current_turns": current_turns, "points": points,
        })
    manifest = _dataset_manifest(dataset, source_hashes, prepared, "cb")
    total = sum(len(item["points"]) for item in prepared)
    raw_bubbles = sum(
        len(point["target"]["message_indices"])
        for item in prepared for point in item["points"]
    )
    if total != EXPECTED_FORMAL_TARGETS or raw_bubbles != EXPECTED_FORMAL_RAW_TARGET_BUBBLES:
        raise ValueError(
            f"formal protocol mismatch: targets={total}, raw_target_bubbles={raw_bubbles}"
        )
    manifest["formal_raw_target_bubbles"] = raw_bubbles
    return prepared, manifest


def _dataset_manifest(
    dataset: Path,
    source_hashes: dict[str, str],
    prepared: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    all_files = sorted(dataset.glob("Chat_*.json"))
    total_sessions = 0
    raw_messages = 0
    all_hashes: dict[str, str] = {}
    for path in all_files:
        chat = json.loads(path.read_text(encoding="utf-8"))
        keys = session_keys(chat)
        total_sessions += len(keys)
        raw_messages += sum(len(chat[key]) for key in keys)
        all_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(all_files) != 10 or total_sessions != EXPECTED_SESSIONS or raw_messages != EXPECTED_RAW_MESSAGES:
        raise ValueError(
            f"REALTALK source mismatch: files={len(all_files)} "
            f"sessions={total_sessions} raw_messages={raw_messages}"
        )
    return {
        "dataset": "REALTALK public preprocessed conversations",
        "official_repository_commit": OFFICIAL_REALTALK_COMMIT,
        "mode": mode,
        "source_file_count": len(all_files),
        "source_session_count": total_sessions,
        "source_raw_message_count": raw_messages,
        "all_source_files_sha256": all_hashes,
        "selected_source_files_sha256": dict(sorted(source_hashes.items())),
        "sample_unit": "merged consecutive same-speaker messages within a session",
        "merge_preserves_text_with_newlines_and_source_ids": True,
        "history_compression_enabled": False,
        "history_truncation_enabled": False,
        "event_and_qa_fields_used": False,
        "image_captioning_added": False,
        "targets": sum(len(item["points"]) for item in prepared),
    }


def build_gate_manifests(prepared: list[dict[str, Any]], mode: str) -> dict[str, list[str]]:
    if mode == "ca-dev":
        per_speaker = [item["points"] for item in prepared]
        first = [points[0]["result_id"] for points in per_speaker]
        first_four = [
            point["result_id"] for depth in range(4)
            for points in per_speaker for point in points[depth:depth + 1]
        ]
        all_ids = [point["result_id"] for item in prepared for point in item["points"]]
        manifests = {"6": first, "24": first_four, "68": all_ids}
    else:
        by_item_session: list[list[list[dict[str, Any]]]] = []
        for item in prepared:
            sessions = []
            for session_id in ("session_1", "session_2", "session_3"):
                sessions.append([
                    point for point in item["points"]
                    if point["target_session"] == session_id
                ])
            by_item_session.append(sessions)
        gate10 = [sessions[0][0]["result_id"] for sessions in by_item_session]
        gate30 = [
            session[0]["result_id"] for sessions in by_item_session for session in sessions
        ]
        gate90 = [
            point["result_id"] for sessions in by_item_session for session in sessions
            for point in session[:3]
        ]
        all_ids = [point["result_id"] for item in prepared for point in item["points"]]
        manifests = {"10": gate10, "30": gate30, "90": gate90, "519": all_ids}
    previous: set[str] = set()
    for gate in sorted((int(value) for value in manifests)):
        ids = manifests[str(gate)]
        if len(ids) != gate or len(ids) != len(set(ids)):
            raise ValueError(f"gate {gate} is not a unique {gate}-sample manifest")
        if not previous.issubset(set(ids)):
            raise ValueError(f"gate {gate} is not nested")
        previous = set(ids)
    return manifests


def build_generation_input(
    item: dict[str, Any], point: dict[str, Any], user_domain: dict[str, Any]
) -> dict[str, Any]:
    observed_in_target_session = sum(
        turn["session_id"] == point["target_session"]
        for turn in point["context_turns"]
    )
    return {
        "mode": item["mode"], "speaker": item["speaker"], "partner": item["partner"],
        "reference_partner": item.get("reference_partner", item["partner"]),
        "reference_scope": item["reference_scope"],
        "reference_turns": item["reference_turns"],
        "current_turns": point["context_turns"],
        "conversation_position": {
            "target_session": point["target_session"],
            "observed_turns_in_target_session": observed_in_target_session,
            "starts_new_session": observed_in_target_session == 0,
        },
        "reference_behavior_statistics": behavior_statistics(
            item["reference_turns"], item["speaker"]
        ),
        "user_domain": user_domain,
    }


def behavior_statistics(turns: list[dict[str, Any]], speaker: str) -> dict[str, Any]:
    """Compute a small, deterministic style prior without exposing Ca examples."""
    target = [turn for turn in turns if turn.get("speaker") == speaker]
    if not target:
        return {
            "target_message_count": 0, "mean_characters": 0.0,
            "mean_bubbles": 0.0, "multi_bubble_rate": 0.0,
            "question_rate": 0.0,
        }
    bubble_counts = [max(1, len(str(turn.get("content", "")).splitlines())) for turn in target]
    character_counts = [len(str(turn.get("content", ""))) for turn in target]
    return {
        "target_message_count": len(target),
        "mean_characters": round(sum(character_counts) / len(character_counts), 3),
        "mean_bubbles": round(sum(bubble_counts) / len(bubble_counts), 3),
        "multi_bubble_rate": round(sum(count > 1 for count in bubble_counts) / len(bubble_counts), 3),
        "question_rate": round(sum("?" in str(turn.get("content", "")) for turn in target) / len(target), 3),
    }


def decision_prompt(
    generation_input: dict[str, Any], self_domain: dict[str, Any]
) -> str:
    visible = evidence_ids(generation_input["current_turns"])
    if generation_input["mode"] == "cb":
        visible |= evidence_ids(generation_input["reference_turns"])
        return DECISION_FORMAL_TEMPLATE.format(
            speaker=generation_input["speaker"], partner=generation_input["partner"],
            reference_partner=generation_input["reference_partner"],
            reference_scope=generation_input["reference_scope"],
            reference_history=format_evidence_turns(generation_input["reference_turns"]),
            self_domain=_json(self_domain), user_domain=_json(generation_input["user_domain"]),
            current_history=format_evidence_turns(generation_input["current_turns"]),
            conversation_position=_json(generation_input["conversation_position"]),
            visible_evidence_ids=_json(sorted(visible)),
            behavior_statistics=_json(generation_input["reference_behavior_statistics"]),
        )
    return DECISION_DEV_TEMPLATE.format(
        speaker=generation_input["speaker"], partner=generation_input["partner"],
        reference_scope=generation_input["reference_scope"],
        current_history=format_evidence_turns(generation_input["current_turns"]),
        conversation_position=_json(generation_input["conversation_position"]),
        self_domain=_json(self_domain), user_domain=_json(generation_input["user_domain"]),
        visible_evidence_ids=_json(sorted(visible)),
        behavior_statistics=_json(generation_input["reference_behavior_statistics"]),
    )


def actor_prompt(
    generation_input: dict[str, Any], self_domain: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    response_guidance = {
        "situation": decision["situation"],
        "alignment": {
            "orientation": decision["alignment"]["orientation"],
            "affected_dimensions": decision["alignment"]["affected_dimensions"],
        },
        "response_guidance": decision["response_guidance"],
    }
    if generation_input["mode"] == "cb":
        return ACTOR_FORMAL_TEMPLATE.format(
            speaker=generation_input["speaker"],
            reference_partner=generation_input["reference_partner"],
            reference_scope=generation_input["reference_scope"],
            reference_history=format_evidence_turns(generation_input["reference_turns"]),
            self_domain=_json(self_domain),
            current_history=format_evidence_turns(generation_input["current_turns"]),
            conversation_position=_json(generation_input["conversation_position"]),
            response_guidance=_json(response_guidance),
        )
    return ACTOR_DEV_TEMPLATE.format(
        speaker=generation_input["speaker"],
        reference_scope=generation_input["reference_scope"],
        current_history=format_evidence_turns(generation_input["current_turns"]),
        conversation_position=_json(generation_input["conversation_position"]),
        self_domain=_json(self_domain), response_guidance=_json(response_guidance),
    )


def _actor_call(
    *, checkpoint: OperationCheckpoint, backend: ChatBackend, operation_key: str,
    speaker: str, user_prompt: str, raw_audit: Path,
) -> dict[str, Any]:
    system_prompt = ACTOR_SYSTEM_TEMPLATE.format(speaker=speaker)
    repair_feedback = {"text": ""}

    def operation() -> ChatResult:
        prompt = user_prompt
        if repair_feedback["text"]:
            prompt += (
                "\n\nCONTRACT REPAIR FROM THE PREVIOUS ATTEMPT:\n"
                + repair_feedback["text"]
                + "\nRegenerate the message while preserving the same soft response guidance."
            )
        return backend.chat(
            system_prompt, prompt, temperature=0.6, top_p=0.9,
            max_tokens=1024, enable_thinking=False,
        )

    def validate(result: ChatResult) -> dict[str, Any]:
        _append_jsonl(raw_audit, {
            "operation_key": operation_key, "model": result.model,
            "raw_response": result.content,
            "reasoning_sha256": stable_hash(result.reasoning_content),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_seconds": result.latency_seconds,
            "network_attempts": result.attempts,
            "finish_reason": result.finish_reason, "response_id": result.response_id,
            "recorded_at_utc": _now(),
        })
        try:
            message = result.content.strip()
            if not message:
                raise ValueError("actor returned empty text")
            if result.finish_reason.casefold() in {"length", "max_tokens"}:
                raise ValueError("actor output was truncated")
            if re.match(rf"^{re.escape(speaker)}\s*:", message, re.IGNORECASE):
                raise ValueError("actor leaked the speaker label")
            if message.startswith(("{", "[")) or re.search(
                r'"(?:situation|turn_plan|response_guidance|alignment|lambda_trace|self_domain)"\s*:', message, re.IGNORECASE
            ):
                raise ValueError("actor leaked private structure")
        except Exception as exc:
            repair_feedback["text"] = str(exc)
            raise
        return {
            "data": message,
            "audit": {
                "model": result.model, "logical_attempts": 1,
                "thinking_enabled": False, "finish_reason": result.finish_reason,
                "response_id": result.response_id,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
        }

    return checkpoint.execute(
        operation_key, operation, validate, 2,
        usage_supplier=lambda: dict(getattr(backend, "token_usage", {})),
    )


def _run_impl(
    config: EvidenceConditionedConfig, backend: ChatBackend | None = None
) -> dict[str, Any]:
    _validate_config(config)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.fresh:
        for name in (
            "checkpoint.json", "raw_responses.jsonl", "predictions.jsonl",
            "self_domains.json", "user_domains.json", "manifest.json",
            "unresolved_errors.jsonl", "GENERATION_COMPLETE", "preflight.json",
        ):
            (output_dir / name).unlink(missing_ok=True)
    elif config.resume and not (output_dir / "checkpoint.json").exists():
        raise ValueError("--resume requires an existing checkpoint.json")
    (output_dir / "GENERATION_COMPLETE").unlink(missing_ok=True)

    prepared, dataset_manifest = (
        prepare_ca_dev(config.dataset_dir)
        if config.mode == "ca-dev" else prepare_formal_cb(config.dataset_dir)
    )
    gate_manifests = build_gate_manifests(prepared, config.mode)
    selected_ids = select_result_ids(
        gate_manifests,
        config.gate,
        config.mode,
        config.contiguous_start,
        config.contiguous_count,
    )
    index = {
        point["result_id"]: (item, point)
        for item in prepared for point in item["points"]
    }
    selected_speakers = {index[result_id][0]["speaker"] for result_id in selected_ids}

    reused_self_domains: dict[str, dict[str, Any]] = {}
    reused_user_domains: dict[str, dict[str, dict[str, Any]]] = {}
    profile_source: dict[str, Any] | None = None
    if config.profile_source_dir:
        reused_self_domains, reused_user_domains, profile_source = _load_profile_source(
            Path(config.profile_source_dir).resolve(),
            prepared,
            selected_speakers,
            dataset_manifest,
            config.mode,
        )

    backend = backend or _backend_from_env(config.model)
    if backend.model != config.model:
        raise ValueError(f"model mismatch: expected {config.model!r}, got {backend.model!r}")
    preflight = _preflight(output_dir, backend, config.model)
    manifest_base = {
        "protocol": PROTOCOL, "mode": config.mode, "gate": config.gate,
        "model": config.model, "thinking_enabled_all_stages": False,
        "dataset": dataset_manifest, "gate_manifests": gate_manifests,
        "selected_ids_sha256": stable_hash(selected_ids),
        "selection": {
            "kind": "contiguous_canonical_window"
            if config.contiguous_start is not None else "gate",
            "canonical_one_based_start": config.contiguous_start,
            "count": config.contiguous_count,
        },
        "prompt_hashes": prompt_hashes(), "schema_hashes": schema_hashes(),
        "implementation_commit": _repository_commit(),
        "legacy_self_user_decision_reused": False,
        "self_user_domains_reused": profile_source is not None,
        "profile_source": profile_source,
        "generated_outputs_rolled_into_history": False,
        "semantic_verification_or_candidate_search": False,
        "omega_enabled": False, "future_user_state_enabled": False,
        "preflight": preflight,
    }
    signature_manifest = {
        key: value for key, value in manifest_base.items()
        if key not in {"gate", "selected_ids_sha256", "preflight"}
    }
    signature = stable_hash({
        "manifest": signature_manifest,
        "config": {
            key: value for key, value in asdict(config).items()
            if key not in {
                "output_dir", "fresh", "resume", "preflight_only", "gate",
                "profile_source_dir",
            }
        },
    })
    checkpoint = OperationCheckpoint(output_dir / "checkpoint.json", signature)
    if config.preflight_only:
        _write_json(output_dir / "manifest.json", {**manifest_base, "status": "preflight_complete"})
        return {"status": "preflight_complete", "output_dir": str(output_dir)}

    raw_audit = output_dir / "raw_responses.jsonl"
    self_domains: dict[str, dict[str, Any]] = {}
    user_domains: dict[str, dict[str, dict[str, Any]]] = {}
    for item in prepared:
        speaker = item["speaker"]
        if speaker not in selected_speakers:
            continue
        if profile_source is not None:
            self_domains[speaker] = reused_self_domains[speaker]
            user_domains[speaker] = reused_user_domains[speaker]
            continue
        target_ids = evidence_ids(item["reference_turns"], speaker)
        self_result = _structured_call(
            checkpoint=checkpoint, backend=backend,
            operation_key=f"self:{_safe_id(speaker)}",
            system_prompt=SELF_SYSTEM_PROMPT,
            user_prompt=SELF_USER_TEMPLATE.format(
                speaker=speaker, partner=item.get("reference_partner", item["partner"]),
                reference_scope=item["reference_scope"],
                reference_history=format_evidence_turns(item["reference_turns"]),
                allowed_evidence_ids=_json(sorted(target_ids)),
            ),
            schema=SELF_DOMAIN_SCHEMA,
            normalizer=lambda value, allowed=target_ids: validate_evidence_ids(
                normalize_self_domain(value), allowed
            ),
            max_tokens=4096, max_attempts=config.operation_max_attempts,
            raw_audit=raw_audit, enable_thinking=False,
            hard_timeout_seconds=config.model_call_timeout_seconds,
        )
        self_domains[speaker] = self_result["data"]

        by_session = {
            session_id: [turn for turn in item["current_turns"] if turn["session_id"] == session_id]
            for session_id in ("session_1", "session_2", "session_3")
        }
        domain = empty_user_domain()
        user_domains[speaker] = {"session_1": domain}
        allowed_partner_ids: set[str] = set()
        needed_sessions = {
            index[result_id][1]["target_session"] for result_id in selected_ids
            if index[result_id][0]["speaker"] == speaker
        }
        for previous_index, next_session in ((1, "session_2"), (2, "session_3")):
            if not any(int(session.split("_")[1]) >= int(next_session.split("_")[1]) for session in needed_sessions):
                continue
            previous_session = f"session_{previous_index}"
            completed = by_session[previous_session]
            allowed_partner_ids |= evidence_ids(completed, item["partner"])
            user_result = _structured_call(
                checkpoint=checkpoint, backend=backend,
                operation_key=f"user:{_safe_id(speaker)}:after:{previous_session}",
                system_prompt=USER_SYSTEM_PROMPT,
                user_prompt=USER_USER_TEMPLATE.format(
                    speaker=speaker, partner=item["partner"],
                    previous_domain=_json(domain),
                    completed_session=format_evidence_turns(completed),
                    allowed_evidence_ids=_json(sorted(allowed_partner_ids)),
                ),
                schema=USER_DOMAIN_SCHEMA,
                normalizer=lambda value, allowed=set(allowed_partner_ids): validate_evidence_ids(
                    normalize_user_domain(value), allowed
                ),
                max_tokens=4096, max_attempts=config.operation_max_attempts,
                raw_audit=raw_audit, enable_thinking=False,
                hard_timeout_seconds=config.model_call_timeout_seconds,
            )
            domain = user_result["data"]
            user_domains[speaker][next_session] = domain

    for result_id in selected_ids:
        if result_id in checkpoint.data["results"]:
            continue
        item, point = index[result_id]
        generation_input = build_generation_input(
            item, point, user_domains[item["speaker"]][point["target_session"]]
        )
        visible = evidence_ids(generation_input["current_turns"])
        if item["mode"] == "cb":
            visible |= evidence_ids(item["reference_turns"])
        decision_result = _structured_call(
            checkpoint=checkpoint, backend=backend,
            operation_key=f"decision:{result_id}",
            system_prompt=DECISION_SYSTEM_PROMPT,
            user_prompt=decision_prompt(generation_input, self_domains[item["speaker"]]),
            schema=DECISION_SCHEMA,
            normalizer=lambda value, allowed=visible: validate_policy_evidence(
                normalize_decision(value), allowed
            ),
            max_tokens=1536, max_attempts=config.operation_max_attempts,
            raw_audit=raw_audit, enable_thinking=False,
            hard_timeout_seconds=config.model_call_timeout_seconds,
        )
        decision = decision_result["data"]
        actor_result = _actor_call(
            checkpoint=checkpoint, backend=backend,
            operation_key=f"actor:{result_id}", speaker=item["speaker"],
            user_prompt=actor_prompt(
                generation_input, self_domains[item["speaker"]], decision
            ),
            raw_audit=raw_audit,
        )
        checkpoint.store_result(result_id, {
            "result_id": result_id, "mode": item["mode"],
            "speaker": item["speaker"], "partner": item["partner"],
            "reference_file": item["reference_file"], "current_file": item["current_file"],
            "target_session": point["target_session"],
            "target_turn_id": point["target"]["turn_id"],
            "source_message_ids": point["target"]["dia_ids"],
            "history_hash": point["history_hash"],
            "visible_history_turns": len(point["context_turns"]),
            "user_domain": generation_input["user_domain"],
            "decision": decision,
            "generated_message": actor_result["data"],
            "ground_truth": point["target_message"],
            "decision_audit": decision_result["audit"],
            "actor_audit": actor_result["audit"],
        })

    unresolved = [
        {"operation_key": key, **value}
        for key, value in sorted(checkpoint.data["failures"].items())
    ]
    results = [
        checkpoint.data["results"][result_id]
        for result_id in selected_ids if result_id in checkpoint.data["results"]
    ]
    _write_json(output_dir / "self_domains.json", self_domains)
    _write_json(output_dir / "user_domains.json", user_domains)
    _write_jsonl(output_dir / "predictions.jsonl", results)
    _write_jsonl(output_dir / "unresolved_errors.jsonl", unresolved)
    status = "generation_complete" if len(results) == len(selected_ids) and not unresolved else "incomplete"
    lambda_values = [
        result["decision"]["alignment"]["lambda_trace"]
        for result in results
    ]
    bubble_counts = [
        len([line for line in result["generated_message"].splitlines() if line.strip()])
        for result in results
    ]
    generated_question_count = sum("?" in result["generated_message"] for result in results)
    manifest = {
        **manifest_base, "status": status, "run_signature": signature,
        "selected_records": len(selected_ids), "completed_records": len(results),
        "unresolved_records": len(unresolved), "completed_at_utc": _now(),
        "token_usage": getattr(backend, "token_usage", {}),
        "runtime_behavior_summary": {
            "lambda_unique_values": sorted(set(lambda_values)),
            "lambda_min": min(lambda_values) if lambda_values else None,
            "lambda_max": max(lambda_values) if lambda_values else None,
            "lambda_mean": round(sum(lambda_values) / len(lambda_values), 6) if lambda_values else None,
            "generated_bubble_count_distribution": {
                str(count): bubble_counts.count(count) for count in sorted(set(bubble_counts))
            },
            "generated_question_rate": round(generated_question_count / len(results), 6) if results else None,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    if status == "generation_complete":
        (output_dir / "GENERATION_COMPLETE").write_text(signature + "\n", encoding="utf-8")
    return {"status": status, "output_dir": str(output_dir), "manifest": manifest}


def _record_terminal_failure(output_dir: Path, error: Exception) -> None:
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint_data: dict[str, Any] = {}
    if checkpoint_path.exists():
        checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    unresolved = [
        {"operation_key": key, **value}
        for key, value in sorted(checkpoint_data.get("failures", {}).items())
    ]
    _write_jsonl(output_dir / "unresolved_errors.jsonl", unresolved)

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "protocol": manifest.get("protocol", PROTOCOL),
        "status": "incomplete",
        "unresolved_records": len(unresolved),
        "terminal_error": {
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        },
        "completed_at_utc": _now(),
    })
    _write_json(manifest_path, manifest)
    (output_dir / "GENERATION_COMPLETE").unlink(missing_ok=True)


def run(
    config: EvidenceConditionedConfig, backend: ChatBackend | None = None
) -> dict[str, Any]:
    try:
        return _run_impl(config, backend)
    except Exception as error:
        _record_terminal_failure(Path(config.output_dir).resolve(), error)
        raise


def _preflight(output_dir: Path, backend: ChatBackend, expected_model: str) -> dict[str, Any]:
    path = output_dir / "preflight.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("expected_model") != expected_model:
            raise ValueError("cached preflight model mismatch")
        return value
    available = backend.available_models()
    if expected_model not in available:
        raise ValueError(f"model {expected_model!r} not exposed by endpoint")
    result = backend.chat(
        "Return exactly READY.", "READY", temperature=0.0, top_p=0.9,
        max_tokens=8, enable_thinking=False,
    )
    if result.content.strip() != "READY":
        raise ValueError(f"model preflight returned {result.content!r}")
    value = {
        "expected_model": expected_model, "reported_model": result.model,
        "available_model_count": len(available), "thinking_enabled": False,
        "completed_at_utc": _now(),
    }
    _write_json(path, value)
    return value


def prompt_hashes() -> dict[str, str]:
    return {
        name: stable_hash(value) for name, value in {
            "self_system": SELF_SYSTEM_PROMPT, "self_user": SELF_USER_TEMPLATE,
            "user_system": USER_SYSTEM_PROMPT, "user_user": USER_USER_TEMPLATE,
            "decision_system": DECISION_SYSTEM_PROMPT,
            "decision_formal": DECISION_FORMAL_TEMPLATE,
            "decision_dev": DECISION_DEV_TEMPLATE,
            "actor_system": ACTOR_SYSTEM_TEMPLATE,
            "actor_formal": ACTOR_FORMAL_TEMPLATE, "actor_dev": ACTOR_DEV_TEMPLATE,
        }.items()
    }


def schema_hashes() -> dict[str, str]:
    return {
        "self": stable_hash(SELF_DOMAIN_SCHEMA),
        "user": stable_hash(USER_DOMAIN_SCHEMA),
        "decision": stable_hash(DECISION_SCHEMA),
    }


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("repository commit is unavailable")
    return value


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _validate_config(config: EvidenceConditionedConfig) -> None:
    if config.mode not in {"ca-dev", "cb"}:
        raise ValueError("mode must be ca-dev or cb")
    allowed = DEV_GATES if config.mode == "ca-dev" else FORMAL_GATES
    if config.gate not in allowed:
        raise ValueError(f"gate must be one of {allowed} for mode {config.mode}")
    if config.model != MODEL:
        raise ValueError(f"model is fixed to {MODEL}")
    if config.operation_max_attempts != 3:
        raise ValueError("structured operations permit exactly three attempts")
    if config.fresh and config.resume:
        raise ValueError("--fresh and --resume are mutually exclusive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("ca-dev", "cb"), default="ca-dev")
    parser.add_argument("--gate", type=int, default=6)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--profile-source-dir")
    parser.add_argument("--contiguous-start", type=int)
    parser.add_argument("--contiguous-count", type=int)
    args = parser.parse_args()
    result = run(EvidenceConditionedConfig(
        dataset_dir=args.dataset_dir, output_dir=args.output_dir,
        mode=args.mode, gate=args.gate, model=args.model,
        fresh=args.fresh, resume=args.resume, preflight_only=args.preflight_only,
        profile_source_dir=args.profile_source_dir,
        contiguous_start=args.contiguous_start,
        contiguous_count=args.contiguous_count,
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

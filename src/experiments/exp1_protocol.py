"""REALTALK-aligned, provider-independent data protocol for Experiment 1."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .persona_simulation import flatten_messages, session_keys


# Train/test conversation assignments reported in REALTALK Table 8.
# A split is speaker-specific: the same chat may train one speaker and test
# their partner.
REALTALK_PERSONA_SPLITS: Tuple[Dict[str, str], ...] = (
    {
        "speaker": "Emi",
        "train_chat": "Chat_4_Emi_Paola.json",
        "test_chat": "Chat_1_Emi_Elise.json",
    },
    {
        "speaker": "Nicolas",
        "train_chat": "Chat_5_Nicolas_Nebraas.json",
        "test_chat": "Chat_6_Vanessa_Nicolas.json",
    },
    {
        "speaker": "Kevin",
        "train_chat": "Chat_3_Kevin_Paola.json",
        "test_chat": "Chat_2_Kevin_Elise.json",
    },
    {
        "speaker": "Akib",
        "train_chat": "Chat_9_Fahim_Akib.json",
        "test_chat": "Chat_8_Akib_Muhhamed.json",
    },
    {
        "speaker": "Muhhamed",
        "train_chat": "Chat_10_Fahim_Muhhamed.json",
        "test_chat": "Chat_8_Akib_Muhhamed.json",
    },
    {
        "speaker": "Nebraas",
        "train_chat": "Chat_5_Nicolas_Nebraas.json",
        "test_chat": "Chat_7_Nebraas_Vanessa.json",
    },
    {
        "speaker": "Paola",
        "train_chat": "Chat_4_Emi_Paola.json",
        "test_chat": "Chat_3_Kevin_Paola.json",
    },
    {
        "speaker": "Vanessa",
        "train_chat": "Chat_7_Nebraas_Vanessa.json",
        "test_chat": "Chat_6_Vanessa_Nicolas.json",
    },
    {
        "speaker": "elise",
        "train_chat": "Chat_2_Kevin_Elise.json",
        "test_chat": "Chat_1_Emi_Elise.json",
    },
    {
        "speaker": "Fahim Khan",
        "train_chat": "Chat_10_Fahim_Muhhamed.json",
        "test_chat": "Chat_9_Fahim_Akib.json",
    },
)


def merge_consecutive_utterances(chat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge adjacent bubbles from one speaker within the same session."""
    merged: List[Dict[str, Any]] = []
    session_turn_counts: Dict[str, int] = {}
    for turn in flatten_messages(chat):
        if (
            merged
            and merged[-1]["session_id"] == turn["session_id"]
            and merged[-1]["speaker"] == turn["speaker"]
        ):
            merged[-1]["content"] += "\n" + turn["content"]
            merged[-1]["message_indices"].append(turn["message_index"])
            if turn.get("dia_id"):
                merged[-1]["dia_ids"].append(turn["dia_id"])
            merged[-1]["date_time_end"] = turn.get("date_time", "")
            continue

        session_id = turn["session_id"]
        semantic_index = session_turn_counts.get(session_id, 0)
        session_turn_counts[session_id] = semantic_index + 1
        merged.append({
            "turn_id": f"{session_id}:turn_{semantic_index}",
            "session_id": session_id,
            "session_index": turn["session_index"],
            "semantic_index": semantic_index,
            "speaker": turn["speaker"],
            "content": turn["content"],
            "message_indices": [turn["message_index"]],
            "dia_ids": [turn["dia_id"]] if turn.get("dia_id") else [],
            "date_time_start": turn.get("date_time", ""),
            "date_time_end": turn.get("date_time", ""),
        })
    return merged


def message_speakers(chat: Dict[str, Any]) -> List[str]:
    """Return the two actual message speakers, preserving first appearance."""
    actual: List[str] = []
    for turn in flatten_messages(chat):
        speaker = turn["speaker"]
        if speaker and speaker.casefold() not in {name.casefold() for name in actual}:
            actual.append(speaker)
    if len(actual) != 2:
        raise ValueError(f"expected exactly two message speakers, found: {actual}")
    return actual


def canonical_speaker(chat: Dict[str, Any], requested: str) -> str:
    """Resolve a paper split speaker against the message data."""
    actual = message_speakers(chat)
    matched = next(
        (speaker for speaker in actual if speaker.casefold() == requested.casefold()),
        None,
    )
    if matched is None:
        raise ValueError(
            f"split speaker {requested!r} is absent from message speakers {actual}"
        )
    return matched


def resolve_chat_roles(chat: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """Resolve legacy user/agent roles and report recoverable mismatches."""
    actual = message_speakers(chat)

    names = chat.get("name") or {}
    declared_user = str(names.get("speaker_1", "")).strip()
    declared_agent = str(names.get("speaker_2", "")).strip()
    warnings: List[str] = []

    def match(name: str) -> str | None:
        return next((speaker for speaker in actual if speaker.casefold() == name.casefold()), None)

    agent = match(declared_agent)
    if agent is None:
        raise ValueError(
            f"declared agent {declared_agent!r} is absent from message speakers {actual}"
        )
    user = match(declared_user)
    if user is None or user.casefold() == agent.casefold():
        user = next(speaker for speaker in actual if speaker.casefold() != agent.casefold())
        warnings.append(
            f"recovered user speaker as {user!r}; declared speaker_1 was {declared_user!r}"
        )
    return user, agent, warnings


def select_realtalk_splits(
    dataset_dir: str | Any,
    chat_filter: Sequence[str] | None = None,
    speaker_filter: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    """Select complete paper-reported speaker splits available in a dataset."""
    from pathlib import Path

    dataset = Path(dataset_dir)
    normalized_chats = {
        value.casefold() for value in (chat_filter or ()) if str(value).strip()
    }
    normalized_speakers = {
        value.casefold() for value in (speaker_filter or ()) if str(value).strip()
    }
    selected: List[Dict[str, str]] = []
    for split in REALTALK_PERSONA_SPLITS:
        if normalized_chats and not (
            split["train_chat"].casefold() in normalized_chats
            or split["test_chat"].casefold() in normalized_chats
        ):
            continue
        if normalized_speakers and split["speaker"].casefold() not in normalized_speakers:
            continue
        missing = [
            name
            for name in (split["train_chat"], split["test_chat"])
            if not (dataset / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"incomplete REALTALK split for {split['speaker']}: missing {missing}"
            )
        selected.append(dict(split))
    return selected


def build_profile_corpus(
    chat: Dict[str, Any],
    target_speaker: str,
    profile_sessions: int = 3,
) -> Dict[str, Any]:
    """Build the fixed Ca corpus used to derive a speaker representation."""
    if profile_sessions < 1:
        raise ValueError("profile_sessions must be at least 1")
    speaker = canonical_speaker(chat, target_speaker)
    selected_sessions = session_keys(chat)[:profile_sessions]
    if len(selected_sessions) < profile_sessions:
        raise ValueError(
            f"{speaker} train chat has only {len(selected_sessions)} sessions; "
            f"{profile_sessions} required"
        )
    selected = set(selected_sessions)
    turns = [
        turn for turn in merge_consecutive_utterances(chat)
        if turn["session_id"] in selected
    ]
    if not any(turn["speaker"].casefold() == speaker.casefold() for turn in turns):
        raise ValueError(f"no training messages found for {speaker}")
    text = format_turns(turns)
    return {
        "speaker": speaker,
        "sessions": selected_sessions,
        "turns": turns,
        "text": text,
        "history_hash": stable_hash(text),
    }


def build_message_level_points(
    chat: Dict[str, Any],
    target_speaker: str,
    test_sessions: int = 3,
    max_context_chars: int = 60000,
    max_eval_points: int = 0,
) -> List[Dict[str, Any]]:
    """Build rolling message-level targets from the selected Cb sessions.

    Each target is one merged message by ``target_speaker``. Its history is
    every real merged turn before it in the selected test segment. Exp1 adds
    the observed target message later because it classifies current state
    instead of generating the next message.
    """
    if test_sessions < 1:
        raise ValueError("test_sessions must be at least 1")
    if max_context_chars < 0:
        raise ValueError("max_context_chars must be non-negative")

    speaker = canonical_speaker(chat, target_speaker)
    selected_sessions = session_keys(chat)[:test_sessions]
    if len(selected_sessions) < test_sessions:
        raise ValueError(
            f"{speaker} test chat has only {len(selected_sessions)} sessions; "
            f"{test_sessions} required"
        )
    selected = set(selected_sessions)
    turns = [
        turn for turn in merge_consecutive_utterances(chat)
        if turn["session_id"] in selected
    ]
    points: List[Dict[str, Any]] = []
    for target_position, target in enumerate(turns):
        if target["speaker"].casefold() != speaker.casefold():
            continue
        context_turns = list(turns[:target_position])
        context_turns, context_truncated = trim_oldest_turns(
            context_turns, max_context_chars
        )
        context_text = format_turns(context_turns)
        target_index = len(points)
        points.append({
            "eval_id": f"message_{target_index}",
            "sample_id": f"message_{target_index}:{target['turn_id']}",
            "message_level_index": target_index,
            "test_sessions": list(selected_sessions),
            "target_session": target["session_id"],
            "target": target,
            "target_message": target["content"],
            "context_turns": context_turns,
            "context_text": context_text,
            "context_truncated": context_truncated,
            "context_session_count": len(set(t["session_id"] for t in context_turns)),
            "history_hash": stable_hash(context_text),
        })
    if max_eval_points > 0:
        return points[:max_eval_points]
    return points


def trim_oldest_turns(
    turns: Sequence[Dict[str, Any]], max_chars: int
) -> Tuple[List[Dict[str, Any]], bool]:
    """Drop complete oldest turns while always retaining the newest turn."""
    kept = list(turns)
    if max_chars == 0:
        return kept, False
    truncated = False
    while len(kept) > 1 and len(format_turns(kept)) > max_chars:
        kept.pop(0)
        truncated = True
    return kept, truncated


def format_turns(turns: Iterable[Dict[str, Any]]) -> str:
    return "\n".join(f"{turn['speaker']}: {turn['content']}" for turn in turns)


def stable_hash(payload: Any) -> str:
    if isinstance(payload, str):
        serialized = payload
    else:
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

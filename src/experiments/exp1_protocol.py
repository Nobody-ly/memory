"""Causal, provider-independent data protocol for Experiment 1."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .persona_simulation import flatten_messages, session_keys


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


def resolve_chat_roles(chat: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """Resolve user/agent roles and report recoverable metadata mismatches."""
    actual: List[str] = []
    for turn in flatten_messages(chat):
        speaker = turn["speaker"]
        if speaker and speaker.casefold() not in {name.casefold() for name in actual}:
            actual.append(speaker)
    if len(actual) != 2:
        raise ValueError(f"expected exactly two message speakers, found: {actual}")

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


def build_session_boundary_points(
    chat: Dict[str, Any],
    user_speaker: str,
    min_context_sessions: int = 2,
    context_sessions: int | None = 3,
    max_context_chars: int = 60000,
    max_eval_points: int = 15,
) -> List[Dict[str, Any]]:
    """Create one causal target at each eligible session boundary."""
    if min_context_sessions < 1:
        raise ValueError("min_context_sessions must be at least 1")
    if context_sessions is not None and context_sessions < 1:
        raise ValueError("context_sessions must be positive or None for all")
    if max_context_chars < 0:
        raise ValueError("max_context_chars must be non-negative")

    sessions = session_keys(chat)
    turns = merge_consecutive_utterances(chat)
    points: List[Dict[str, Any]] = []
    for boundary_index in range(min_context_sessions, len(sessions)):
        completed_sessions = sessions[:boundary_index]
        target_session = sessions[boundary_index]
        target_position = next(
            (
                index
                for index, turn in enumerate(turns)
                if turn["session_id"] == target_session
                and turn["speaker"].casefold() == user_speaker.casefold()
            ),
            None,
        )
        if target_position is None:
            continue
        target = turns[target_position]
        profile_turns = [
            turn for turn in turns if turn["session_id"] in completed_sessions
        ]
        selected_sessions = (
            completed_sessions
            if context_sessions is None
            else completed_sessions[-context_sessions:]
        )
        context_turns = [
            turn
            for index, turn in enumerate(turns[:target_position])
            if turn["session_id"] in selected_sessions
            or turn["session_id"] == target_session
        ]
        context_turns, context_truncated = trim_oldest_turns(
            context_turns, max_context_chars
        )
        profile_text = format_turns(profile_turns)
        context_text = format_turns(context_turns)
        points.append({
            "eval_id": f"session_boundary_{boundary_index}",
            "sample_id": f"session_boundary_{boundary_index}:{target['turn_id']}",
            "boundary_index": boundary_index,
            "completed_sessions": list(completed_sessions),
            "target_session": target_session,
            "target": target,
            "target_message": target["content"],
            "profile_turns": profile_turns,
            "profile_text": profile_text,
            "profile_history_hash": stable_hash(profile_text),
            "context_turns": context_turns,
            "context_text": context_text,
            "context_truncated": context_truncated,
            "context_session_count": len(set(t["session_id"] for t in context_turns)),
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

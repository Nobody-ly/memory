"""Reconstruct PersonaEmp-style inputs from the fixed WildChat source.

This module deliberately separates three layers of provenance:
1. paper-specified constraints (WildChat, 6--249 turns, DeepSeek-v3.2,
   implicit-memory retention);
2. public PersonaEmp code applied after raw records exist; and
3. reconstructed settings that the papers do not disclose (memory prompt,
   category cap, semantic-dedup threshold, and random seed).

It does not claim an author-identical AlpsBench reconstruction because the
upstream annotation prompt, sampling cap, deduplication threshold, and human
correction records were never released.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from .client import ChatBackend, OpenAICompatibleChatBackend
from .generation import _parse_json_object, prompt_hash


WILDCHAT_REPOSITORY = "allenai/WildChat-1M"
WILDCHAT_REVISION = "78acdff91e618128920151298128e3f85d6e423d"
MIN_TURNS = 6
MAX_TURNS = 249
PAPER_MEMORY_MODEL = "deepseek-v3.2"
MAX_MEMORY_ITEMS = 8
LOCAL_NORMALIZATION_VERSION = "task_content_and_evidence_v4"
DEFAULT_DEDUP_ENCODER = "intfloat/e5-base-v2"
DEFAULT_DEDUP_THRESHOLD = 0.92
DOCUMENT_EDITING_PATTERN = re.compile(
    r"\b(?:check|cehck|correct|fix)\s+(?:the\s+)?grammar\b"
    r"|\bgrammar\s+(?:check|correction)\b"
    r"|\b(?:proofread|rewrite|polish)\s+(?:this|the|my)\b"
    r"|\btranslate\s+(?:this\s+)?(?:into|to|in)\b",
    re.IGNORECASE,
)
TRANSIENT_SPEECH_ACT_VALUE_PATTERN = re.compile(
    r"^(?:the\s+)?user\s+(?:asks?|asked|is\s+asking|wants?\s+to\s+know|"
    r"is\s+curious\s+about|is\s+inquiring\s+about|is\s+looking\s+for|"
    r"seeks?\s+information\s+about)\b",
    re.IGNORECASE,
)
HYPOTHETICAL_TASK_VALUE_PATTERN = re.compile(
    r"\b(?:hypothetical|fictional|role[- ]play\s+character|"
    r"scenario\s+where|a\s+(?:person|boy|girl|man|woman)\s+named)\b",
    re.IGNORECASE,
)
INTENT_ALLOWLIST = (
    "Learning Support",
    "Conversational Engagement",
    "Personal Advice",
    "Decision Support",
    "Business",
    "Career Advice",
    "Moral and Ethical Queries",
    "Reflection and Insight",
    "Existential Questions",
    "Societal and Cultural Inquiry",
    "Philosophical",
)
MEMORY_TOP_LEVEL_LABELS = (
    "Personal_Background",
    "States_Experiences",
    "Possessions",
    "Preferences",
    "Thoughts",
    "Plans",
    "Social_Relationships",
    "Constraints_and_Boundaries",
    "UNMAPPED",
)
MEMORY_SYSTEM_PROMPT = """You extract durable, personalized memory from real
human-AI conversations for a long-term personalization benchmark. Use only
facts supported by user-authored turns. Distinguish direct statements from
implicit but well-supported inferences. Preserve uncertainty instead of
inventing detail. Every memory must cite concrete user-turn indices.

Treat requested stories, fictional characters, role-play settings, quoted
material, and hypothetical examples as task content, not facts about the user.
Never assign a character's identity, relationships, possessions, emotions, or
events to the user unless the user explicitly identifies themself as that
character. A single topical request does not establish a stable interest,
preference, profession, plan, emotional state, or communication style."""
MEMORY_USER_TEMPLATE = """Conversation (turns are zero-indexed):
{conversation}

Return only JSON. Extract a compact set of durable, user-specific memories.
For label, use a hierarchical path beginning with exactly one of:
{labels}

Use `UNMAPPED` only when no listed family fits and provide a concise
label_suggestion. `type` is `direct` for explicit user statements and
`implicit` only for a stable inference supported by the cited text. Do not
extract temporary assistant content, generic facts, or unsafe diagnoses.
`direct` requires an explicit self-fact, preference, past experience, belief,
plan, possession, relationship, or durable constraint. A request, question,
command, correction, or test of assistant capabilities is not itself durable
memory: never output items such as "the user asked/requested/inquired about X".
Do not discard a durable direct statement merely because it appears inside a
conversation dominated by requests or questions. Explicit identity claims,
beliefs, preferences, past experiences, plans, possessions, relationships, and
constraints remain direct memory when the user presents them as their own.
Record what the user claims rather than deciding whether an unusual claim is
true; calibrate confidence and preserve uncertainty. The fictional-content
rule still applies when the statement is explicitly framed as story, role-play,
quotation, character description, or hypothetical material.
Text pasted for grammar checking, translation, rewriting, summarization, or
other document editing is task content. Do not infer that its people, finances,
relationships, events, or claims describe the user. A resume or biography may
support direct memory only when the user explicitly presents it as their own.
Use repeated requests only as evidence for one consolidated `implicit` pattern.
An `implicit` memory MUST be supported by at least two distinct user-authored
turns. Put every supporting user-turn index in `supporting_turn_indices`.
One-off topical curiosity or a single task request is not implicit memory.
After collecting direct candidates, scan repeated user behavior for a small
number of supported implicit patterns. Before returning, consolidate candidates
into the smallest non-overlapping set: multiple turns about the same emotional
state, preference, relationship, experience, plan, or constraint form one
memory, not one memory per turn or paraphrase. Prefer omission over
fragmentation and return at most 8 compact memories for the whole conversation.

Also assign every applicable intent from the given allowlist. Select `Other`
only if none applies:
{intents}
"""
MEMORY_SCHEMA = {
    "name": "alpsbench_memory_reconstruction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "intents": {
                "type": "array",
                "items": {"type": "string", "enum": [*INTENT_ALLOWLIST, "Other"]},
            },
            "memory_items": {
                "type": "array",
                "maxItems": MAX_MEMORY_ITEMS,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["direct", "implicit"]},
                        "label": {"type": "string"},
                        "label_suggestion": {"type": "string"},
                        "value": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "evidence_turn_index": {"type": "integer", "minimum": 0},
                        "supporting_turn_indices": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                        "evidence_text": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "time_scope": {
                            "type": "string",
                            "enum": ["long_term", "ongoing", "current", "unknown"],
                        },
                        "emotion": {"type": "string"},
                        "preference_attitude": {"type": "string"},
                    },
                    "required": [
                        "type",
                        "label",
                        "label_suggestion",
                        "value",
                        "reasoning",
                        "evidence_turn_index",
                        "supporting_turn_indices",
                        "evidence_text",
                        "confidence",
                        "time_scope",
                        "emotion",
                        "preference_attitude",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intents", "memory_items"],
        "additionalProperties": False,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _normalise_role(value: Any) -> str | None:
    role = str(value or "").strip().lower()
    if role in {"user", "human", "prompter"}:
        return "user"
    if role in {"assistant", "gpt", "bot", "model"}:
        return "assistant"
    return None


def _text_from_turn(turn: dict[str, Any]) -> str:
    for key in ("text", "content", "message"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _conversation_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("conversation") or row.get("messages") or row.get("dialogue") or []
    if not isinstance(raw, list):
        return []
    turns: list[dict[str, str]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = _normalise_role(turn.get("role") or turn.get("speaker"))
        text = _text_from_turn(turn)
        if role and text:
            turns.append({"role": role, "text": text})
    return turns


def _source_id(row: dict[str, Any], *, source_file: str, source_row: int) -> str:
    for key in ("conversation_id", "conversation_uuid", "id", "uuid"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    digest = hashlib.sha256(f"{source_file}:{source_row}".encode("utf-8")).hexdigest()
    return f"wildchat-{digest[:24]}"


def _safe_session_id(source_id: str) -> str:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    return f"wc_{digest[:24]}"


@dataclass(frozen=True)
class SourceStats:
    raw_rows: int
    usable_rows: int
    accepted_long_dialogues: int
    rejected_missing_dialogue: int
    rejected_language: int
    rejected_turn_range: int
    rejected_without_user_turn: int


def select_long_dialogues(
    rows: Iterable[tuple[str, int, dict[str, Any]]],
    *,
    limit: int | None = None,
    source_language: str | None = None,
) -> tuple[list[dict[str, Any]], SourceStats]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    raw_rows = usable_rows = accepted = missing = language = turn_range = no_user = 0
    selected: list[dict[str, Any]] = []
    for source_file, source_row, row in rows:
        raw_rows += 1
        turns = _conversation_from_row(row)
        if not turns:
            missing += 1
            continue
        usable_rows += 1
        row_language = str(row.get("language") or "").strip()
        if (
            source_language is not None
            and row_language.casefold() != source_language.casefold()
        ):
            language += 1
            continue
        if not MIN_TURNS <= len(turns) <= MAX_TURNS:
            turn_range += 1
            continue
        if not any(turn["role"] == "user" for turn in turns):
            no_user += 1
            continue
        source_id = _source_id(row, source_file=source_file, source_row=source_row)
        selected.append(
            {
                "source_conversation_id": source_id,
                "session_id": _safe_session_id(source_id),
                "source_file": source_file,
                "source_row": source_row,
                "source_language": row_language or None,
                "turns": turns,
            }
        )
        accepted += 1
        if limit is not None and accepted >= limit:
            break
    return selected, SourceStats(
        raw_rows=raw_rows,
        usable_rows=usable_rows,
        accepted_long_dialogues=accepted,
        rejected_missing_dialogue=missing,
        rejected_language=language,
        rejected_turn_range=turn_range,
        rejected_without_user_turn=no_user,
    )


def iter_snapshot_rows(snapshot_dir: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
    jsonl_files = sorted(snapshot_dir.rglob("*.jsonl"))
    if parquet_files:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read WildChat parquet files") from exc
        for path in parquet_files:
            parquet = pq.ParquetFile(path)
            row_index = 0
            for batch in parquet.iter_batches(batch_size=1_024):
                for row in batch.to_pylist():
                    if isinstance(row, dict):
                        yield str(path.relative_to(snapshot_dir)), row_index, row
                    row_index += 1
        return
    if jsonl_files:
        for path in jsonl_files:
            for row_index, row in enumerate(_read_jsonl(path)):
                yield str(path.relative_to(snapshot_dir)), row_index, row
        return
    raise RuntimeError(f"no parquet or JSONL source files found under {snapshot_dir}")


def download_wildchat_snapshot(
    output_dir: Path,
    *,
    allow_patterns: list[str] | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to download WildChat") from exc
    destination = output_dir / "wildchat_snapshot"
    return Path(
        snapshot_download(
            repo_id=WILDCHAT_REPOSITORY,
            repo_type="dataset",
            revision=WILDCHAT_REVISION,
            local_dir=str(destination),
            allow_patterns=allow_patterns,
        )
    )


def _looks_like_document_editing_task(text: str) -> bool:
    return bool(DOCUMENT_EDITING_PATTERN.search(text))


def _format_conversation(turns: list[dict[str, str]]) -> str:
    rows = []
    for index, turn in enumerate(turns):
        annotation = ""
        if turn["role"] == "user" and _looks_like_document_editing_task(turn["text"]):
            annotation = (
                " [document-editing request: draft facts are not user facts unless "
                "explicitly identified as the user's own]"
            )
        rows.append(f"[{index}] {turn['role']}{annotation}: {turn['text']}")
    return "\n".join(rows)


def _label_is_valid(value: str) -> bool:
    top = value.split("/", 1)[0].strip()
    return top in MEMORY_TOP_LEVEL_LABELS


def _normalise_memory_item(
    item: dict[str, Any],
    *,
    session_id: str,
    turns: list[dict[str, str]],
    ordinal: int,
) -> dict[str, Any] | None:
    value = str(item.get("value") or "").strip()
    label = str(item.get("label") or "").strip()
    item_type = str(item.get("type") or "").strip().lower()
    if not value or item_type not in {"direct", "implicit"}:
        return None
    if not _label_is_valid(label):
        label = "UNMAPPED"
    try:
        turn_index = int(item.get("evidence_turn_index") or 0)
    except (TypeError, ValueError):
        turn_index = 0
    if not 0 <= turn_index < len(turns) or turns[turn_index]["role"] != "user":
        user_indices = [i for i, turn in enumerate(turns) if turn["role"] == "user"]
        if not user_indices:
            return None
        turn_index = user_indices[0]
    supporting_indices: list[int] = []
    for raw_index in item.get("supporting_turn_indices") or []:
        try:
            support_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if (
            0 <= support_index < len(turns)
            and turns[support_index]["role"] == "user"
            and support_index not in supporting_indices
        ):
            supporting_indices.append(support_index)
    if turn_index not in supporting_indices:
        supporting_indices.insert(0, turn_index)
    if item_type == "implicit" and len(supporting_indices) < 2:
        return None
    if item_type == "direct" and supporting_indices and all(
        _looks_like_document_editing_task(turns[index]["text"])
        for index in supporting_indices
    ):
        return None
    if item_type == "direct" and (
        TRANSIENT_SPEECH_ACT_VALUE_PATTERN.search(value)
        or HYPOTHETICAL_TASK_VALUE_PATTERN.search(value)
    ):
        return None
    evidence = turns[turn_index]["text"]
    reported_evidence = str(item.get("evidence_text") or "").strip()
    if reported_evidence and reported_evidence not in evidence:
        # Preserve the verifiable source turn rather than a paraphrased citation.
        reported_evidence = evidence
    confidence = float(item.get("confidence") or 0.0)
    return {
        "memory_id": f"{session_id}:m{ordinal:03d}",
        "type": item_type,
        "label": label,
        "label_suggestion": str(item.get("label_suggestion") or "").strip(),
        "value": value,
        "reasoning": str(item.get("reasoning") or "").strip(),
        "evidence": {
            "session_id": session_id,
            "utterance_index": turn_index,
            "supporting_utterance_indices": supporting_indices,
            "text": reported_evidence or evidence,
        },
        "confidence": min(max(confidence, 0.0), 1.0),
        "time_scope": str(item.get("time_scope") or "unknown").strip(),
        "emotion": str(item.get("emotion") or "").strip() or None,
        "preference_attitude": str(item.get("preference_attitude") or "").strip() or None,
    }


def _deduplicate_memory_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("label") or "").lower(),
            re.sub(r"\s+", " ", str(item.get("value") or "").lower()).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


class MemoryExtractionCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.values: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for row in _read_jsonl(path):
                key = str(row.get("cache_key") or "")
                if key:
                    self.values[key] = row

    def get(self, key: str) -> dict[str, Any] | None:
        return self.values.get(key)

    def save(self, row: dict[str, Any]) -> None:
        key = str(row["cache_key"])
        if key in self.values:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as destination:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            destination.flush()
        self.values[key] = row


@dataclass(frozen=True)
class MemoryExtractionStats:
    attempted: int
    succeeded: int
    cached: int
    content_rejected: int
    failed: int
    memories: int
    direct_memories: int
    implicit_memories: int


class PaperMemoryExtractor:
    def __init__(self, backend: ChatBackend, cache: MemoryExtractionCache) -> None:
        if backend.model != PAPER_MEMORY_MODEL:
            raise ValueError(
                f"AlpsBench specifies {PAPER_MEMORY_MODEL}; got {backend.model}"
            )
        self.backend = backend
        self.cache = cache

    def _key(self, record: dict[str, Any]) -> tuple[str, dict[str, str]]:
        turns = record["turns"]
        provenance = {
            "model": self.backend.model,
            "local_normalization_version": LOCAL_NORMALIZATION_VERSION,
            "system_prompt_sha256": prompt_hash(MEMORY_SYSTEM_PROMPT),
            "user_prompt_sha256": prompt_hash(MEMORY_USER_TEMPLATE),
            "schema_sha256": prompt_hash(json.dumps(MEMORY_SCHEMA, sort_keys=True)),
            "conversation_sha256": prompt_hash(_format_conversation(turns)),
        }
        value = {"session_id": record["session_id"], **provenance}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest(), provenance

    def extract(self, record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        key, provenance = self._key(record)
        cached = self.cache.get(key)
        if cached is not None:
            if cached.get("status") != "success":
                raise RuntimeError(str(cached.get("error") or "cached memory extraction failure"))
            return dict(cached["record"]), True
        prompt = MEMORY_USER_TEMPLATE.format(
            conversation=_format_conversation(record["turns"]),
            labels=", ".join(MEMORY_TOP_LEVEL_LABELS),
            intents=", ".join(INTENT_ALLOWLIST),
        )
        try:
            result = self.backend.chat(
                MEMORY_SYSTEM_PROMPT,
                prompt,
                temperature=0.0,
                max_tokens=3_000,
                response_schema=MEMORY_SCHEMA,
            )
            parsed = _parse_json_object(result.content)
            raw_memories = parsed.get("memory_items") or []
            if not isinstance(raw_memories, list):
                raise ValueError("memory_items is not a list")
            memories = _deduplicate_memory_items(
                [
                    normalized
                    for index, item in enumerate(raw_memories, 1)
                    if isinstance(item, dict)
                    and (
                        normalized := _normalise_memory_item(
                            item,
                            session_id=record["session_id"],
                            turns=record["turns"],
                            ordinal=index,
                        )
                    )
                    is not None
                ]
            )
            intents = list(
                dict.fromkeys(
                    value
                    for value in parsed.get("intents") or []
                    if value in {*INTENT_ALLOWLIST, "Other"}
                )
            )
            if not intents:
                intents = ["Other"]
            result_record = {
                **record,
                "sessions": [{"session_id": record["session_id"], "turns": record["turns"]}],
                "intents_ranked": [
                    {"intent_category": value, "intent_subtype": ""}
                    for value in intents
                ],
                "memory_items": memories,
                "reconstruction_metadata": {
                    "source": "WildChat-1M",
                    "source_revision": WILDCHAT_REVISION,
                    "memory_model": self.backend.model,
                    "memory_prompt_sha256": provenance["system_prompt_sha256"],
                    "local_normalization_version": LOCAL_NORMALIZATION_VERSION,
                    "manual_annotation_performed": False,
                },
            }
            self.cache.save(
                {
                    "cache_key": key,
                    "status": "success",
                    "provenance": provenance,
                    "record": result_record,
                }
            )
            return result_record, False
        except Exception as exc:
            # Cache only completed extractions. A transient provider failure must
            # remain retryable when the same reconstruction directory is resumed.
            raise


def _terminal_content_rejection_reason(exc: Exception) -> str | None:
    message = str(exc).lower()
    if (
        "data_inspection_failed" in message
        and "inappropriate content" in message
    ):
        return "provider_data_inspection_failed"
    return None


def extract_memories(
    records: Iterable[dict[str, Any]], extractor: PaperMemoryExtractor
) -> tuple[
    list[dict[str, Any]],
    MemoryExtractionStats,
    list[dict[str, str]],
    list[dict[str, str]],
]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    attempted = succeeded = cached = memories = direct = implicit = 0
    for record in records:
        attempted += 1
        try:
            extracted, from_cache = extractor.extract(record)
        except Exception as exc:
            reason = _terminal_content_rejection_reason(exc)
            if reason is not None:
                exclusions.append(
                    {
                        "session_id": str(record["session_id"]),
                        "status": "content_rejected",
                        "reason_code": reason,
                    }
                )
                continue
            failures.append({"session_id": str(record["session_id"]), "error": str(exc)})
            continue
        results.append(extracted)
        succeeded += 1
        cached += int(from_cache)
        items = extracted.get("memory_items") or []
        memories += len(items)
        direct += sum(item.get("type") == "direct" for item in items if isinstance(item, dict))
        implicit += sum(item.get("type") == "implicit" for item in items if isinstance(item, dict))
    return results, MemoryExtractionStats(
        attempted=attempted,
        succeeded=succeeded,
        cached=cached,
        content_rejected=len(exclusions),
        failed=len(failures),
        memories=memories,
        direct_memories=direct,
        implicit_memories=implicit,
    ), failures, exclusions


def _top_label(memory: dict[str, Any]) -> str:
    return str(memory.get("label") or "UNMAPPED").split("/", 1)[0]


def paper_style_curation(
    records: list[dict[str, Any]],
    *,
    category_cap: int,
    encoder_name: str = DEFAULT_DEDUP_ENCODER,
    similarity_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    skip_semantic_dedup: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply paper-described retention rules with explicit reconstructed options.

    The paper names category caps and implicit-memory priority but not the cap
    values or tie-breakers. This deterministic implementation retains records
    with at least one implicit memory and assigns a record to its most common
    top-level label before applying the cap. An encoder-based semantic pass is
    available only as a separately recorded reconstruction sensitivity setting;
    it is disabled by default because the paper gives no encoder or threshold.
    """
    if category_cap < 1:
        raise ValueError("category_cap must be positive")
    buckets: dict[str, list[dict[str, Any]]] = {}
    rejected_no_implicit = 0
    for record in sorted(records, key=lambda value: str(value["session_id"])):
        memories = [item for item in record.get("memory_items") or [] if isinstance(item, dict)]
        if not any(item.get("type") == "implicit" for item in memories):
            rejected_no_implicit += 1
            continue
        labels = Counter(_top_label(item) for item in memories)
        category = sorted(labels, key=lambda name: (-labels[name], name))[0] if labels else "UNMAPPED"
        buckets.setdefault(category, []).append(record)
    capped: list[dict[str, Any]] = []
    before_caps = {category: len(items) for category, items in sorted(buckets.items())}
    for category, items in sorted(buckets.items()):
        capped.extend(items[:category_cap])

    dedup_stats: dict[str, Any] = {
        "enabled": not skip_semantic_dedup,
        "encoder": encoder_name,
        "similarity_threshold": similarity_threshold,
    }
    if skip_semantic_dedup:
        selected = capped
        dedup_stats["status"] = "skipped_by_flag"
    else:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers and numpy are required for semantic deduplication"
            ) from exc
        encoder = SentenceTransformer(encoder_name)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in capped:
            memories = record.get("memory_items") or []
            labels = Counter(_top_label(item) for item in memories if isinstance(item, dict))
            category = sorted(labels, key=lambda name: (-labels[name], name))[0] if labels else "UNMAPPED"
            grouped.setdefault(category, []).append(record)
        selected = []
        duplicate_count = 0
        for category, items in sorted(grouped.items()):
            texts = [
                "passage: " + category + " " + " ".join(
                    str(item.get("value") or "")
                    for item in record.get("memory_items") or []
                    if isinstance(item, dict)
                )
                for record in items
            ]
            vectors = encoder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            kept_vectors: list[Any] = []
            for record, vector in zip(items, vectors):
                if kept_vectors and max(float(np.dot(vector, previous)) for previous in kept_vectors) >= similarity_threshold:
                    duplicate_count += 1
                    continue
                selected.append(record)
                kept_vectors.append(vector)
        dedup_stats.update({"status": "completed", "removed_records": duplicate_count})
    after_caps = Counter()
    for record in selected:
        labels = Counter(_top_label(item) for item in record.get("memory_items") or [] if isinstance(item, dict))
        category = sorted(labels, key=lambda name: (-labels[name], name))[0] if labels else "UNMAPPED"
        after_caps[category] += 1
    return selected, {
        "input_records": len(records),
        "rejected_without_implicit_memory": rejected_no_implicit,
        "records_before_category_cap": before_caps,
        "records_after_category_cap": dict(sorted(after_caps.items())),
        "category_cap": category_cap,
        "semantic_deduplication": dedup_stats,
    }


def _tokens(value: str) -> set[str]:
    return {part for part in re.split(r"\W+", value.lower()) if part}


def _pair_score(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, float, float, float]:
    left_parts = str(left.get("label") or "").lower().split("/")
    right_parts = str(right.get("label") or "").lower().split("/")
    prefix = 0
    for a, b in zip(left_parts, right_parts):
        if a != b:
            break
        prefix += 1
    label_score = 2 * prefix / max(len(left_parts) + len(right_parts), 1)
    l_tokens, r_tokens = _tokens(str(left.get("value") or "")), _tokens(str(right.get("value") or ""))
    value_score = len(l_tokens & r_tokens) / max(len(l_tokens | r_tokens), 1)
    type_score = float(left.get("type") == right.get("type"))
    return 0.45 * label_score + 0.45 * value_score + 0.10 * type_score, label_score, value_score, type_score


def compare_memory_sets(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[float, int, int, float, float, float]] = []
    for p_index, p_item in enumerate(predicted):
        for g_index, g_item in enumerate(gold):
            score, label, value, item_type = _pair_score(p_item, g_item)
            pairs.append((score, p_index, g_index, label, value, item_type))
    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    matches: list[tuple[float, float, float, float]] = []
    for score, p_index, g_index, label, value, item_type in sorted(pairs, reverse=True):
        if score < 0.30 or p_index in matched_pred or g_index in matched_gold:
            continue
        matched_pred.add(p_index)
        matched_gold.add(g_index)
        matches.append((score, label, value, item_type))
    precision = len(matches) / len(predicted) if predicted else 0.0
    recall = len(matches) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted_count": len(predicted),
        "gold_count": len(gold),
        "matched_count": len(matches),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_match_score": round(sum(item[0] for item in matches) / len(matches), 6) if matches else 0.0,
        "mean_label_score": round(sum(item[1] for item in matches) / len(matches), 6) if matches else 0.0,
        "mean_value_score": round(sum(item[2] for item in matches) / len(matches), 6) if matches else 0.0,
        "type_match_rate": round(sum(item[3] for item in matches) / len(matches), 6) if matches else 0.0,
        "extra_count": len(predicted) - len(matches),
        "missing_count": len(gold) - len(matches),
    }


def _public_gold_record_to_source(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("input") or row
    dialogue = payload.get("dialogue") or payload.get("conversation") or []
    turns = []
    for turn in dialogue:
        if isinstance(turn, dict):
            role = _normalise_role(turn.get("role") or turn.get("speaker"))
            text = _text_from_turn(turn)
            if role and text:
                turns.append({"role": role, "text": text})
    identifier = str(row.get("benchmark_id") or row.get("session_id") or "gold")
    return {
        "source_conversation_id": identifier,
        "session_id": _safe_session_id(identifier),
        "source_file": "public_gold",
        "source_row": 0,
        "turns": turns,
    }


def audit_public_gold(
    input_path: Path,
    reference_path: Path,
    extractor: PaperMemoryExtractor,
    *,
    limit: int,
) -> dict[str, Any]:
    inputs = list(_read_jsonl(input_path))[:limit]
    references = {
        str(row.get("benchmark_id") or row.get("session_id") or ""): row
        for row in _read_jsonl(reference_path)
    }
    rows = []
    for source_row in inputs:
        identifier = str(source_row.get("benchmark_id") or source_row.get("session_id") or "")
        reference = references.get(identifier)
        if reference is None:
            continue
        source = _public_gold_record_to_source(source_row)
        predicted, _ = extractor.extract(source)
        gold = ((reference.get("gold") or {}).get("memory_items") or [])
        comparison = compare_memory_sets(predicted.get("memory_items") or [], gold)
        rows.append({
            "benchmark_id": identifier,
            "comparison": comparison,
            "predicted_memory_items": predicted.get("memory_items") or [],
            "gold_memory_items": gold,
        })
    aggregate = {
        "records": len(rows),
        "mean_f1": round(sum(row["comparison"]["f1"] for row in rows) / len(rows), 6) if rows else 0.0,
        "mean_precision": round(sum(row["comparison"]["precision"] for row in rows) / len(rows), 6) if rows else 0.0,
        "mean_recall": round(sum(row["comparison"]["recall"] for row in rows) / len(rows), 6) if rows else 0.0,
        "mean_match_score": round(sum(row["comparison"]["mean_match_score"] for row in rows) / len(rows), 6) if rows else 0.0,
        "rows": rows,
    }
    return aggregate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct PersonaEmp inputs from fixed WildChat-1M."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--download-pattern",
        action="append",
        default=[],
        help=(
            "Optional Hugging Face allow-pattern. Repeat for multiple files. "
            "Use a fixed Parquet shard only for a bounded pilot; omit for all shards."
        ),
    )
    parser.add_argument("--source-limit", type=int)
    parser.add_argument("--source-language")
    parser.add_argument("--category-cap", type=int, default=350)
    parser.add_argument("--dedup-encoder", default=DEFAULT_DEDUP_ENCODER)
    parser.add_argument("--dedup-threshold", type=float, default=DEFAULT_DEDUP_THRESHOLD)
    parser.add_argument(
        "--enable-reconstructed-semantic-dedup",
        action="store_true",
        help=(
            "Enable the optional E5 threshold pass. This is not paper-defined "
            "and is disabled in the default reconstruction."
        ),
    )
    parser.add_argument("--gold-input", type=Path)
    parser.add_argument("--gold-reference", type=Path)
    parser.add_argument("--gold-limit", type=int, default=12)
    parser.add_argument("--env-prefix", default="PERSONAEMP_MEMORY")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    snapshot = args.snapshot_dir.resolve() if args.snapshot_dir else None
    if args.download:
        if snapshot is not None:
            raise ValueError("--download and --snapshot-dir cannot be combined")
        snapshot = download_wildchat_snapshot(
            output_dir,
            allow_patterns=args.download_pattern or None,
        )
    if snapshot is None:
        raise ValueError("provide --snapshot-dir or use --download")
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    selected, source_stats = select_long_dialogues(
        iter_snapshot_rows(snapshot),
        limit=args.source_limit,
        source_language=args.source_language,
    )
    _write_jsonl(output_dir / "stages" / "long_dialogues.jsonl", selected)
    backend = OpenAICompatibleChatBackend.from_env(args.env_prefix)
    extractor = PaperMemoryExtractor(
        backend, MemoryExtractionCache(output_dir / "cache" / "memory_extraction.jsonl")
    )
    extracted, extraction_stats, failures, exclusions = extract_memories(
        selected, extractor
    )
    _write_jsonl(output_dir / "stages" / "memory_extracted.jsonl", extracted)
    _write_jsonl(output_dir / "stages" / "memory_failures.jsonl", failures)
    _write_jsonl(output_dir / "stages" / "memory_exclusions.jsonl", exclusions)
    curated, curation_stats = paper_style_curation(
        extracted,
        category_cap=args.category_cap,
        encoder_name=args.dedup_encoder,
        similarity_threshold=args.dedup_threshold,
        skip_semantic_dedup=not args.enable_reconstructed_semantic_dedup,
    )
    _atomic_json(output_dir / "by_label_json" / "wildchat_reconstruction.json", curated)
    audit_path = None
    if bool(args.gold_input) != bool(args.gold_reference):
        raise ValueError("--gold-input and --gold-reference must be supplied together")
    if args.gold_input and args.gold_reference:
        audit = audit_public_gold(
            args.gold_input,
            args.gold_reference,
            extractor,
            limit=args.gold_limit,
        )
        audit_path = output_dir / "quality" / "public_gold_memory_comparison.json"
        _atomic_json(audit_path, audit)
    review_queue = output_dir / "quality" / "manual_review_queue.jsonl"
    _write_jsonl(
        review_queue,
        (
            {
                "session_id": record["session_id"],
                "memory_items": record.get("memory_items") or [],
                "reviewer": "Codex",
                "status": "pending",
                "manual_annotation_performed": False,
            }
            for record in curated[: min(20, len(curated))]
        ),
    )
    manifest = {
        "created_at": _utc_now(),
        "protocol": "personaemp_wildchat_reconstruction_v1",
        "source": {
            "repository": WILDCHAT_REPOSITORY,
            "revision": WILDCHAT_REVISION,
            "snapshot_dir": str(snapshot),
            "download_patterns": args.download_pattern,
            "turn_range": [MIN_TURNS, MAX_TURNS],
            "pilot_source_language_filter": args.source_language,
        },
        "paper_defined": {
            "memory_model": PAPER_MEMORY_MODEL,
            "implicit_memory_retention": True,
            "manual_annotation_performed": False,
        },
        "reconstructed_settings": {
            "memory_prompt_sha256": prompt_hash(MEMORY_SYSTEM_PROMPT),
            "local_normalization_version": LOCAL_NORMALIZATION_VERSION,
            "memory_schema_sha256": prompt_hash(json.dumps(MEMORY_SCHEMA, sort_keys=True)),
            "category_cap": args.category_cap,
            "max_memory_items_per_conversation": {
                "value": MAX_MEMORY_ITEMS,
                "basis": "covers 930 of 932 public AlpsBench Task 1 gold records",
            },
            "content_rejection_policy": {
                "status": "provider_compatibility_rule",
                "terminal_reason_code": "provider_data_inspection_failed",
                "excluded_from_downstream": True,
                "counted_as_unresolved_failure": False,
            },
            "implicit_memory_policy": {
                "minimum_distinct_user_turns": 2,
                "fictional_character_facts_are_user_facts": False,
                "single_topical_request_establishes_stable_trait": False,
            },
            "document_editing_task_policy": {
                "annotated_in_prompt": True,
                "direct_memory_from_only_annotated_turns": False,
                "repeated_behavior_may_support_implicit_memory": True,
                "pattern_sha256": prompt_hash(DOCUMENT_EDITING_PATTERN.pattern),
            },
            "local_task_content_filters": {
                "direct_transient_speech_act_value_sha256": prompt_hash(
                    TRANSIENT_SPEECH_ACT_VALUE_PATTERN.pattern
                ),
                "direct_hypothetical_task_value_sha256": prompt_hash(
                    HYPOTHETICAL_TASK_VALUE_PATTERN.pattern
                ),
            },
            "semantic_deduplication": {
                "encoder": args.dedup_encoder,
                "threshold": args.dedup_threshold,
                "enabled_as_reconstruction_sensitivity": (
                    args.enable_reconstructed_semantic_dedup
                ),
            },
        },
        "source_stats": asdict(source_stats),
        "memory_extraction_stats": asdict(extraction_stats),
        "memory_extraction_complete": extraction_stats.failed == 0,
        "curation_stats": curation_stats,
        "quality_audit": str(audit_path) if audit_path else None,
        "manual_review_queue": str(review_queue),
        "artifacts": {
            "long_dialogues": str(output_dir / "stages" / "long_dialogues.jsonl"),
            "memory_extracted": str(output_dir / "stages" / "memory_extracted.jsonl"),
            "memory_failures": str(output_dir / "stages" / "memory_failures.jsonl"),
            "memory_exclusions": str(
                output_dir / "stages" / "memory_exclusions.jsonl"
            ),
            "by_label_json": str(output_dir / "by_label_json" / "wildchat_reconstruction.json"),
        },
        "table1_direct_comparison_allowed": False,
    }
    _atomic_json(output_dir / "wildchat_reconstruction_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

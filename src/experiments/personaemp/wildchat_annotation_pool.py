"""Reconstruct the pre-PersonaEmp WildChat annotation pool.

This pipeline stops at the ``by_label_json`` boundary consumed by the public
PersonaEmp preparation code. It intentionally does not apply AlpsBench's final
2,500-sample implicit-memory/category-cap benchmark curation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from .client import ChatBackend, OpenAICompatibleChatBackend
from .generation import _parse_json_object, prompt_hash
from .wildchat_reconstruction import (
    MAX_TURNS,
    MIN_TURNS,
    PAPER_MEMORY_MODEL,
    WILDCHAT_REPOSITORY,
    WILDCHAT_REVISION,
    iter_snapshot_rows,
)


PROTOCOL = "personaemp_wildchat_annotation_pool_v1"
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
DEFAULT_INTENT_CATEGORIES = (
    "Technical and Professional Intent",
    "Problem-Solving Intent",
    "Educational Intent",
    "Transactional Intent",
    "Informational Intent",
    "Creative Intent",
    "Personal Interaction Intent",
    "Ethical and Philosophical Intent",
    "Data and Information Management",
)
DEFAULT_INTENT_SUBTYPES = (
    "Technical Guidance",
    "Troubleshooting Assistance",
    "Learning Support",
    "Data Processing",
    "Factual Queries",
    "Content Creation",
    "Service Utilization",
    "Industry-Specific Inquiries",
    "Conversational Engagement",
    "Personal Advice",
    "Decision Support",
    "Reflection and Insight",
    "Business and Career Advice",
    "Explanatory Inquiries",
    "Tutorial Requests",
    "Skill Development",
    "Societal and Cultural Inquiry",
    "Task Automation",
    "Planning and Organization",
    "Existential Questions",
    "Moral and Ethical Queries",
    "Curricular Planning",
    "Idea Generation",
    "Artistic Exploration",
    "Other",
)

ANNOTATION_SYSTEM_PROMPT = """You create structured personalization annotations
from real human-AI dialogue. Follow the public AlpsBench memory contract. Extract
only user-grounded information, distinguish direct statements from indirect
inferences, preserve uncertainty, and cite one concrete utterance for every
memory. Do not treat assistant claims as user facts. Role-play, quoted text,
hypothetical scenarios, and one-off task content should not become user facts
unless the dialogue itself supports a genuine user attribute or recurring
preference. Return only JSON matching the supplied schema."""

ANNOTATION_USER_TEMPLATE = """Conversation (zero-indexed utterances):
{conversation}

Memory labels must begin with one of:
{memory_labels}

Use UNMAPPED with a concise hierarchical label_suggestion when none fits.
`type` is `direct` for explicit user information and `indirect` for a supported
inference. Keep memories compact and non-duplicative; there is no fixed memory
count. Also rank all applicable dialogue intents using only these categories and
subtypes.

Intent categories:
{intent_categories}

Intent subtypes:
{intent_subtypes}
"""


def _annotation_schema(
    intent_categories: tuple[str, ...], intent_subtypes: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "name": "alpsbench_annotation_pool_record",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intents_ranked": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "intent_category": {
                                "type": "string",
                                "enum": list(intent_categories),
                            },
                            "intent_subtype": {
                                "type": "string",
                                "enum": list(intent_subtypes),
                            },
                        },
                        "required": ["intent_category", "intent_subtype"],
                        "additionalProperties": False,
                    },
                },
                "memory_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["direct", "indirect"],
                            },
                            "label": {"type": "string"},
                            "label_suggestion": {"type": "string"},
                            "value": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "evidence_turn_index": {"type": "integer"},
                            "confidence": {"type": "number"},
                            "time_scope": {"type": "string"},
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
                            "confidence",
                            "time_scope",
                            "emotion",
                            "preference_attitude",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["intents_ranked", "memory_items"],
            "additionalProperties": False,
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _normalise_role(value: Any) -> str | None:
    role = str(value or "").strip().lower()
    if role in {"user", "human"}:
        return "user"
    if role in {"assistant", "ai", "bot", "chatgpt"}:
        return "assistant"
    return None


def _normalise_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("conversation") or []
    if not isinstance(raw, list):
        return []
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = _normalise_role(item.get("role"))
        text = str(item.get("content") or "").strip()
        if role and text:
            messages.append({"role": role, "text": text})
    return messages


def _raw_message_count(row: dict[str, Any]) -> int:
    raw = row.get("conversation")
    return len(raw) if isinstance(raw, list) else 0


def _source_key(source_file: str, source_row: int, row: dict[str, Any]) -> str:
    conversation_hash = str(row.get("conversation_hash") or "").strip()
    identity = f"{conversation_hash}|{source_file}|{source_row}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _header_key(row: dict[str, Any]) -> str:
    header = row.get("header") if isinstance(row.get("header"), dict) else {}
    parts = (
        str(row.get("hashed_ip") or ""),
        str(header.get("user-agent") or ""),
        str(header.get("accept-language") or ""),
    )
    return "|".join(parts) if any(parts) else ""


def _length_bucket(length: int) -> str:
    for upper in (11, 23, 47, 95, 249):
        if length <= upper:
            lower = 6 if upper == 11 else {23: 12, 47: 24, 95: 48, 249: 96}[upper]
            return f"{lower:03d}-{upper:03d}"
    raise ValueError(length)


def _language_bucket(language: str) -> str:
    return language if language in {"English", "Chinese", "Russian"} else "Other"


def _candidate(
    source_file: str, source_row: int, row: dict[str, Any], messages: list[dict[str, str]]
) -> dict[str, Any]:
    source_key = _source_key(source_file, source_row, row)
    return {
        "source_key": source_key,
        "source_conversation_hash": str(row.get("conversation_hash") or "") or None,
        "source_file": source_file,
        "source_row": source_row,
        "session_id": f"sess_{source_key[:12]}",
        "source_language": str(row.get("language") or "") or None,
        "source_hashed_ip": str(row.get("hashed_ip") or "") or None,
        "source_timestamp": str(row.get("timestamp") or "") or None,
        "raw_message_count": _raw_message_count(row),
        "normalized_message_count": len(messages),
        "turns": messages,
    }


@dataclass(frozen=True)
class ScanStats:
    raw_rows: int
    usable_dialogues: int
    eligible_normalized_messages: int
    eligible_raw_messages: int
    rejected_turn_range: int
    unique_hashed_ip_all: int
    unique_hashed_ip_eligible: int
    unique_user_device_all: int
    unique_user_device_eligible: int
    sample_records: int
    language_counts: dict[str, int]
    length_bucket_counts: dict[str, int]


def _offer(
    heap: list[tuple[int, str, dict[str, Any]]],
    limit: int,
    score: int,
    candidate: dict[str, Any],
) -> None:
    item = (-score, str(candidate["source_key"]), candidate)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def scan_and_sample(
    rows: Iterable[tuple[str, int, dict[str, Any]]],
    *,
    sample_size: int,
    seed: str,
) -> tuple[list[dict[str, Any]], ScanStats]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    strata = [
        (language, length)
        for language in ("English", "Chinese", "Russian", "Other")
        for length in ("006-011", "012-023", "024-047", "048-095", "096-249")
    ]
    per_stratum = math.ceil(sample_size / len(strata))
    stratum_heaps: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = {
        value: [] for value in strata
    }
    global_heap: list[tuple[int, str, dict[str, Any]]] = []
    raw_rows = usable = eligible = eligible_raw = rejected = 0
    all_ips: set[str] = set()
    eligible_ips: set[str] = set()
    all_devices: set[str] = set()
    eligible_devices: set[str] = set()
    languages: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    for source_file, source_row, row in rows:
        raw_rows += 1
        ip = str(row.get("hashed_ip") or "")
        device = _header_key(row)
        if ip:
            all_ips.add(ip)
        if device:
            all_devices.add(device)
        messages = _normalise_messages(row)
        if not messages:
            continue
        usable += 1
        raw_count = _raw_message_count(row)
        eligible_raw += int(MIN_TURNS <= raw_count <= MAX_TURNS)
        if not MIN_TURNS <= len(messages) <= MAX_TURNS:
            rejected += 1
            continue
        if not any(message["role"] == "user" for message in messages):
            rejected += 1
            continue
        eligible += 1
        if ip:
            eligible_ips.add(ip)
        if device:
            eligible_devices.add(device)
        language = str(row.get("language") or "Unknown")
        length_bucket = _length_bucket(len(messages))
        languages[language] += 1
        lengths[length_bucket] += 1
        record = _candidate(source_file, source_row, row, messages)
        score = int.from_bytes(
            hashlib.sha256(f"{seed}|{record['source_key']}".encode("utf-8")).digest(),
            "big",
        )
        _offer(global_heap, sample_size, score, record)
        _offer(
            stratum_heaps[(_language_bucket(language), length_bucket)],
            per_stratum,
            score,
            record,
        )
    selected: dict[str, dict[str, Any]] = {}
    for heap in stratum_heaps.values():
        for _, _, record in heap:
            selected[str(record["source_key"])] = record
    if len(selected) < sample_size:
        for _, _, record in sorted(global_heap, reverse=True):
            selected.setdefault(str(record["source_key"]), record)
            if len(selected) >= sample_size:
                break
    sample = sorted(selected.values(), key=lambda value: str(value["source_key"]))[
        :sample_size
    ]
    stats = ScanStats(
        raw_rows=raw_rows,
        usable_dialogues=usable,
        eligible_normalized_messages=eligible,
        eligible_raw_messages=eligible_raw,
        rejected_turn_range=rejected,
        unique_hashed_ip_all=len(all_ips),
        unique_hashed_ip_eligible=len(eligible_ips),
        unique_user_device_all=len(all_devices),
        unique_user_device_eligible=len(eligible_devices),
        sample_records=len(sample),
        language_counts=dict(languages.most_common()),
        length_bucket_counts=dict(sorted(lengths.items())),
    )
    return sample, stats


def load_intent_taxonomy(
    stats_path: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if stats_path is None:
        return DEFAULT_INTENT_CATEGORIES, DEFAULT_INTENT_SUBTYPES
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    categories = tuple(data.get("intent_category", {}).keys())
    subtypes = tuple(data.get("intent_subtype", {}).keys())
    return categories or DEFAULT_INTENT_CATEGORIES, subtypes or DEFAULT_INTENT_SUBTYPES


def _format_conversation(turns: list[dict[str, str]]) -> str:
    return "\n".join(
        f"[{index}] {turn['role']}: {turn['text']}"
        for index, turn in enumerate(turns)
    )


def _valid_label(label: str) -> bool:
    return label.split("/", 1)[0] in MEMORY_TOP_LEVEL_LABELS


class AnnotationPoolExtractor:
    def __init__(
        self,
        backend: ChatBackend,
        intent_categories: tuple[str, ...],
        intent_subtypes: tuple[str, ...],
    ) -> None:
        if backend.model != PAPER_MEMORY_MODEL:
            raise ValueError(f"expected {PAPER_MEMORY_MODEL}; got {backend.model}")
        self.backend = backend
        self.intent_categories = intent_categories
        self.intent_subtypes = intent_subtypes
        self.schema = _annotation_schema(intent_categories, intent_subtypes)

    def extract(self, source: dict[str, Any]) -> dict[str, Any]:
        turns = source["turns"]
        prompt = ANNOTATION_USER_TEMPLATE.format(
            conversation=_format_conversation(turns),
            memory_labels=", ".join(MEMORY_TOP_LEVEL_LABELS),
            intent_categories=", ".join(self.intent_categories),
            intent_subtypes=", ".join(self.intent_subtypes),
        )
        result = self.backend.chat(
            ANNOTATION_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=3_000,
            response_schema=self.schema,
        )
        parsed = _parse_json_object(result.content)
        memories: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for ordinal, item in enumerate(parsed.get("memory_items") or [], 1):
            if not isinstance(item, dict):
                continue
            value = str(item.get("value") or "").strip()
            label = str(item.get("label") or "UNMAPPED").strip()
            memory_type = str(item.get("type") or "").strip()
            if not value or memory_type not in {"direct", "indirect"}:
                continue
            if not _valid_label(label):
                label = "UNMAPPED"
            index = int(item.get("evidence_turn_index") or 0)
            index = max(0, min(index, len(turns) - 1))
            key = (label.casefold(), re.sub(r"\s+", " ", value.casefold()))
            if key in seen:
                continue
            seen.add(key)
            confidence = float(item.get("confidence") or 0.0)
            memories.append(
                {
                    "memory_id": f"m{ordinal}",
                    "type": memory_type,
                    "label": label,
                    "label_suggestion": str(item.get("label_suggestion") or "") or None,
                    "value": value,
                    "reasoning": str(item.get("reasoning") or ""),
                    "evidence": {
                        "session_id": source["session_id"],
                        "utterance_index": index,
                        "text": turns[index]["text"],
                    },
                    "confidence": min(max(confidence, 0.0), 1.0),
                    "time_scope": str(item.get("time_scope") or "unknown"),
                    "emotion": str(item.get("emotion") or "") or None,
                    "preference_attitude": str(
                        item.get("preference_attitude") or ""
                    )
                    or None,
                    "updated_at": source.get("source_timestamp"),
                }
            )
        intents: list[dict[str, str]] = []
        intent_seen: set[tuple[str, str]] = set()
        for item in parsed.get("intents_ranked") or []:
            if not isinstance(item, dict):
                continue
            category = str(item.get("intent_category") or "")
            subtype = str(item.get("intent_subtype") or "")
            pair = (category, subtype)
            if (
                category in self.intent_categories
                and subtype in self.intent_subtypes
                and pair not in intent_seen
            ):
                intent_seen.add(pair)
                intents.append(
                    {"intent_category": category, "intent_subtype": subtype}
                )
        return {
            "line_index": source["source_row"],
            "sessions": [
                {
                    "session_id": source["session_id"],
                    "started_at": None,
                    "ended_at": source.get("source_timestamp"),
                    "turns": [
                        {"utterance_index": index, **turn}
                        for index, turn in enumerate(turns)
                    ],
                }
            ],
            "dialogue": turns,
            "intents_ranked": intents,
            "memory_items": memories,
            "reconstruction_metadata": {
                "protocol": PROTOCOL,
                "source_repository": WILDCHAT_REPOSITORY,
                "source_revision": WILDCHAT_REVISION,
                "source_key": source["source_key"],
                "memory_and_intent_model": self.backend.model,
                "memory_prompt_sha256": prompt_hash(ANNOTATION_SYSTEM_PROMPT),
                "schema_sha256": prompt_hash(json.dumps(self.schema, sort_keys=True)),
                "manual_verification": False,
            },
        }


def _bucket_name(record: dict[str, Any]) -> str:
    memories = record.get("memory_items") or []
    if not memories:
        return "NO_MEMORY"
    memory = memories[0]
    label = str(memory.get("label") or "UNMAPPED")
    if label == "UNMAPPED":
        label = str(memory.get("label_suggestion") or "UNMAPPED")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "__", label).strip("_.")
    return cleaned[:160] or "UNMAPPED"


def write_by_label(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_bucket_name(record), []).append(record)
    destination = output_dir / "by_label_json"
    destination.mkdir(parents=True, exist_ok=True)
    for name, items in sorted(grouped.items()):
        _atomic_json(destination / f"{name}.json", items)
    return {name: len(items) for name, items in sorted(grouped.items())}


def annotate_sample(
    sample: list[dict[str, Any]], extractor: AnnotationPoolExtractor
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source in sample:
        try:
            records.append(extractor.extract(source))
        except Exception as exc:
            failures.append(
                {"source_key": str(source["source_key"]), "error": str(exc)}
            )
    return records, failures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=240)
    parser.add_argument("--sample-seed", default="personaemp-annotation-pool-v1")
    parser.add_argument("--intent-stats", type=Path)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--env-prefix", default="PERSONAEMP_MEMORY")
    return parser


def main() -> int:
    args = _parser().parse_args()
    snapshot = args.snapshot_dir.resolve()
    output = args.output_dir.resolve()
    sample, stats = scan_and_sample(
        iter_snapshot_rows(snapshot),
        sample_size=args.sample_size,
        seed=args.sample_seed,
    )
    _write_jsonl(output / "stages" / "annotation_sample.jsonl", sample)
    _atomic_json(output / "quality" / "source_scan_stats.json", asdict(stats))
    manifest: dict[str, Any] = {
        "created_at": _utc_now(),
        "protocol": PROTOCOL,
        "source": {
            "repository": WILDCHAT_REPOSITORY,
            "revision": WILDCHAT_REVISION,
            "snapshot_dir": str(snapshot),
            "turn_range_normalized_messages": [MIN_TURNS, MAX_TURNS],
            "language_filter": None,
            "conversation_merging": False,
        },
        "sample": {"size": args.sample_size, "seed": args.sample_seed},
        "source_stats": asdict(stats),
        "annotation_status": "not_requested",
        "table1_direct_comparison_allowed": False,
    }
    if args.annotate:
        categories, subtypes = load_intent_taxonomy(args.intent_stats)
        backend = OpenAICompatibleChatBackend.from_env(args.env_prefix)
        extractor = AnnotationPoolExtractor(backend, categories, subtypes)
        records, failures = annotate_sample(sample, extractor)
        _write_jsonl(output / "stages" / "annotation_records.jsonl", records)
        _write_jsonl(output / "stages" / "annotation_failures.jsonl", failures)
        buckets = write_by_label(output, records)
        memory_items = sum(len(record.get("memory_items") or []) for record in records)
        intents = sum(len(record.get("intents_ranked") or []) for record in records)
        manifest.update(
            {
                "annotation_status": "complete" if not failures else "incomplete",
                "annotation": {
                    "model": backend.model,
                    "attempted": len(sample),
                    "succeeded": len(records),
                    "failed": len(failures),
                    "memory_items": memory_items,
                    "intents": intents,
                    "by_label_files": len(buckets),
                    "by_label_records": sum(buckets.values()),
                    "manual_verification": False,
                    "intent_generation_is_reconstructed": True,
                    "by_label_bucket_rule_is_reconstructed": True,
                },
            }
        )
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

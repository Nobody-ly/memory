from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class PersonaEmpDatasetError(ValueError):
    """Raised when a file does not match the public PersonaEmp schema."""


def _require_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PersonaEmpDatasetError(f"{field} must be a non-empty string")
    return text


def _persona_text(value: Any) -> str:
    if isinstance(value, dict):
        profile = value.get("persona_profile")
        if profile:
            return str(profile).strip()
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "").strip()


def _memory_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or "").strip()
    return str(value or "").strip()


def _scenario_text(query: dict[str, Any]) -> str:
    situation = query.get("situation")
    if isinstance(situation, dict):
        return str(situation.get("situation") or "").strip()
    return str(situation or "").strip()


def canonical_category(value: Any) -> str:
    text = str(value or "unknown").strip()
    aliases = {
        "hei": "High-EQ Interaction",
        "heq": "High-EQ Interaction",
        "high-eq interaction": "High-EQ Interaction",
        "emotional support": "Emotional Support",
        "es": "Emotional Support",
        "social strategy": "Social Strategy",
        "ss": "Social Strategy",
    }
    return aliases.get(text.lower(), text)


def _relevant_memory_indices(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    indices: list[int] = []
    for item in value:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index >= 1 and index not in indices:
            indices.append(index)
    return tuple(indices)


@dataclass(frozen=True)
class PersonaEmpSample:
    session_id: str
    query_id: str
    query: str
    scenario: str
    category: str
    persona_text: str
    memory_items: tuple[str, ...]
    conversation: tuple[dict[str, str], ...]
    relevant_memory_indices: tuple[int, ...]
    session_index: int
    query_index: int

    @property
    def internal_key(self) -> str:
        """Return the occurrence identity; public IDs are not globally unique."""

        return (
            f"{self.session_index}:{self.query_index}:"
            f"{self.session_id}:{self.query_id}"
        )

    @property
    def sample_key(self) -> str:
        return f"{self.session_id}:{self.query_id}"


@dataclass(frozen=True)
class PersonaEmpDataset:
    path: Path
    fingerprint: str
    raw_sessions: tuple[dict[str, Any], ...]
    samples: tuple[PersonaEmpSample, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        allow_duplicate_display_ids: bool = False,
    ) -> "PersonaEmpDataset":
        dataset_path = Path(path).resolve()
        try:
            raw_bytes = dataset_path.read_bytes()
        except OSError as exc:
            raise PersonaEmpDatasetError(f"cannot read dataset: {dataset_path}") from exc

        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PersonaEmpDatasetError(f"invalid UTF-8 JSON: {dataset_path}") from exc

        if not isinstance(data, list) or not data:
            raise PersonaEmpDatasetError("dataset root must be a non-empty JSON list")

        samples: list[PersonaEmpSample] = []
        seen_session_ids: set[str] = set()
        seen_query_ids: set[str] = set()

        for session_index, session in enumerate(data):
            if not isinstance(session, dict):
                raise PersonaEmpDatasetError(f"session[{session_index}] must be an object")

            session_id = _require_text(
                session.get("session_id") or session.get("original_sid"),
                f"session[{session_index}].session_id",
            )
            if not allow_duplicate_display_ids and session_id in seen_session_ids:
                raise PersonaEmpDatasetError(f"duplicate session_id: {session_id}")
            seen_session_ids.add(session_id)

            persona_text = _persona_text(session.get("persona"))
            if not persona_text:
                raise PersonaEmpDatasetError(f"session[{session_index}].persona is empty")

            raw_memory = session.get("extracted_memory")
            if not isinstance(raw_memory, list):
                raise PersonaEmpDatasetError(
                    f"session[{session_index}].extracted_memory must be a list"
                )
            memory_items = tuple(
                text for item in raw_memory if (text := _memory_text(item))
            )
            if not memory_items:
                raise PersonaEmpDatasetError(
                    f"session[{session_index}].extracted_memory has no usable values"
                )

            raw_conversation = session.get("conversation") or []
            if not isinstance(raw_conversation, list):
                raise PersonaEmpDatasetError(
                    f"session[{session_index}].conversation must be a list"
                )
            conversation: list[dict[str, str]] = []
            for turn_index, turn in enumerate(raw_conversation):
                if not isinstance(turn, dict):
                    raise PersonaEmpDatasetError(
                        f"session[{session_index}].conversation[{turn_index}] must be an object"
                    )
                text = str(turn.get("text") or turn.get("content") or "").strip()
                if not text:
                    continue
                conversation.append(
                    {
                        "role": str(turn.get("role") or "unknown").strip(),
                        "text": text,
                    }
                )

            queries = session.get("queries")
            if not isinstance(queries, list) or not queries:
                raise PersonaEmpDatasetError(
                    f"session[{session_index}].queries must be a non-empty list"
                )

            for query_index, query_item in enumerate(queries):
                if not isinstance(query_item, dict):
                    raise PersonaEmpDatasetError(
                        f"session[{session_index}].queries[{query_index}] must be an object"
                    )
                query_id = _require_text(
                    query_item.get("query_id"),
                    f"session[{session_index}].queries[{query_index}].query_id",
                )
                if not allow_duplicate_display_ids and query_id in seen_query_ids:
                    raise PersonaEmpDatasetError(f"duplicate query_id: {query_id}")
                seen_query_ids.add(query_id)

                query = _require_text(
                    query_item.get("query"),
                    f"session[{session_index}].queries[{query_index}].query",
                )
                scenario = _require_text(
                    _scenario_text(query_item),
                    f"session[{session_index}].queries[{query_index}].situation",
                )
                samples.append(
                    PersonaEmpSample(
                        session_id=session_id,
                        query_id=query_id,
                        query=query,
                        scenario=scenario,
                        category=canonical_category(query_item.get("category")),
                        persona_text=persona_text,
                        memory_items=memory_items,
                        conversation=tuple(conversation),
                        relevant_memory_indices=_relevant_memory_indices(
                            query_item.get("relevant_mem")
                        ),
                        session_index=session_index,
                        query_index=query_index,
                    )
                )

        return cls(
            path=dataset_path,
            fingerprint=hashlib.sha256(raw_bytes).hexdigest(),
            raw_sessions=tuple(data),
            samples=tuple(samples),
        )

    def iter_samples(self, limit: int | None = None) -> Iterator[PersonaEmpSample]:
        samples = self.samples if limit is None else self.samples[: max(limit, 0)]
        yield from samples

    def iter_balanced(
        self,
        per_category: int,
    ) -> Iterator[PersonaEmpSample]:
        if per_category < 1:
            raise ValueError("per_category must be positive")
        categories = (
            "Emotional Support",
            "High-EQ Interaction",
            "Social Strategy",
        )
        grouped = {
            category: [
                sample for sample in self.samples if sample.category == category
            ]
            for category in categories
        }
        missing = [
            category
            for category, samples in grouped.items()
            if len(samples) < per_category
        ]
        if missing:
            raise ValueError(
                "not enough samples for balanced selection: "
                + ", ".join(missing)
            )
        for index in range(per_category):
            for category in categories:
                yield grouped[category][index]

    def prediction_template(self) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for session in self.raw_sessions:
            predictions.append(
                {
                    "session_id": str(
                        session.get("session_id") or session.get("original_sid") or ""
                    ),
                    "responses": [
                        {
                            "query_id": str(query.get("query_id") or ""),
                            "response": "",
                        }
                        for query in session.get("queries", [])
                    ],
                }
            )
        return predictions

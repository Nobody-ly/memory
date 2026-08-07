from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...epistemic_decay import PROFILE_LAYERS, compute_omega
from ...profile_utils import flatten_static_profile
from ...prompts.templates_en import (
    EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
)
from .client import ChatBackend, ChatResult
from .dataset import PersonaEmpSample


PERSONAEMP_RESPONSE_SYSTEM_PROMPT = """You are a warm, empathetic conversation partner completing a single-turn personalized empathy benchmark.

Respond under all of these requirements:
1. Use the same language as the user's query.
2. Directly address the user's current need. When the user asks for advice, a decision, wording, or practical help, give at least one actionable suggestion or example phrase before any optional follow-up question.
3. Write exactly one paragraph containing 2 to 4 concise, natural sentences.
4. Validate the user's feelings when appropriate, without sounding clinical, formal, or patronizing.
5. Personalize only from the evidence provided. Do not mention memories, profiles, hidden context, or how the response was generated.
6. Do not invent user facts and do not describe yourself as an AI.
7. Ask at most one follow-up question.
8. Output only the final response."""

RESPONSE_MAX_TOKENS = 350
PROFILE_MAX_TOKENS = 6000
ALIGNMENT_MAX_TOKENS = 1800
STRUCTURED_JSON_PARSER_VERSION = "first_complete_or_close_unbalanced_v2"
RAG_ENCODER_MODEL = "intfloat/e5-base-v2"
RAG_ENCODER_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
PROFILE_RESPONSE_SCHEMA = {
    "name": "five_layer_profile",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            layer: {"type": "object", "additionalProperties": True}
            for layer in PROFILE_LAYERS
        },
        "required": list(PROFILE_LAYERS),
        "additionalProperties": False,
    },
}
ALIGNMENT_RESPONSE_SCHEMA = {
    "name": "deep_empathy_alignment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            field: {"type": "object", "additionalProperties": True}
            for field in (
                "understanding",
                "prediction",
                "exploration",
                "alignment",
                "empathy_state",
            )
        },
        "required": [
            "understanding",
            "prediction",
            "exploration",
            "alignment",
            "empathy_state",
        ],
        "additionalProperties": False,
    },
}
BASE_MODEL_USER_PROMPT = """You will be provided with memories extracted from previous dialogue.
Use them as background evidence and generate the final response.

User memory evidence:
{memory}

User query:
{query}
"""
MEMORY_SUMMARY_SYSTEM_PROMPT = """You summarize user characteristics from
long-term memory evidence for personalized conversation. Produce a concise,
flat summary of stable traits, preferences, experiences, emotional needs, and
support preferences that are grounded in the supplied memories. Do not use a
hierarchical profile, predict a future state, propose exploration, or invent
facts. Output only the summary."""
MEMORY_SUMMARY_USER_PROMPT = """User memory evidence:
{memory}
"""
MEMORY_RESPONSE_USER_PROMPT = """Generate the final response using the flat
user-characteristic summary as background evidence.

Flat user-characteristic summary:
{summary}

User query:
{query}
"""
RAG_RESPONSE_USER_PROMPT = """Generate the final response using only the three
retrieved memory items as background evidence.

Retrieved user memory evidence:
{memory}

User query:
{query}
"""
OURS_USER_PROMPT = """Generate the final response using the following evidence and derived reasoning.

User memory evidence:
{memory}

Derived five-layer user profile:
{profile}

Derived deep-empathy state:
{alignment}

User query:
{query}
"""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_reasoning(text: str) -> str:
    value = text.strip()
    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    value = _strip_reasoning(text)
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response does not contain a JSON object")
    complete_end = _first_complete_json_container_end(value, start)
    candidate = value[
        start : (complete_end + 1 if complete_end is not None else end + 1)
    ]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        repaired = _close_unbalanced_json_containers(candidate)
        if exc.pos < len(candidate) - 1 or repaired == candidate:
            raise
        parsed = json.loads(repaired)
    if not isinstance(parsed, dict):
        raise ValueError("response JSON root must be an object")
    return parsed


def _first_complete_json_container_end(
    value: str,
    start: int,
) -> int | None:
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}

    for index in range(start, len(value)):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in pairs:
            stack.append(pairs[character])
        elif character in ("}", "]"):
            if not stack or stack.pop() != character:
                return None
            if not stack:
                return index
    return None


def _close_unbalanced_json_containers(value: str) -> str:
    """Complete only unclosed JSON containers at end of an otherwise valid value."""
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}

    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in pairs:
            stack.append(pairs[character])
        elif character in ("}", "]"):
            if not stack or stack.pop() != character:
                return value

    if in_string or not stack:
        return value
    return value + "".join(reversed(stack))


def _memory_block(sample: PersonaEmpSample) -> str:
    return "\n".join(
        f"{index}. {value}" for index, value in enumerate(sample.memory_items, 1)
    )


def _profile_corpus(sample: PersonaEmpSample) -> str:
    return "\n".join(
        [
            "Extracted long-term memory evidence:",
            _memory_block(sample),
        ]
    )


def _profile_text(profile: dict[str, Any]) -> str:
    flattened = flatten_static_profile(profile)
    lines: list[str] = []
    for layer in PROFILE_LAYERS:
        fields = flattened.get(layer, {})
        if not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            if value not in (None, "", [], {}):
                lines.append(f"[{layer}] {key}: {value}")
    return "\n".join(lines) or json.dumps(
        flattened,
        ensure_ascii=False,
        sort_keys=True,
    )


def _validate_profile(profile: dict[str, Any]) -> None:
    missing = [layer for layer in PROFILE_LAYERS if layer not in profile]
    if missing:
        raise ValueError(f"profile is missing layers: {', '.join(missing)}")
    invalid = [
        layer for layer in PROFILE_LAYERS if not isinstance(profile.get(layer), dict)
    ]
    if invalid:
        raise ValueError(f"profile layers must be objects: {', '.join(invalid)}")


@dataclass(frozen=True)
class StageUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    attempts: int
    logical_calls: int

    @classmethod
    def from_result(cls, result: ChatResult) -> "StageUsage":
        return cls(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_seconds=result.latency_seconds,
            attempts=result.attempts,
            logical_calls=1,
        )

    @classmethod
    def combine(cls, results: list[ChatResult]) -> "StageUsage":
        if not results:
            raise ValueError("at least one result is required")
        return cls(
            model=results[-1].model,
            prompt_tokens=sum(result.prompt_tokens for result in results),
            completion_tokens=sum(
                result.completion_tokens for result in results
            ),
            latency_seconds=round(
                sum(result.latency_seconds for result in results),
                4,
            ),
            attempts=sum(result.attempts for result in results),
            logical_calls=len(results),
        )


@dataclass(frozen=True)
class GenerationOutput:
    response: str
    method: str
    profile_hash: str | None
    alignment_hash: str | None
    omega: float | None
    stages: dict[str, StageUsage]
    qualitative_artifacts: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "response": self.response,
            "method": self.method,
            "profile_hash": self.profile_hash,
            "alignment_hash": self.alignment_hash,
            "omega": self.omega,
            "stages": {
                name: asdict(usage) for name, usage in self.stages.items()
            },
        }
        if self.qualitative_artifacts is not None:
            record["qualitative_artifacts"] = self.qualitative_artifacts
        return record


class ProfileCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, cache_key: str) -> Path:
        return self.directory / f"{cache_key}.json"

    def load(
        self,
        cache_key: str,
    ) -> tuple[dict[str, Any], StageUsage | None] | None:
        path = self._path(cache_key)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid profile cache entry: {path}")
        if value.get("format_version") == 2:
            profile = value.get("profile")
            usage_value = value.get("generation_usage")
            if not isinstance(profile, dict):
                raise ValueError(f"invalid profile cache entry: {path}")
            usage = (
                StageUsage(**usage_value)
                if isinstance(usage_value, dict)
                else None
            )
        else:
            profile = value
            usage = None
        _validate_profile(profile)
        return profile, usage

    def save(
        self,
        cache_key: str,
        profile: dict[str, Any],
        usage: StageUsage,
    ) -> None:
        _validate_profile(profile)
        path = self._path(cache_key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "profile": profile,
                    "generation_usage": asdict(usage),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)


class JsonCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, cache_key: str) -> Path:
        return self.directory / f"{cache_key}.json"

    def load(self, cache_key: str) -> dict[str, Any] | None:
        path = self._path(cache_key)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid cache entry: {path}")
        return value

    def save(self, cache_key: str, value: dict[str, Any]) -> None:
        path = self._path(cache_key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)


class ProfileBuilder:
    def __init__(
        self,
        backend: ChatBackend,
        cache: ProfileCache,
        schema_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.schema_attempts = schema_attempts

    def cache_key(self, sample: PersonaEmpSample) -> str:
        payload = {
            "session_id": sample.session_id,
            "memory": sample.memory_items,
            "model": self.backend.model,
            "system_prompt_hash": prompt_hash(PROFILE_EXTRACTION_SYSTEM_PROMPT),
            "user_prompt_hash": prompt_hash(PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE),
            "response_schema": PROFILE_RESPONSE_SCHEMA,
            "max_tokens": PROFILE_MAX_TOKENS,
            "parser_version": STRUCTURED_JSON_PARSER_VERSION,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def build(
        self,
        sample: PersonaEmpSample,
    ) -> tuple[dict[str, Any], StageUsage | None]:
        cache_key = self.cache_key(sample)
        cached = self.cache.load(cache_key)
        if cached is not None:
            profile, _generation_usage = cached
            return profile, None

        user_prompt = PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
            user_name="the user",
            corpus=_profile_corpus(sample),
        )
        last_error: Exception | None = None
        logical_results: list[ChatResult] = []
        for _ in range(self.schema_attempts):
            result = self.backend.chat(
                PROFILE_EXTRACTION_SYSTEM_PROMPT.format(user_name="the user"),
                user_prompt,
                temperature=0.2,
                max_tokens=PROFILE_MAX_TOKENS,
                response_schema=PROFILE_RESPONSE_SCHEMA,
            )
            logical_results.append(result)
            try:
                profile = _parse_json_object(result.content)
                _validate_profile(profile)
                usage = StageUsage.combine(logical_results)
                self.cache.save(cache_key, profile, usage)
                return profile, usage
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        raise RuntimeError(
            f"profile extraction failed schema validation after "
            f"{self.schema_attempts} logical attempts: {last_error}"
        )


class BaseModelGenerator:
    method = "base_model"

    def __init__(self, backend: ChatBackend) -> None:
        self.backend = backend

    def generate(self, sample: PersonaEmpSample) -> GenerationOutput:
        result = self.backend.chat(
            PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
            BASE_MODEL_USER_PROMPT.format(
                memory=_memory_block(sample),
                query=sample.query,
            ),
            temperature=0.6,
            max_tokens=RESPONSE_MAX_TOKENS,
        )
        return GenerationOutput(
            response=_strip_reasoning(result.content),
            method=self.method,
            profile_hash=None,
            alignment_hash=None,
            omega=None,
            stages={"response": StageUsage.from_result(result)},
        )


class MemorySummaryBuilder:
    def __init__(self, backend: ChatBackend, cache: JsonCache) -> None:
        self.backend = backend
        self.cache = cache

    def cache_key(self, sample: PersonaEmpSample) -> str:
        value = {
            "memory": sample.memory_items,
            "model": self.backend.model,
            "system_prompt": prompt_hash(MEMORY_SUMMARY_SYSTEM_PROMPT),
            "user_prompt": prompt_hash(MEMORY_SUMMARY_USER_PROMPT),
        }
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def build(self, sample: PersonaEmpSample) -> tuple[str, StageUsage | None]:
        cache_key = self.cache_key(sample)
        cached = self.cache.load(cache_key)
        if cached is not None:
            summary = str(cached.get("summary") or "").strip()
            if not summary:
                raise ValueError("cached memory summary is empty")
            return summary, None
        result = self.backend.chat(
            MEMORY_SUMMARY_SYSTEM_PROMPT,
            MEMORY_SUMMARY_USER_PROMPT.format(memory=_memory_block(sample)),
            temperature=0.2,
            max_tokens=900,
        )
        summary = _strip_reasoning(result.content)
        if not summary:
            raise ValueError("memory summary is empty")
        usage = StageUsage.from_result(result)
        self.cache.save(
            cache_key,
            {
                "format_version": 1,
                "summary": summary,
                "generation_usage": asdict(usage),
            },
        )
        return summary, usage


class MemoryGenerator:
    method = "memory"

    def __init__(
        self,
        backend: ChatBackend,
        summary_builder: MemorySummaryBuilder,
    ) -> None:
        self.backend = backend
        self.summary_builder = summary_builder

    def generate(self, sample: PersonaEmpSample) -> GenerationOutput:
        summary, summary_usage = self.summary_builder.build(sample)
        result = self.backend.chat(
            PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
            MEMORY_RESPONSE_USER_PROMPT.format(
                summary=summary,
                query=sample.query,
            ),
            temperature=0.6,
            max_tokens=RESPONSE_MAX_TOKENS,
        )
        stages = {"response": StageUsage.from_result(result)}
        if summary_usage is not None:
            stages["summary"] = summary_usage
        return GenerationOutput(
            response=_strip_reasoning(result.content),
            method=self.method,
            profile_hash=None,
            alignment_hash=None,
            omega=None,
            stages=stages,
            qualitative_artifacts={
                "flat_memory_summary_sha256": hashlib.sha256(
                    summary.encode("utf-8")
                ).hexdigest()
            },
        )


class SentenceTransformerEncoder:
    def __init__(
        self,
        model_name: str = RAG_ENCODER_MODEL,
        revision: str = RAG_ENCODER_REVISION,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the RAG baseline"
            ) from exc
        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(model_name, revision=revision)

    def encode_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        )[0]
        return [float(value) for value in vector]

    def encode_memories(self, memories: tuple[str, ...]) -> list[list[float]]:
        vectors = self.model.encode(
            [f"passage: {memory}" for memory in memories],
            normalize_embeddings=True,
        )
        return [[float(value) for value in vector] for vector in vectors]


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right))


class RAGRetriever:
    def __init__(
        self,
        encoder: Any,
        cache: JsonCache,
        top_k: int = 3,
    ) -> None:
        if top_k != 3:
            raise ValueError("PersonaEmp RAG must retrieve exactly three memories")
        self.encoder = encoder
        self.cache = cache
        self.top_k = top_k

    def _memory_key(self, sample: PersonaEmpSample) -> str:
        value = {
            "memory": sample.memory_items,
            "encoder": self.encoder.model_name,
            "encoder_revision": getattr(self.encoder, "revision", None),
        }
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _memory_vectors(self, sample: PersonaEmpSample) -> list[list[float]]:
        key = self._memory_key(sample)
        cached = self.cache.load(key)
        if cached is not None:
            vectors = cached.get("vectors")
            if isinstance(vectors, list):
                return [[float(value) for value in row] for row in vectors]
        vectors = self.encoder.encode_memories(sample.memory_items)
        self.cache.save(
            key,
            {
                "format_version": 1,
                "encoder": self.encoder.model_name,
                "encoder_revision": getattr(self.encoder, "revision", None),
                "vectors": vectors,
            },
        )
        return vectors

    def retrieve(
        self,
        sample: PersonaEmpSample,
    ) -> tuple[tuple[int, str, float], ...]:
        memory_vectors = self._memory_vectors(sample)
        query_vector = self.encoder.encode_query(sample.query)
        ranked = sorted(
            (
                (index + 1, memory, _dot(query_vector, vector))
                for index, (memory, vector) in enumerate(
                    zip(sample.memory_items, memory_vectors)
                )
            ),
            key=lambda value: (-value[2], value[0]),
        )
        return tuple(ranked[: min(self.top_k, len(ranked))])


class RAGGenerator:
    method = "rag"

    def __init__(self, backend: ChatBackend, retriever: RAGRetriever) -> None:
        self.backend = backend
        self.retriever = retriever

    def generate(self, sample: PersonaEmpSample) -> GenerationOutput:
        retrieved = self.retriever.retrieve(sample)
        memory = "\n".join(
            f"{rank}. {text}" for rank, (_index, text, _score) in enumerate(
                retrieved,
                1,
            )
        )
        result = self.backend.chat(
            PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
            RAG_RESPONSE_USER_PROMPT.format(
                memory=memory,
                query=sample.query,
            ),
            temperature=0.6,
            max_tokens=RESPONSE_MAX_TOKENS,
        )
        retrieved_indices = [index for index, _text, _score in retrieved]
        relevant = set(sample.relevant_memory_indices)
        recall = (
            len(relevant.intersection(retrieved_indices)) / len(relevant)
            if relevant
            else None
        )
        return GenerationOutput(
            response=_strip_reasoning(result.content),
            method=self.method,
            profile_hash=None,
            alignment_hash=None,
            omega=None,
            stages={"response": StageUsage.from_result(result)},
            qualitative_artifacts={
                "retrieved_memory_indices": retrieved_indices,
                "retrieval_scores": [
                    round(score, 8) for _index, _text, score in retrieved
                ],
                "recall_at_3": recall,
            },
        )


class DeepEmpathyGenerator:
    method = "ours"

    def __init__(
        self,
        backend: ChatBackend,
        profile_builder: ProfileBuilder,
        schema_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.profile_builder = profile_builder
        self.schema_attempts = schema_attempts

    def _alignment(
        self,
        sample: PersonaEmpSample,
        profile: dict[str, Any],
        omega: float,
    ) -> tuple[dict[str, Any], StageUsage]:
        user_prompt = EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE.format(
            recent_context=_memory_block(sample),
            user_message=sample.query,
            user_profile=json.dumps(
                flatten_static_profile(profile),
                ensure_ascii=False,
            ),
            agent_persona="{}",
            current_state="{}",
            epistemic_omega=omega,
        )

        last_error: Exception | None = None
        logical_results: list[ChatResult] = []
        for _ in range(self.schema_attempts):
            result = self.backend.chat(
                EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.2,
                max_tokens=ALIGNMENT_MAX_TOKENS,
                response_schema=ALIGNMENT_RESPONSE_SCHEMA,
            )
            logical_results.append(result)
            try:
                alignment = _parse_json_object(result.content)
                if not isinstance(alignment.get("empathy_state"), dict):
                    raise ValueError("alignment.empathy_state must be an object")
                if not isinstance(alignment.get("prediction"), dict):
                    raise ValueError("alignment.prediction must be an object")
                if not isinstance(alignment.get("exploration"), dict):
                    raise ValueError("alignment.exploration must be an object")
                return alignment, StageUsage.combine(logical_results)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        raise RuntimeError(
            f"empathy alignment failed schema validation after "
            f"{self.schema_attempts} logical attempts: {last_error}"
        )

    def generate(self, sample: PersonaEmpSample) -> GenerationOutput:
        profile, profile_usage = self.profile_builder.build(sample)
        interaction_count = len(sample.memory_items)
        omega = compute_omega(interaction_count, profile)
        alignment, alignment_usage = self._alignment(sample, profile, omega)

        response_prompt = OURS_USER_PROMPT.format(
            memory=_memory_block(sample),
            profile=_profile_text(profile),
            alignment=json.dumps(alignment, ensure_ascii=False),
            query=sample.query,
        )
        response_result = self.backend.chat(
            PERSONAEMP_RESPONSE_SYSTEM_PROMPT,
            response_prompt,
            temperature=0.6,
            max_tokens=RESPONSE_MAX_TOKENS,
        )

        stages = {
            "alignment": alignment_usage,
            "response": StageUsage.from_result(response_result),
        }
        if profile_usage is not None:
            stages["profile"] = profile_usage

        profile_hash = hashlib.sha256(
            json.dumps(
                profile,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        alignment_hash = hashlib.sha256(
            json.dumps(
                alignment,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return GenerationOutput(
            response=_strip_reasoning(response_result.content),
            method=self.method,
            profile_hash=profile_hash,
            alignment_hash=alignment_hash,
            omega=omega,
            stages=stages,
            qualitative_artifacts={
                "five_layer_profile": flatten_static_profile(profile),
                "understanding": alignment.get("understanding", {}),
                "prediction": alignment.get("prediction", {}),
                "exploration": alignment.get("exploration", {}),
                "empathy_state": alignment.get("empathy_state", {}),
            },
        )

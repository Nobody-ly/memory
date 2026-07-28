from __future__ import annotations

import configparser
import json
import os
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient

load_dotenv()


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) else value


from .prompts.templates import (
    MID_TERM_MEMORY_SYSTEM_PROMPT,
    MID_TERM_MEMORY_USER_PROMPT_TEMPLATE,
    LONG_TERM_MEMORY_SYSTEM_PROMPT,
    LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE,
)
from .logger import logger
from .utils import parse_json

_DEFAULT_EMBEDDING_DIM = 1536


class MemoryOSLocal:

    def __init__(
        self,
        collection_name: str = "memoryos_local",
        persist_path: str = "./data/milvus_memory.db",
        config_path: str = "config.ini",
        embedding_model_name: Optional[str] = None,
    ):
        self.collection_name = collection_name
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")

        api_config = config["API"]
        self.embedding_model_name = embedding_model_name or api_config.get(
            "embedding_model", fallback="text-embedding-v4"
        )
        self.embedding_dimension = api_config.getint(
            "embedding_dimension", fallback=_DEFAULT_EMBEDDING_DIM
        )
        self.embedding_client = OpenAI(
            api_key=_env("API_KEY"),
            base_url=_env("BASE_URL"),
        )

        milvus_config = config["Milvus"] if config.has_section("Milvus") else {}
        uri = milvus_config.get("uri", "").strip() or str(persist_path)
        token = milvus_config.get("token", "").strip()
        self.client = MilvusClient(uri=uri, token=token or None)

        self._ensure_collection()

        self.short_term_memory: List[Dict[str, Any]] = []
        self.summary_prune_messages = 10
        self.long_term_trigger_summaries = 3
        self.long_term_source_summaries = 5
        self.last_mid_count = self._get_memory_count()

    def _ensure_collection(self) -> None:
        if not self.client.has_collection(self.collection_name):
            from pymilvus import DataType
            schema = self.client.create_schema(auto_id=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.embedding_dimension)
            schema.add_field("content", DataType.VARCHAR, max_length=65535, nullable=True)
            schema.add_field("memory_type", DataType.VARCHAR, max_length=32, nullable=True)
            schema.add_field("topic", DataType.VARCHAR, max_length=256, nullable=True)
            schema.add_field("importance", DataType.VARCHAR, max_length=32, nullable=True)
            schema.add_field("created_at", DataType.VARCHAR, max_length=64, nullable=True)
            schema.add_field("kind", DataType.VARCHAR, max_length=64, nullable=True)
            schema.add_field("confidence", DataType.FLOAT, nullable=True)
            schema.add_field("mid_term_count", DataType.INT64, nullable=True)
            index_params = self.client.prepare_index_params()
            index_params.add_index("vector", index_type="FLAT", metric_type="COSINE")
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
        # Preload collection into memory so first search isn't slow (lazy load otherwise)
        try:
            t0 = perf_counter()
            self.client.load_collection(collection_name=self.collection_name)
            logger.info(f"[MILVUS] collection '{self.collection_name}' loaded in {perf_counter()-t0:.3f}s")
        except Exception as e:
            logger.warning(f"[MILVUS] load_collection failed: {e}")

    # ---------- 嵌入 ----------
    def _embed_text(self, text: str) -> List[float]:
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model_name,
            input=text,
            extra_body={"dimensions": self.embedding_dimension},
        )
        return list(response.data[0].embedding)

    def _upsert_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding_text: Optional[str] = None,
    ) -> None:
        vector = self._embed_text(embedding_text or content)
        self.client.upsert(
            collection_name=self.collection_name,
            data=[
                {
                    "id": doc_id,
                    "vector": vector,
                    "content": content,
                    **metadata,
                }
            ],
        )

    # ---------- 短期记忆（内存） ----------
    def append_stm(self, role: str, content: str) -> Dict[str, Any]:
        entry = {
            "id": f"stm_{len(self.short_term_memory) + 1}_{role}_{datetime.now().timestamp()}",
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.short_term_memory.append(entry)
        return entry

    def get_recent_messages(self, limit: int = 14) -> List[Dict[str, Any]]:
        return self.short_term_memory[-limit:]

    def flush_short_term_memory(self, llm, min_messages: int = 2) -> List[str]:
        if len(self.short_term_memory) < min_messages:
            return []

        memory_id = self.build_mid_term_summary(
            llm,
            summary_source_messages=len(self.short_term_memory)
        )
        return [memory_id]

    def clear_stm(self) -> None:
        self.short_term_memory.clear()

    # ---------- 中期记忆 ----------
    def add_mid_term_summary(
        self,
        topic: str,
        summary: str,
        related_states: Optional[List[str]] = None,
        related_messages: Optional[List[str]] = None,
        importance: str = "",
    ) -> str:
        memory_id = f"mtm_{datetime.now().timestamp()}"
        self._upsert_document(
            memory_id,
            json.dumps(
                {
                    "summary": summary,
                    "related_states": list(related_states or []),
                    "related_messages": list(related_messages or []),
                },
                ensure_ascii=False,
            ),
            {
                "memory_type": "mid_term",
                "topic": topic,
                "importance": importance,
                "created_at": datetime.now().isoformat(),
            },
            embedding_text=summary,
        )
        return memory_id

    def build_mid_term_summary(self, llm, summary_source_messages: int) -> str:
        source_messages = self.short_term_memory[:summary_source_messages]
        logger.info(
            "[MEMORY_WRITE_START] type=mid_term "
            f"source_messages={len(source_messages)} stm_count={len(self.short_term_memory)}"
        )
        source_message_map = {
            m["id"]: f'{m["role"]}: {m["content"]}' for m in source_messages
        }
        summary_result = parse_json(
            llm.chat(
                MID_TERM_MEMORY_SYSTEM_PROMPT,
                MID_TERM_MEMORY_USER_PROMPT_TEMPLATE.format(
                    source_message_map=source_message_map
                ),
            )
        )
        related_messages = [
            source_message_map[mid]
            for mid in summary_result.get("related_message_ids", [])
            if mid in source_message_map
        ]
        memory_id = self.add_mid_term_summary(
            topic=summary_result.get("topic", ""),
            summary=summary_result.get("summary", ""),
            related_states=summary_result.get("related_states", []),
            related_messages=related_messages,
            importance=summary_result.get("importance", "medium"),
        )
        self.short_term_memory = self.short_term_memory[self.summary_prune_messages:]
        logger.info(
            "[MEMORY_WRITE_DONE] type=mid_term "
            f"id={memory_id} pruned={self.summary_prune_messages} "
            f"remaining_stm={len(self.short_term_memory)}"
        )
        return memory_id

    # ---------- 长期记忆 ----------
    def add_long_term_memory(
        self,
        content: str,
        memory_kind: str,
        confidence: float = 0.0,
        source_summary_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        memory_id = f"ltm_{datetime.now().timestamp()}"
        self._upsert_document(
            memory_id,
            json.dumps(
                {
                    "type": memory_kind,
                    "content": content,
                    "source_summary_ids": list(source_summary_ids or []),
                },
                ensure_ascii=False,
            ),
            {
                "memory_type": "long_term",
                "kind": memory_kind,
                "confidence": float(confidence),
                "created_at": datetime.now().isoformat(),
                **dict(metadata or {}),
            },
        )
        return memory_id

    def extract_long_term_memory(self, llm) -> Optional[str]:
        mid_terms = self._get_memories_by_type("mid_term")
        new_count = len(mid_terms) - self.last_mid_count
        if len(mid_terms) < 3 or new_count < self.long_term_trigger_summaries:
            return None

        source_mid_terms = mid_terms[-self.long_term_source_summaries:]
        logger.info(
            "[MEMORY_WRITE_START] type=long_term "
            f"source_mid_terms={len(source_mid_terms)} total_mid_terms={len(mid_terms)}"
        )
        result = parse_json(
            llm.chat(
                LONG_TERM_MEMORY_SYSTEM_PROMPT,
                LONG_TERM_MEMORY_USER_PROMPT_TEMPLATE.format(
                    mid_term_summaries=json.dumps(
                        source_mid_terms, ensure_ascii=False, indent=2
                    )
                ),
            )
        )
        self.last_mid_count = len(mid_terms)
        if not result.get("content"):
            return None
        memory_id = self.add_long_term_memory(
            content=result.get("content", ""),
            memory_kind=result.get("type", ""),
            confidence=result.get("confidence", 0.0),
            source_summary_ids=[m["id"] for m in source_mid_terms],
            metadata={"mid_term_count": self.last_mid_count},
        )
        logger.info(f"[MEMORY_WRITE_DONE] type=long_term id={memory_id}")
        return memory_id

    # ---------- 检索 ----------
    def search_memories(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filter_expr = f'memory_type == "{memory_type}"' if memory_type else ""

        results = self.client.search(
            collection_name=self.collection_name,
            data=[self._embed_text(query)],
            limit=top_k,
            filter=filter_expr or None,
            output_fields=["content", "memory_type", "topic", "importance",
                           "created_at", "kind", "confidence"],
        )
        
        memories = []
        for hit in results[0]:
            memories.append(
                {
                    "id": hit["id"],
                    "content": hit["entity"].get("content", ""),
                    "metadata": {k: v for k, v in hit["entity"].items() if k != "content"},
                    "similarity_score": hit["distance"],
                }
            )
        return memories

    def retrieve_relevant_memory(
        self,
        user_input: str,
        recent_limit: int = 6,
        mid_term_limit: int = 3,
    ) -> Dict[str, Any]:
        mid_term_summaries = self.search_memories(user_input, top_k=mid_term_limit, memory_type="mid_term")
        return {
            "recent_messages": self.get_recent_messages(limit=recent_limit),
            "mid_term_summaries": mid_term_summaries
        }

    def get_memories_by_ids(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        if not memory_ids:
            return []

        results: List[Dict[str, Any]] = []
        rows = self.client.get(
            collection_name=self.collection_name,
            ids=memory_ids,
            output_fields=["content", "memory_type", "kind", "confidence",
                            "created_at", "mid_term_count"],
        )
        for row in rows:
            rid = row["id"]
            results.append(
                {
                    "id": rid,
                    "content": row.get("content", ""),
                    "metadata": {k: v for k, v in row.items() if k not in ("id", "content")},
                }
            )

        result_map = {item["id"]: item for item in results}
        return [result_map[mid] for mid in memory_ids if mid in result_map]

    def _get_memories_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f'memory_type == "{memory_type}"',
            output_fields=["content", "memory_type", "topic", "importance",
                           "created_at", "kind", "confidence", "mid_term_count"],
        )
        memories = [
            {
                "id": row["id"],
                "content": row.get("content", ""),
                "metadata": {k: v for k, v in row.items() if k not in ("id", "content")},
            }
            for row in rows
        ]
        return sorted(memories, key=lambda x: x["metadata"].get("created_at", ""))

    def _get_memory_count(self) -> int:
        long_terms = self._get_memories_by_type("mid_term")
        counts = [
            int(m["metadata"].get("mid_term_count", 0))
            for m in long_terms
            if m["metadata"].get("mid_term_count") is not None
        ]
        return max(counts, default=0)

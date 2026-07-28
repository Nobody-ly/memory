from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .logger import logger
from .profile_schema import PROFILE_FIELDS, PROFILE_LAYERS, normalize_bare_profile
from .utils import load_json
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """你是陪伴智能体的用户画像更新器。任务只更新固定五层用户画像。

输入会给你两部分：
1. current_profile：当前已保存的长期画像。它只用于理解既有内容，不是本次更新的证据。
2. raw_dialogue_batch：最新、尚未处理的一批对话。每条包含 user 与 assistant；只有 user 内容可以支持本次更新。

重要边界：
- assistant 只用于补足 user 碎片的对话上下文，不得作为用户事实或画像证据。
- 不要评估、改写或回复用户；不要改变主对话。
- 只输出服务端 JSON Schema 所要求的 JSON，不要 Markdown 或解释。
- 每层必须输出 summary 和三个固定字段。没有可靠更新时返回 null；不要为了填满字段猜测。
- summary 是字符串；字段值是字符串数组。文字写成对这个人的直接、具体、长期判断，不写观察报告或推理过程。
- 旧画像中未被本批可靠信息影响的内容必须保留：如果更新某个数组字段，返回该字段更新后的完整数组，包含仍然成立的旧条目和本批新增/纠正后的条目。
- 仅在本批出现明确纠正或稳定变化时替换已有字段；不要因一条模糊或瞬时表达删除长期内容。

噪声与无更新：
- 语音误收音、环境声被转写成的无意义词串、重复乱码、纯寒暄、礼貌回执、无上下文碎片、测试或要求生成画像的指令，都不是画像证据。
- 不要因为消息短就忽略。"我辞职了"、"我很难受"、"不想去了"等短句仍可能有意义，应结合本批上下文谨慎判断。
- 只有能稳定归属到下列固定字段的信息才更新；普通闲聊、一次性事件、无法确认的收音和无意义内容应全部返回 null。这种无更新是正常结果。

固定字段语义：
- core：values（价值观），motivations（持续驱动力），long_term_goals（近年长期目标）。
- regulation：stress_response（压力反应），emotion_regulation（情绪调节），conflict_style（冲突处理）。
- cognition：thinking_style（思维方式），decision_style（决策方式），beliefs（稳定信念）。
- identity：self_identity（心理自我认识），social_identity（社会身份），life_context（人生阶段/处境）。
- behavior：interaction_style（交流风格），habits（长期行为习惯），preferences（稳定偏好）。

写作要求：
- summary 用一到两句，概括该层已存在且本批支持的稳定认识。
- 数组条目使用简洁的完整判断句；去除重复，不造字段，不写置信度、证据 ID、时间戳或元数据。
- 禁止“本次对话”“用户表示”“可见其”“最终画像”“测试”等观察者、过程或测试措辞。
"""

META_LANGUAGE_MARKERS = (
    "最终画像", "本次对话", "这次对话", "本批对话", "本批消息", "用户表示",
    "用户自称", "用户说自己", "总结来说", "画像呈现", "呈现为", "收敛为",
    "人格底色", "核心底色", "作为测试", "测试中",
)


class ProfileUpdateError(ValueError):
    pass


def _reject_meta_language(value: str, path: str) -> None:
    for marker in META_LANGUAGE_MARKERS:
        if marker in value:
            raise ProfileUpdateError(f"{path} contains report-style wording: {marker}")
    if value.lstrip().startswith(("他", "她", "对方")):
        raise ProfileUpdateError(f"{path} must be a direct profile statement")


def _nullable_string_array_schema() -> Dict[str, Any]:
    return {
        "anyOf": [
            {"type": "null"},
            {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        ]
    }


def build_profile_response_format() -> Dict[str, Any]:
    """Strict fixed-field response schema. Evidence is not part of model output."""
    layer_properties: Dict[str, Any] = {}
    for layer in PROFILE_LAYERS:
        layer_properties[layer] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"anyOf": [{"type": "null"}, {"type": "string", "minLength": 1}]},
                **{field: _nullable_string_array_schema() for field in PROFILE_FIELDS[layer]},
            },
            "required": ["summary", *PROFILE_FIELDS[layer]],
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "profile_patch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "layers": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": layer_properties,
                        "required": list(PROFILE_LAYERS),
                    }
                },
                "required": ["layers"],
            },
        },
    }


def _atomic_save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@dataclass(frozen=True)
class RawDialogueTurn:
    message_id: str
    user: str
    assistant: str
    created_at: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "message_id": self.message_id,
            "user": self.user,
            "assistant": self.assistant,
            "created_at": self.created_at,
        }


def _clean_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileUpdateError(f"{path} must be a non-empty string")
    result = value.strip()
    _reject_meta_language(result, path)
    return result


def _clean_values(value: Any, path: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ProfileUpdateError(f"{path} must be a non-empty string array")
    if len(value) > 12:
        raise ProfileUpdateError(f"{path} has too many values")
    cleaned: List[str] = []
    for index, item in enumerate(value):
        item_value = _clean_string(item, f"{path}[{index}]")
        if len(item_value) > 240:
            raise ProfileUpdateError(f"{path}[{index}] is too long")
        if item_value not in cleaned:
            cleaned.append(item_value)
    return cleaned


def validate_patch(data: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(data, Mapping) or set(data) != {"layers"}:
        raise ProfileUpdateError("patch must contain only layers")
    layers = data.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != set(PROFILE_LAYERS):
        raise ProfileUpdateError("patch must contain exactly the fixed five layers")

    result: Dict[str, Dict[str, Any]] = {}
    for layer in PROFILE_LAYERS:
        section = layers[layer]
        required = {"summary", *PROFILE_FIELDS[layer]}
        if not isinstance(section, Mapping) or set(section) != required:
            raise ProfileUpdateError(f"layers.{layer} has invalid fields")
        normalized: Dict[str, Any] = {}
        summary = section["summary"]
        if summary is not None:
            normalized["summary"] = _clean_string(summary, f"layers.{layer}.summary")
        for field in PROFILE_FIELDS[layer]:
            raw = section[field]
            if raw is not None:
                normalized[field] = _clean_values(raw, f"layers.{layer}.{field}")
        result[layer] = normalized
    return result


def merge_patch(profile: Mapping[str, Any], patch: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Apply only explicit field updates and keep every other stored value."""
    merged = normalize_bare_profile(profile)
    for layer in PROFILE_LAYERS:
        updates = patch.get(layer, {})
        for field in ("summary", *PROFILE_FIELDS[layer]):
            if field in updates:
                merged[layer][field] = copy.deepcopy(updates[field])
    return merged


class KimiProfileExtractor:
    """Independent OpenAI-compatible Profile extractor with task-local retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_attempts: int = 2,
        timeout_seconds: float = 60.0,
        client: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PROFILE_API_KEY")
        self.base_url = base_url or os.getenv("PROFILE_BASE_URL", "https://api.moonshot.cn/v1")
        self.model = model or os.getenv("PROFILE_MODEL", "kimi-k2.6")
        self.max_attempts = max(1, max_attempts)
        self.client = client
        if self.client is None and self.api_key:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProfileUpdateError("openai dependency is required for profile extraction") from exc
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout_seconds)

    @property
    def available(self) -> bool:
        return self.client is not None

    def extract(self, current_profile: Mapping[str, Any], turns: Iterable[Mapping[str, str]]) -> Dict[str, Dict[str, Any]]:
        if not self.available:
            raise ProfileUpdateError("PROFILE_API_KEY is not configured")
        turn_list = list(turns)
        payload = {
            "field_whitelist": {layer: list(PROFILE_FIELDS[layer]) for layer in PROFILE_LAYERS},
            "current_profile": normalize_bare_profile(current_profile),
            "raw_dialogue_batch": turn_list,
            "output_example": {
                "layers": {
                    layer: {"summary": None, **{field: None for field in PROFILE_FIELDS[layer]}}
                    for layer in PROFILE_LAYERS
                }
            },
        }
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            raw = ""
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.6,
                    max_tokens=3000,
                    response_format=build_profile_response_format(),
                    extra_body={"thinking": {"type": "disabled"}},
                )
                raw = (response.choices[0].message.content or "").strip()
                return validate_patch(json.loads(raw))
            except Exception as exc:
                last_error = exc
                logger.warning("[PROFILE_BATCH] attempt=%s/%s failed: %s", attempt, self.max_attempts, exc)
                if attempt < self.max_attempts:
                    if raw:
                        messages.append({"role": "assistant", "content": raw[:12000]})
                    messages.append({"role": "user", "content": "上次输出校验失败：" + str(exc)[:1200] + "。请仅返回修正后的完整 JSON。"})
        raise ProfileUpdateError(f"profile extraction failed after {self.max_attempts} attempts: {last_error}")


class ProfileBatchUpdater:
    """Persistent raw-dialogue queue. Only this Profile path performs noise judgement."""

    def __init__(
        self,
        profile_path: str,
        extractor: Optional[KimiProfileExtractor] = None,
        min_user_messages: Optional[int] = None,
        max_wait_seconds: Optional[int] = None,
        on_profile_updated: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.profile_path = Path(profile_path)
        self.queue_path = self.profile_path.with_suffix(self.profile_path.suffix + ".pending.json")
        self.extractor = extractor or KimiProfileExtractor()
        message_threshold = min_user_messages if min_user_messages is not None else int(os.getenv("PROFILE_BATCH_MESSAGES", "8"))
        wait_threshold = max_wait_seconds if max_wait_seconds is not None else int(os.getenv("PROFILE_BATCH_SECONDS", "900"))
        retry_threshold = int(os.getenv("PROFILE_BATCH_RETRY_SECONDS", "60"))
        self.min_user_messages = max(1, message_threshold)
        self.max_wait_seconds = max(1, wait_threshold)
        self.retry_seconds = max(1, retry_threshold)
        self.on_profile_updated = on_profile_updated
        self._lock = threading.RLock()
        self._process_lock = threading.Lock()
        self._running = False
        self._closed = False
        self._timer: Optional[threading.Timer] = None
        self._schedule_existing_queue()

    @staticmethod
    def _empty_queue() -> Dict[str, Any]:
        return {"first_enqueued_at": None, "turns": []}

    def _reset_invalid_queue(self, reason: str) -> Dict[str, Any]:
        logger.error(
            "[PROFILE_BATCH] invalid pending queue reset: path=%s reason=%s",
            self.queue_path,
            reason,
        )
        queue = self._empty_queue()
        self._save_queue(queue)
        return queue

    def _load_queue(self) -> Dict[str, Any]:
        if not self.queue_path.exists():
            queue = self._empty_queue()
            self._save_queue(queue)
            logger.info("[PROFILE_BATCH] initialized pending queue: %s", self.queue_path)
            return queue

        try:
            raw = self.queue_path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("[PROFILE_BATCH] failed to read pending queue: %s", self.queue_path)
            raise

        if not raw.strip():
            queue = self._empty_queue()
            self._save_queue(queue)
            logger.info("[PROFILE_BATCH] normalized blank pending queue: %s", self.queue_path)
            return queue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return self._reset_invalid_queue(f"invalid JSON: {exc}")

        if data == {}:
            queue = self._empty_queue()
            self._save_queue(queue)
            return queue
        if not isinstance(data, dict):
            return self._reset_invalid_queue("root must be a JSON object")

        turns = data.get("turns", [])
        if not isinstance(turns, list):
            return self._reset_invalid_queue("turns must be a JSON array")

        normalized_turns = []
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                return self._reset_invalid_queue(f"turns[{index}] must be an object")
            normalized = {
                "message_id": turn.get("message_id"),
                "user": turn.get("user"),
                "assistant": turn.get("assistant", ""),
                "created_at": turn.get("created_at"),
            }
            required = (
                normalized["message_id"],
                normalized["user"],
                normalized["created_at"],
            )
            if not all(isinstance(value, str) and value.strip() for value in required):
                return self._reset_invalid_queue(
                    f"turns[{index}] requires non-empty message_id, user and created_at"
                )
            if not isinstance(normalized["assistant"], str):
                return self._reset_invalid_queue(f"turns[{index}].assistant must be a string")
            try:
                datetime.fromisoformat(normalized["created_at"].replace("Z", "+00:00"))
            except ValueError:
                return self._reset_invalid_queue(f"turns[{index}].created_at is invalid")
            normalized_turns.append(normalized)

        first_enqueued_at = data.get("first_enqueued_at")
        if normalized_turns:
            if not isinstance(first_enqueued_at, str) or not first_enqueued_at.strip():
                return self._reset_invalid_queue(
                    "first_enqueued_at is required when turns are present"
                )
            try:
                datetime.fromisoformat(first_enqueued_at.replace("Z", "+00:00"))
            except ValueError:
                return self._reset_invalid_queue("first_enqueued_at is invalid")
        else:
            first_enqueued_at = None

        queue = {
            "first_enqueued_at": first_enqueued_at,
            "turns": normalized_turns,
        }
        if data != queue:
            self._save_queue(queue)
        return queue

    def _save_queue(self, queue: Mapping[str, Any]) -> None:
        _atomic_save_json(self.queue_path, queue)

    def submit_turn(self, user: str, assistant: str) -> str:
        user_text = str(user or "").strip()
        assistant_text = str(assistant or "").strip()
        if not user_text or not assistant_text:
            raise ValueError("user and assistant are required")
        now = datetime.now(timezone.utc).isoformat()
        turn = RawDialogueTurn(uuid.uuid4().hex, user_text, assistant_text, now)
        with self._lock:
            if self._closed:
                raise RuntimeError("profile batch updater is closed")
            queue = self._load_queue()
            if not queue["turns"]:
                queue["first_enqueued_at"] = now
            queue["turns"].append(turn.as_dict())
            self._save_queue(queue)
            if len(queue["turns"]) >= self.min_user_messages:
                self._start_worker_locked()
            else:
                self._schedule_timer_locked(queue)
        return turn.message_id

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cancel_timer_locked()
        # Wait for an in-flight extraction/commit before another Agent can bind
        # to the same profile and pending files.
        with self._process_lock:
            pass

    def _cancel_timer_locked(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_existing_queue(self) -> None:
        with self._lock:
            if self._closed:
                return
            queue = self._load_queue()
            if not queue["turns"]:
                self._cancel_timer_locked()
            elif len(queue["turns"]) >= self.min_user_messages:
                self._start_worker_locked()
            else:
                self._schedule_timer_locked(queue)

    def _schedule_timer_locked(self, queue: Mapping[str, Any]) -> None:
        if self._closed or not queue.get("turns"):
            self._cancel_timer_locked()
            return
        if self._timer and self._timer.is_alive():
            return
        first = queue.get("first_enqueued_at")
        elapsed = 0.0
        if isinstance(first, str):
            try:
                elapsed = max(0.0, time.time() - datetime.fromisoformat(first).timestamp())
            except ValueError:
                pass
        self._timer = threading.Timer(max(0.05, self.max_wait_seconds - elapsed), self._timer_elapsed)
        self._timer.daemon = True
        self._timer.start()

    def _schedule_retry_locked(self) -> None:
        if self._closed:
            return
        self._cancel_timer_locked()
        self._timer = threading.Timer(self.retry_seconds, self._timer_elapsed)
        self._timer.daemon = True
        self._timer.start()
        logger.warning(
            "[PROFILE_BATCH] pending batch retry scheduled in %ss: %s",
            self.retry_seconds,
            self.queue_path,
        )

    def _timer_elapsed(self) -> None:
        with self._lock:
            self._timer = None
            if self._closed:
                return
            queue = self._load_queue()
            if queue["turns"]:
                self._start_worker_locked()

    def _start_worker_locked(self) -> None:
        if self._closed:
            return
        self._cancel_timer_locked()
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        succeeded = False
        try:
            succeeded = self.process_pending()
        finally:
            with self._lock:
                self._running = False
                if self._closed:
                    self._cancel_timer_locked()
                    return
                queue = self._load_queue()
                if not queue["turns"]:
                    self._cancel_timer_locked()
                elif not succeeded:
                    self._schedule_retry_locked()
                elif len(queue["turns"]) >= self.min_user_messages:
                    self._start_worker_locked()
                else:
                    self._schedule_timer_locked(queue)

    def process_pending(self) -> bool:
        with self._process_lock:
            with self._lock:
                if self._closed:
                    return False
            return self._process_pending()

    def _process_pending(self) -> bool:
        with self._lock:
            queue = self._load_queue()
            turns = list(queue["turns"])
        if not turns:
            return False
        if not self.extractor.available:
            logger.warning("[PROFILE_BATCH] pending turns kept because PROFILE_API_KEY is not configured")
            return False
        consumed_ids = {turn["message_id"] for turn in turns}
        try:
            current = load_json(str(self.profile_path)) if self.profile_path.exists() else {}
            patch = self.extractor.extract(normalize_bare_profile(current), turns)
            merged = merge_patch(current, patch)
            _atomic_save_json(self.profile_path, merged)
        except Exception as exc:
            logger.exception("[PROFILE_BATCH] batch failed; pending turns retained: %s", exc)
            return False
        with self._lock:
            latest = self._load_queue()
            remaining = [turn for turn in latest["turns"] if turn.get("message_id") not in consumed_ids]
            self._save_queue({"first_enqueued_at": remaining[0]["created_at"] if remaining else None, "turns": remaining})
            if not remaining:
                self._cancel_timer_locked()
        if self.on_profile_updated:
            self.on_profile_updated(merged)
        logger.info("[PROFILE_BATCH] processed %s raw dialogue turns", len(turns))
        return True

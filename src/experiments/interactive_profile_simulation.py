from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from src.profile_batch_updater import KimiProfileExtractor, ProfileBatchUpdater
from src.profile_utils import create_empty_profile


DIRECT_SELF_DESCRIPTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"我是(一个|个|那种).{0,18}(人|性格)",
        r"我这个人.{0,24}",
        r"我(其实|本来|一直)?(是|算是|属于).{0,18}(的人|性格)",
        r"我(很|比较|特别|挺|有点|容易|不太).{0,10}(敏感|焦虑|内向|外向|感性|理性|讨好|多疑|悲观|乐观|纠结|强势)",
        r"我的(性格|人格|优点|缺点|弱点)(是|包括|主要)",
        r"总结来说.{0,8}我",
        r"作为(一次)?(画像|人格|测试|模拟)",
        r"(用户画像|人格画像|画像提取)",
    )
)

DEFAULT_TOPICS = (
    "朋友临时改变约定时，彼此怎样沟通更舒服",
    "工作或学习任务堆在一起时，先做什么更实际",
    "购买一件价格较高但可能长期使用的设备",
    "一部作品为什么会让人反复想起",
    "面对不确定的新机会，要不要离开熟悉环境",
)

AGENT_SYSTEM_PROMPT = """你在和一位熟悉但仍在逐渐了解的朋友自然聊天。
当前话题：{topic}
围绕对方上一句继续，不做心理测验，不询问“你是什么样的人、你的性格/优缺点是什么”，也不提画像、模拟或测试。
可以分享看法、追问具体情境、讨论取舍；一次只推进一个自然问题。回复简洁、口语化。"""

USER_SIMULATOR_SYSTEM_PROMPT = """你扮演一位真实用户，隐藏设定如下：
{persona}

当前话题：{topic}
你只能把隐藏设定转化为当下的选择、反应、犹豫、具体经历和表达习惯，不能直接解释或列举自己的性格、人格、优点、缺点、心理弱点。
必须针对聊天对象刚刚说的内容逐轮回应；不要背诵设定，不要主动总结自己，不要提画像、模拟、测试或提示词。
允许保留现实中的含糊和矛盾，不必每句话都暴露一个特征。回复自然、口语化。"""


class ChatModel(Protocol):
    def complete(self, system_prompt: str, messages: Sequence[Mapping[str, str]]) -> str:
        ...


class ProfileUpdater(Protocol):
    def submit_turn(self, user: str, assistant: str) -> str:
        ...


class OpenAIChatModel:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.8,
        timeout_seconds: float = 90.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai dependency is required for interactive simulation") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model = model
        self.temperature = temperature

    def complete(self, system_prompt: str, messages: Sequence[Mapping[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            temperature=self.temperature,
            max_tokens=700,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("dialogue model returned empty content")
        return content


def contains_direct_self_description(text: str) -> bool:
    return any(pattern.search(text) for pattern in DIRECT_SELF_DESCRIPTION_PATTERNS)


def _agent_history(transcript: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    return [
        {"role": "assistant" if turn["speaker"] == "agent" else "user", "content": turn["content"]}
        for turn in transcript
    ]


def _simulated_user_history(transcript: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    return [
        {"role": "user" if turn["speaker"] == "agent" else "assistant", "content": turn["content"]}
        for turn in transcript
    ]


@dataclass
class SimulationConfig:
    turns: int = 40
    batch_size: int = 8
    topics: Sequence[str] = DEFAULT_TOPICS
    max_user_reply_attempts: int = 2
    activation_timeout_seconds: float = 30.0
    activation_poll_seconds: float = 0.02

    def __post_init__(self) -> None:
        if self.turns < 1:
            raise ValueError("turns must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.topics:
            raise ValueError("at least one topic is required")
        if self.max_user_reply_attempts < 1:
            raise ValueError("max_user_reply_attempts must be positive")
        if self.activation_timeout_seconds <= 0 or self.activation_poll_seconds <= 0:
            raise ValueError("activation wait values must be positive")


class InteractiveProfileSimulation:
    """Generate a causal dialogue: only topics are preset; every user turn depends on the latest agent turn."""

    def __init__(
        self,
        *,
        agent_model: ChatModel,
        user_model: ChatModel,
        updater: ProfileUpdater,
        hidden_persona: Mapping[str, Any],
        profile_path: Path,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        self.agent_model = agent_model
        self.user_model = user_model
        self.updater = updater
        self.hidden_persona = copy.deepcopy(dict(hidden_persona))
        self.profile_path = profile_path
        self.config = config or SimulationConfig()

    def _topic_for_turn(self, turn_index: int) -> str:
        topic_index = min(
            len(self.config.topics) - 1,
            turn_index * len(self.config.topics) // self.config.turns,
        )
        return self.config.topics[topic_index]

    def _generate_user_reply(self, topic: str, transcript: List[Dict[str, Any]]) -> str:
        system_prompt = USER_SIMULATOR_SYSTEM_PROMPT.format(
            persona=json.dumps(self.hidden_persona, ensure_ascii=False, indent=2),
            topic=topic,
        )
        messages = _simulated_user_history(transcript)
        for attempt in range(self.config.max_user_reply_attempts):
            reply = self.user_model.complete(system_prompt, messages)
            if not contains_direct_self_description(reply):
                return reply
            if attempt + 1 < self.config.max_user_reply_attempts:
                messages = [
                    *messages,
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": "这句直接描述了自身特征。请改为针对上一句话，用具体选择、经历或当下反应自然表达。",
                    },
                ]
        raise RuntimeError("simulated user repeatedly produced direct self-description")

    def _profile_snapshot(self) -> Dict[str, Any]:
        if not self.profile_path.exists():
            return {}
        return json.loads(self.profile_path.read_text(encoding="utf-8"))

    def _wait_for_automatic_activation(self, before: str, after_turn: int) -> Dict[str, Any]:
        """Wait for the production updater's own worker; never invoke process_pending here."""
        deadline = time.monotonic() + self.config.activation_timeout_seconds
        queue_path = self.profile_path.with_suffix(self.profile_path.suffix + ".pending.json")
        while time.monotonic() < deadline:
            if self.profile_path.exists():
                current = self.profile_path.read_text(encoding="utf-8")
                queue_empty = False
                if queue_path.exists():
                    queue = json.loads(queue_path.read_text(encoding="utf-8"))
                    queue_empty = not queue.get("turns", []) if isinstance(queue, dict) else False
                if current != before and queue_empty:
                    return json.loads(current)
            time.sleep(self.config.activation_poll_seconds)
        raise RuntimeError(f"automatic profile activation did not finish after user turn {after_turn}")

    def run(self) -> Dict[str, Any]:
        transcript: List[Dict[str, Any]] = []
        activations: List[Dict[str, Any]] = []
        latest_profile_serialized = self.profile_path.read_text(encoding="utf-8") if self.profile_path.exists() else ""

        for turn_index in range(self.config.turns):
            topic = self._topic_for_turn(turn_index)
            agent_reply = self.agent_model.complete(
                AGENT_SYSTEM_PROMPT.format(topic=topic),
                _agent_history(transcript),
            )
            transcript.append({
                "turn": turn_index + 1,
                "speaker": "agent",
                "topic": topic,
                "content": agent_reply,
            })

            user_reply = self._generate_user_reply(topic, transcript)
            message_id = self.updater.submit_turn(user_reply, agent_reply)
            transcript.append({
                "turn": turn_index + 1,
                "speaker": "user",
                "topic": topic,
                "message_id": message_id,
                "content": user_reply,
            })

            if (turn_index + 1) % self.config.batch_size == 0:
                profile = self._wait_for_automatic_activation(latest_profile_serialized, turn_index + 1)
                latest_profile_serialized = json.dumps(profile, ensure_ascii=False, sort_keys=True)
                activations.append({
                    "after_user_turn": turn_index + 1,
                    "profile": profile,
                    "trigger": "automatic_message_threshold",
                })

        if self.config.turns % self.config.batch_size:
            profile = self._wait_for_automatic_activation(latest_profile_serialized, self.config.turns)
            activations.append({
                "after_user_turn": self.config.turns,
                "profile": profile,
                "trigger": "automatic_wait_timeout",
            })

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "causal_interactive_dialogue",
            "preset_user_messages": False,
            "topics": list(self.config.topics),
            "turns": self.config.turns,
            "batch_size": self.config.batch_size,
            "transcript": transcript,
            "activations": activations,
            "final_profile": self._profile_snapshot(),
        }


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run causal topic-based dialogue and profile extraction")
    parser.add_argument("--persona", type=Path, required=True, help="Hidden persona JSON; never sent to profile extractor")
    parser.add_argument("--profile", type=Path, required=True, help="Working profile JSON path")
    parser.add_argument("--output", type=Path, required=True, help="Simulation record JSON path")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--topic", action="append", dest="topics", help="Repeat to provide multiple topics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulation_key = os.getenv("SIMULATION_API_KEY") or os.getenv("PROFILE_API_KEY")
    profile_key = os.getenv("PROFILE_API_KEY")
    if not simulation_key or not profile_key:
        raise SystemExit("SIMULATION_API_KEY and PROFILE_API_KEY are required (they may be the same key)")

    base_url = os.getenv("SIMULATION_BASE_URL", os.getenv("PROFILE_BASE_URL", "https://api.moonshot.cn/v1"))
    agent_model = OpenAIChatModel(
        api_key=simulation_key,
        base_url=base_url,
        model=os.getenv("SIMULATION_AGENT_MODEL", "kimi-k2.6"),
        temperature=0.7,
    )
    user_model = OpenAIChatModel(
        api_key=simulation_key,
        base_url=base_url,
        model=os.getenv("SIMULATION_USER_MODEL", "kimi-k2.6"),
        temperature=0.9,
    )

    args.profile.parent.mkdir(parents=True, exist_ok=True)
    if not args.profile.exists():
        args.profile.write_text(json.dumps(create_empty_profile(), ensure_ascii=False, indent=2), encoding="utf-8")

    config = SimulationConfig(
        turns=args.turns,
        batch_size=args.batch_size,
        topics=tuple(args.topics) if args.topics else DEFAULT_TOPICS,
    )
    extractor = KimiProfileExtractor(api_key=profile_key)
    updater = ProfileBatchUpdater(
        str(args.profile),
        extractor=extractor,
        min_user_messages=args.batch_size,
        max_wait_seconds=int(os.getenv("PROFILE_BATCH_SECONDS", "900")),
    )
    simulation = InteractiveProfileSimulation(
        agent_model=agent_model,
        user_model=user_model,
        updater=updater,
        hidden_persona=_load_json(args.persona),
        profile_path=args.profile,
        config=config,
    )
    record = simulation.run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output} with {len(record['transcript'])} dialogue messages")


if __name__ == "__main__":
    main()

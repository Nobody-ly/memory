"""Adapters that run the repository's real Deep Empathy state components."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..llm_client import LLMClient
from ..prompts.templates_en import (
    BACKGROUND_REASONING_SYSTEM_PROMPT,
    BACKGROUND_REASONING_USER_PROMPT_TEMPLATE,
)
from .exp2_schema import (
    FRAMEWORK_STATE_RESPONSE_SCHEMA,
    normalize_framework_state,
)


FRAMEWORK_STATE_MAX_TOKENS = 2048


def latest_complete_exchange(
    context_turns: List[Dict[str, Any]],
    user_speaker: str,
    partner_speaker: str,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Return the latest observed user turn and the partner reply that followed."""
    user_index = next(
        (
            index
            for index in range(len(context_turns) - 1, -1, -1)
            if context_turns[index]["speaker"].casefold()
            == user_speaker.casefold()
        ),
        None,
    )
    if user_index is None:
        return [], "", ""

    exchange = context_turns[user_index:]
    partner_messages = [
        turn["content"]
        for turn in exchange[1:]
        if turn["speaker"].casefold() == partner_speaker.casefold()
    ]
    return (
        exchange,
        exchange[0]["content"],
        "\n".join(partner_messages).strip(),
    )


class FrameworkStateReasoner:
    """Run the original background state prompt with a strict output contract."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def derive(
        self,
        *,
        user_input: str,
        assistant_response: str,
        static_profile: Dict[str, Any],
        previous_state: Dict[str, Any],
        previous_context: List[Dict[str, Any]],
        agent_persona: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not user_input or not assistant_response:
            return {}
        prompt = BACKGROUND_REASONING_USER_PROMPT_TEMPLATE.format(
            user_input=user_input,
            assistant_response=assistant_response,
            static_profile=json.dumps(
                static_profile, ensure_ascii=False, indent=2
            ),
            current_state=json.dumps(
                previous_state.get("current_state", {}),
                ensure_ascii=False,
                indent=2,
            ),
            current_context=json.dumps(
                previous_context, ensure_ascii=False, indent=2
            ),
            persona_config=json.dumps(
                agent_persona, ensure_ascii=False, indent=2
            ),
            relevant_memory="{}",
        )
        raw = self.llm.chat(
            BACKGROUND_REASONING_SYSTEM_PROMPT,
            prompt,
            temperature=0.2,
            max_tokens=FRAMEWORK_STATE_MAX_TOKENS,
            response_schema=FRAMEWORK_STATE_RESPONSE_SCHEMA,
        )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "strict framework-state response was not valid JSON"
            ) from exc
        return normalize_framework_state(value)

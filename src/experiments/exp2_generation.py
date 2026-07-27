"""Response-generation conditions and REALTALK-aligned response metrics."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..llm_client import LLMClient
from ..metrics import compute_style_similarity
from ..prompts.templates_en import DDIRECT_RESPONSE_SYSTEM_PROMPT
from .persona_simulation import condense_persona, condense_profile


RESPONSE_MAX_TOKENS = 300
BERTSCORE_MODEL = "roberta-large"
BERTSCORE_NUM_LAYERS = 17

_HISTORY_BLOCK = """CONVERSATION HISTORY:
{history}

"""
_PROFILE_BLOCK = """ABOUT {user_speaker}:
{profile}

"""
_FRAMEWORK_BLOCK = """DEEP EMPATHY FRAMEWORK STATE:
{framework}

"""
_RESPONSE_TASK = """You are {agent_speaker}, replying to {user_speaker}.

YOUR PERSONA:
{persona}

{user_speaker}'s current message:
"{user_message}"

Treat {user_speaker} as a dataset identifier. Do not address the user by a
stored name, nickname, or alternate identity unless the current conversation
itself establishes that name.
Do not invent a recent event or current activity. Personal claims must be
supported by the persona or observed conversation; otherwise respond generally.
Do not introduce yourself or ask either person's name unless identity is
directly relevant to the current message.

Generate only {agent_speaker}'s natural next message."""


def build_response_prompt(
    *,
    method: str,
    user_message: str,
    context_turns: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]],
    persona: Dict[str, Any],
    framework_guidance: Optional[Dict[str, Any]],
    agent_speaker: str,
    user_speaker: str,
) -> str:
    """Build four nested conditions while keeping the current message common."""
    parts: List[str] = []
    if method != "llm_only" and context_turns:
        history = "\n".join(
            f"{turn['speaker']}: {turn['content']}" for turn in context_turns
        )
        parts.append(_HISTORY_BLOCK.format(history=history))
    if method in {"user_profile", "full_framework"} and profile:
        parts.append(_PROFILE_BLOCK.format(
            user_speaker=user_speaker,
            profile=condense_profile(profile),
        ))
    if method == "full_framework" and framework_guidance:
        parts.append(_FRAMEWORK_BLOCK.format(
            framework=json.dumps(
                framework_guidance, ensure_ascii=False, indent=2
            )
        ))
    parts.append(_RESPONSE_TASK.format(
        agent_speaker=agent_speaker,
        user_speaker=user_speaker,
        persona=condense_persona(persona),
        user_message=user_message,
    ))
    return "".join(parts)


class Exp2ResponseGenerator:
    def __init__(self, llm: LLMClient, method: str):
        self.llm = llm
        self.method = method

    def generate(
        self,
        *,
        user_message: str,
        context_turns: List[Dict[str, Any]],
        profile: Optional[Dict[str, Any]],
        persona: Dict[str, Any],
        framework_guidance: Optional[Dict[str, Any]],
        agent_speaker: str,
        user_speaker: str,
    ) -> str:
        prompt = build_response_prompt(
            method=self.method,
            user_message=user_message,
            context_turns=context_turns,
            profile=profile,
            persona=persona,
            framework_guidance=framework_guidance,
            agent_speaker=agent_speaker,
            user_speaker=user_speaker,
        )
        response = self.llm.chat(
            DDIRECT_RESPONSE_SYSTEM_PROMPT,
            prompt,
            temperature=0.7,
            max_tokens=RESPONSE_MAX_TOKENS,
        ).strip()
        if not response:
            raise ValueError("generated response must not be empty")
        return response


def compute_response_scores(
    *,
    reference_text: str,
    candidate_text: str,
    reference_ei: Dict[str, Any],
    candidate_ei: Dict[str, Any],
    bertscore_f1: Optional[float] = None,
) -> Dict[str, Any]:
    scores = {
        **compute_style_similarity(reference_text, candidate_text),
        "reflectiveness_accuracy": float(
            candidate_ei["reflective"] == reference_ei["reflective"]
        ),
        "grounding_accuracy": float(
            candidate_ei["grounding"] == reference_ei["grounding"]
        ),
        "sentiment_accuracy": float(
            candidate_ei["sentiment"] == reference_ei["sentiment"]
        ),
        "emotion_accuracy": float(
            candidate_ei["emotion"] == reference_ei["emotion"]
        ),
        "intimacy_absolute_difference": round(
            abs(candidate_ei["intimacy"] - reference_ei["intimacy"]), 4
        ),
        "empathy_absolute_difference": float(
            abs(
                _empathy_total(candidate_ei["empathy"])
                - _empathy_total(reference_ei["empathy"])
            )
        ),
        "candidate_epitome_total": float(_empathy_total(candidate_ei["empathy"])),
        "reference_epitome_total": float(_empathy_total(reference_ei["empathy"])),
    }
    if bertscore_f1 is not None:
        scores["bertscore_f1"] = round(float(bertscore_f1), 6)
    return scores


def compute_bertscore_f1(
    references: Sequence[str],
    candidates: Sequence[str],
) -> List[float]:
    if len(references) != len(candidates):
        raise ValueError("BERTScore references and candidates must align")
    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore is unavailable; install the realtalk-eval extra"
        ) from exc
    _, _, f1 = score(
        list(candidates),
        list(references),
        lang="en",
        model_type=BERTSCORE_MODEL,
        num_layers=BERTSCORE_NUM_LAYERS,
        idf=False,
        rescale_with_baseline=False,
        verbose=False,
    )
    return [float(value) for value in f1.tolist()]


def add_batched_bertscore(results: List[Dict[str, Any]]) -> None:
    locations: List[tuple[Dict[str, Any], str, str]] = []
    for result in results:
        reference = result.get("ground_truth_response")
        if not reference:
            continue
        for method_data in result["methods"].values():
            generation = method_data.get("generation")
            if generation and generation.get("response"):
                locations.append((
                    generation["scores"],
                    reference,
                    generation["response"],
                ))
    if not locations:
        return
    values = compute_bertscore_f1(
        [item[1] for item in locations],
        [item[2] for item in locations],
    )
    for (scores, _, _), value in zip(locations, values):
        scores["bertscore_f1"] = round(value, 6)


def _empathy_total(value: Dict[str, Any]) -> int:
    return sum(
        int(value.get(field, 0))
        for field in ("emotional_reaction", "interpretation", "exploration")
    )

"""
Experiment 2b: Empathy Alignment Reasoning Qualitative Analysis

This experiment validates the self-domain + user-domain collaborative alignment
mechanism for empathy. It:
1. Selects cases with negative emotions from REALTALK conversations
2. Runs the empathy alignment reasoning to produce an empathy state
3. Generates responses WITH and WITHOUT the alignment reasoning
4. Evaluates both using the EPITOME framework
5. Analyzes whether alignment reasoning produces more appropriate empathy
"""
from __future__ import annotations

import argparse
import json
import re
import string
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..llm_client import LLMClient
from ..utils import load_json, save_json, parse_json
from ..epistemic_decay import EpistemicDecayTracker, get_exploration_label
from ..prompts.templates_en import (
    EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
)
from ..prompts.eval_templates_en import (
    EPITOME_EVALUATION_SYSTEM_PROMPT,
    EPITOME_EVALUATION_USER_PROMPT_TEMPLATE,
)
from .persona_simulation import (
    detect_speakers,
    flatten_messages,
    format_conversation_history,
    condense_profile,
    condense_persona,
)

EMPATHY_ALIGNMENT_MAX_TOKENS = 2400


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------

def robust_parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON with fallback for common LLM formatting errors."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fix missing commas between key-value pairs
    fixed = re.sub(r'"\s*\n\s*"', '",\n"', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Fix missing commas after string values
    fixed = re.sub(r'"\s*\n(\s*")', '",\n\1', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        raise e


# ---------------------------------------------------------------------------
# Case selection: find emotionally rich interactions
# ---------------------------------------------------------------------------

NEGATIVE_INDICATORS = [
    "sad", "upset", "stressed", "anxious", "worried", "tired", "exhausted",
    "depressed", "lonely", "frustrated", "angry", "scared", "afraid", "nervous",
    "overwhelmed", "burnt out", "burned out", "sick", "ill", "under the weather",
    "struggling", "difficult", "hard time", "rough", "bad day", "terrible",
    "didn't work out", "failed", "lost", "miss", "breakup", "broke up",
    "death", "died", "passed away", "hurt", "pain", "cry", "crying",
    "can't sleep", "insomnia", "panic", "dread", "hopeless", "give up",
    "not feeling well", "feeling down", "feeling bad", "feeling terrible",
    "so hard", "too much", "can't handle", "stressed out", "worried about",
    "scared of", "afraid of", "anxious about", "struggling with",
]

POSITIVE_INDICATORS = [
    "happy", "excited", "great", "amazing", "wonderful", "love it", "awesome",
    "fantastic", "celebrate", "achievement", "success", "promoted", "won",
    "graduated", "engaged", "pregnant", "birthday", "anniversary",
    "so happy", "really excited", "looking forward", "can't wait",
]


# Strip punctuation so word boundaries are clean
_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def _tokenize(text: str) -> set:
    """Lowercase, strip punctuation, split into a set of words for O(1) lookup."""
    return set(text.lower().translate(_PUNCT_TABLE).split())


def detect_emotion_type(content: str) -> str:
    """Detect if a message contains negative, positive, or neutral emotional content."""
    words = _tokenize(content)
    neg_count = sum(1 for kw in NEGATIVE_INDICATORS if kw in words)
    pos_count = sum(1 for kw in POSITIVE_INDICATORS if kw in words)
    if neg_count > pos_count and neg_count > 0:
        return "negative"
    if pos_count > neg_count and pos_count > 0:
        return "positive"
    return "neutral"


def select_emotional_cases(
    chat: Dict[str, Any],
    agent_speaker: str,
    min_context_messages: int = 10,
    max_cases: int = 5,
) -> List[Dict[str, Any]]:
    """Select cases where the USER expresses a clear emotional state."""
    turns = flatten_messages(chat)
    names = chat.get("name", {})
    user_speaker = str(names.get("speaker_1", "")).strip()

    cases: List[Dict[str, Any]] = []
    neg_cases: List[Dict[str, Any]] = []
    pos_cases: List[Dict[str, Any]] = []

    for i, turn in enumerate(turns):
        if turn["speaker"] != user_speaker:
            continue

        emotion = detect_emotion_type(turn["content"])
        if emotion == "neutral":
            continue

        # Find the agent's actual response
        ground_truth = None
        gt_dia_id = None
        for j in range(i + 1, len(turns)):
            if turns[j]["speaker"] == agent_speaker:
                ground_truth = turns[j]["content"]
                gt_dia_id = turns[j].get("dia_id", "")
                break

        if ground_truth is None:
            continue

        context_turns = turns[:i]
        if len(context_turns) < min_context_messages:
            continue

        case = {
            "case_id": f"case_{turn.get('dia_id', i)}",
            "session_id": turn["session_id"],
            "user_message": turn["content"],
            "emotion_type": emotion,
            "ground_truth_response": ground_truth,
            "ground_truth_dia_id": gt_dia_id,
            "context_turns": context_turns,
        }

        if emotion == "negative":
            neg_cases.append(case)
        else:
            pos_cases.append(case)

    # Prioritize negative cases
    selected = neg_cases[:max_cases]
    remaining = max_cases - len(selected)
    if remaining > 0:
        selected.extend(pos_cases[:remaining])
    return selected


# ---------------------------------------------------------------------------
# Empathy Alignment Reasoner
# ---------------------------------------------------------------------------

class EmpathyAlignmentReasoner:
    """Performs self-domain + user-domain alignment reasoning."""

    def __init__(self, llm: LLMClient, interaction_count: int = 0):
        self.llm = llm
        self.epistemic_tracker = EpistemicDecayTracker(initial_count=interaction_count)

    def reason(
        self,
        user_message: str,
        context_turns: List[Dict[str, Any]],
        profile: Dict[str, Any],
        persona: Dict[str, Any],
        current_state: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        recent_context = format_conversation_history(context_turns[-10:])
        profile_text = condense_profile(profile)
        persona_text = condense_persona(persona)
        state_text = json.dumps(current_state or {}, ensure_ascii=False)

        # Compute omega(t) based on profile maturity and interaction count
        state_axis = profile.get("state_axis")
        if isinstance(state_axis, dict):
            static_profile = state_axis.get("static_profile", {})
        else:
            static_profile = profile
        omega = self.epistemic_tracker.compute(static_profile)
        omega_label = get_exploration_label(omega)

        user_prompt = EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE.format(
            recent_context=recent_context,
            user_message=user_message,
            user_profile=profile_text,
            agent_persona=persona_text,
            current_state=state_text,
            epistemic_omega=omega,
        )

        try:
            result = robust_parse_json(self.llm.chat(
                EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.3,
                max_tokens=EMPATHY_ALIGNMENT_MAX_TOKENS,
            ))
            self.epistemic_tracker.increment()
            return result
        except Exception as e:
            print(f"[Empathy Reasoning Error] {e}")
            return {}


# ---------------------------------------------------------------------------
# Response generators
# ---------------------------------------------------------------------------

class AlignedResponseGenerator:
    """Generates response USING empathy alignment reasoning."""

    SYSTEM_PROMPT = """You are {agent_speaker}, a companion agent in a long-term conversation with {user_speaker}.

YOUR PERSONA:
{persona_text}

ABOUT {user_speaker}:
{profile_text}

EMPLOYED EMPATHY STATE (from alignment reasoning):
{empathy_state}

CONVERSATION CONTEXT:
{history}

{user_speaker}'s message: "{user_message}"

TASK: Generate your response as {agent_speaker}, following the empathy state guidance above.

The empathy state tells you exactly how empathetic to be and what tone to use. Follow it closely.

Generate ONLY your message text:"""

    def __init__(self, llm: LLMClient, profile: Dict[str, Any], persona: Dict[str, Any]):
        self.llm = llm
        self.profile = profile
        self.persona = persona

    def generate(
        self,
        user_message: str,
        context_turns: List[Dict[str, Any]],
        empathy_state: Dict[str, Any],
        agent_speaker: str,
        user_speaker: str,
    ) -> str:
        history = format_conversation_history(context_turns[-15:])
        profile_text = condense_profile(self.profile)
        persona_text = condense_persona(self.persona)

        user_prompt = self.SYSTEM_PROMPT.format(
            agent_speaker=agent_speaker,
            user_speaker=user_speaker,
            persona_text=persona_text,
            profile_text=profile_text,
            empathy_state=json.dumps(empathy_state, ensure_ascii=False, indent=2),
            history=history,
            user_message=user_message,
        )
        return self.llm.chat(
            "You are a helpful assistant generating empathetic conversation responses.",
            user_prompt,
            temperature=0.7,
            max_tokens=200,
        )


class DirectResponseGenerator:
    """Generates response WITHOUT empathy alignment (direct generation)."""

    SYSTEM_PROMPT = """You are {agent_speaker}, a companion agent in a long-term conversation with {user_speaker}.

YOUR PERSONA:
{persona_text}

ABOUT {user_speaker}:
{profile_text}

CONVERSATION CONTEXT:
{history}

{user_speaker}'s message: "{user_message}"

TASK: Generate your response as {agent_speaker}.

Generate ONLY your message text:"""

    def __init__(self, llm: LLMClient, profile: Dict[str, Any], persona: Dict[str, Any]):
        self.llm = llm
        self.profile = profile
        self.persona = persona

    def generate(
        self,
        user_message: str,
        context_turns: List[Dict[str, Any]],
        agent_speaker: str,
        user_speaker: str,
    ) -> str:
        history = format_conversation_history(context_turns[-15:])
        profile_text = condense_profile(self.profile)
        persona_text = condense_persona(self.persona)

        user_prompt = self.SYSTEM_PROMPT.format(
            agent_speaker=agent_speaker,
            user_speaker=user_speaker,
            persona_text=persona_text,
            profile_text=profile_text,
            history=history,
            user_message=user_message,
        )
        return self.llm.chat(
            "You are a helpful assistant generating conversation responses.",
            user_prompt,
            temperature=0.7,
            max_tokens=200,
        )


# ---------------------------------------------------------------------------
# EPITOME Evaluator
# ---------------------------------------------------------------------------

class EPITOMEEvaluator:
    """Evaluates responses using the EPITOME empathy framework."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(
        self,
        context_turns: List[Dict[str, Any]],
        user_message: str,
        response: str,
        user_emotion: str,
    ) -> Dict[str, Any]:
        context = format_conversation_history(context_turns[-5:])

        user_prompt = EPITOME_EVALUATION_USER_PROMPT_TEMPLATE.format(
            context=context,
            user_message=user_message,
            response=response,
            user_emotion=user_emotion,
        )
        try:
            result = robust_parse_json(self.llm.chat(
                EPITOME_EVALUATION_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.1,
                max_tokens=600,
            ))
            return result
        except Exception as e:
            print(f"[EPITOME Eval Error] {e}")
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

@dataclass
class Experiment2bConfig:
    dataset_dir: str = "dataset"
    output_dir: str = "data/empathy_alignment_eval"
    min_context_messages: int = 10
    max_cases_per_chat: int = 3
    chat_filter: Optional[List[str]] = None


def run_empathy_alignment_experiment(config: Experiment2bConfig) -> Dict[str, Any]:
    dataset_dir = Path(config.dataset_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_files = sorted(dataset_dir.glob("Chat_*.json"))
    if config.chat_filter:
        chat_files = [f for f in chat_files if f.stem in config.chat_filter]

    print(f"[Empathy Alignment] Processing {len(chat_files)} chat files")

    llm = LLMClient()
    reasoner = EmpathyAlignmentReasoner(llm)
    epitome_eval = EPITOMEEvaluator(llm)

    all_results: List[Dict[str, Any]] = []

    for chat_file in chat_files:
        print(f"\n[Empathy Alignment] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, agent_speaker = detect_speakers(chat)

        profile_path = dataset_dir / "output" / "user" / f"{user_speaker.lower().replace(' ', '_')}_profile.json"
        persona_path = dataset_dir / "output" / "agent" / f"{agent_speaker.lower().replace(' ', '_')}_persona.json"

        if not profile_path.exists() or not persona_path.exists():
            print(f"  [Skip] Missing profile or persona")
            continue

        profile = load_json(str(profile_path))
        persona = load_json(str(persona_path))

        aligned_gen = AlignedResponseGenerator(llm, profile, persona)
        direct_gen = DirectResponseGenerator(llm, profile, persona)

        cases = select_emotional_cases(
            chat, agent_speaker, config.min_context_messages, config.max_cases_per_chat
        )
        neg_count = sum(1 for c in cases if c["emotion_type"] == "negative")
        pos_count = sum(1 for c in cases if c["emotion_type"] == "positive")
        print(f"  Found {len(cases)} emotional cases ({neg_count} negative, {pos_count} positive)")

        for case in cases:
            print(f"  Processing {case['case_id']} ({case['emotion_type']})...")
            context_turns = case["context_turns"]
            user_message = case["user_message"]
            user_emotion = case["emotion_type"]

            print(f"    [1/4] Running empathy alignment reasoning...")
            empathy_reasoning = reasoner.reason(
                user_message, context_turns, profile, persona
            )
            empathy_state = empathy_reasoning.get("empathy_state", {})

            print(f"    [2/4] Generating aligned response...")
            aligned_response = aligned_gen.generate(
                user_message, context_turns, empathy_state, agent_speaker, user_speaker
            )

            print(f"    [3/4] Generating direct response...")
            direct_response = direct_gen.generate(
                user_message, context_turns, agent_speaker, user_speaker
            )

            print(f"    [4/4] Evaluating with EPITOME...")
            aligned_eval = epitome_eval.evaluate(
                context_turns, user_message, aligned_response, user_emotion
            )
            direct_eval = epitome_eval.evaluate(
                context_turns, user_message, direct_response, user_emotion
            )

            result = {
                "chat_file": chat_file.name,
                "case_id": case["case_id"],
                "emotion_type": user_emotion,
                "user_message": user_message,
                "ground_truth_response": case["ground_truth_response"],
                "empathy_reasoning": empathy_reasoning,
                "aligned_response": aligned_response,
                "direct_response": direct_response,
                "aligned_epitome": aligned_eval,
                "direct_epitome": direct_eval,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            a_total = aligned_eval.get("total_empathy_score", 0)
            d_total = direct_eval.get("total_empathy_score", 0)
            a_approp = aligned_eval.get("appropriateness", "?")
            d_approp = direct_eval.get("appropriateness", "?")
            print(f"    Aligned: EPITOME={a_total}/6, approp={a_approp}")
            print(f"    Direct:  EPITOME={d_total}/6, approp={d_approp}")

    results_path = output_dir / "empathy_alignment_results.json"
    save_json(str(results_path), {"results": all_results, "config": vars(config)})
    print(f"\n[Empathy Alignment] Results saved to {results_path}")

    summary = aggregate_empathy_results(all_results)
    summary_path = output_dir / "empathy_alignment_summary.json"
    save_json(str(summary_path), summary)
    print(f"[Empathy Alignment] Summary saved to {summary_path}")

    return summary


def aggregate_empathy_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    aligned_scores = []
    direct_scores = []
    neg_aligned = []
    neg_direct = []

    for r in results:
        a_eval = r.get("aligned_epitome", {})
        d_eval = r.get("direct_epitome", {})
        if "error" in a_eval or "error" in d_eval:
            continue
        aligned_scores.append(a_eval)
        direct_scores.append(d_eval)
        if r.get("emotion_type") == "negative":
            neg_aligned.append(a_eval)
            neg_direct.append(d_eval)

    n = len(aligned_scores)

    def avg(key, scores):
        vals = [s.get(key, 0) for s in scores if isinstance(s.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def avg_sub(key, scores):
        vals = []
        for s in scores:
            sub = s.get(key, {})
            if isinstance(sub, dict):
                v = sub.get("score", 0)
                if isinstance(v, (int, float)):
                    vals.append(v)
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {
        "num_cases": n,
        "emotion_breakdown": {
            "negative": sum(1 for r in results if r.get("emotion_type") == "negative"),
            "positive": sum(1 for r in results if r.get("emotion_type") == "positive"),
        },
        "aligned_response": {
            "avg_total_empathy": avg("total_empathy_score", aligned_scores),
            "avg_emotional_reaction": avg_sub("emotional_reaction", aligned_scores),
            "avg_interpretation": avg_sub("interpretation", aligned_scores),
            "avg_exploration": avg_sub("exploration", aligned_scores),
            "appropriateness": {
                "appropriate": sum(1 for s in aligned_scores if s.get("appropriateness") == "appropriate"),
                "excessive": sum(1 for s in aligned_scores if s.get("appropriateness") == "excessive"),
                "insufficient": sum(1 for s in aligned_scores if s.get("appropriateness") == "insufficient"),
            },
        },
        "direct_response": {
            "avg_total_empathy": avg("total_empathy_score", direct_scores),
            "avg_emotional_reaction": avg_sub("emotional_reaction", direct_scores),
            "avg_interpretation": avg_sub("interpretation", direct_scores),
            "avg_exploration": avg_sub("exploration", direct_scores),
            "appropriateness": {
                "appropriate": sum(1 for s in direct_scores if s.get("appropriateness") == "appropriate"),
                "excessive": sum(1 for s in direct_scores if s.get("appropriateness") == "excessive"),
                "insufficient": sum(1 for s in direct_scores if s.get("appropriateness") == "insufficient"),
            },
        },
    }

    a_total = summary["aligned_response"]["avg_total_empathy"]
    d_total = summary["direct_response"]["avg_total_empathy"]
    summary["improvement"] = {
        "total_empathy_delta": round(a_total - d_total, 3),
        "total_empathy_pct": round((a_total - d_total) / d_total * 100, 1) if d_total > 0 else 0,
    }

    if neg_aligned and neg_direct:
        summary["negative_cases_only"] = {
            "num_cases": len(neg_aligned),
            "aligned_avg_empathy": avg("total_empathy_score", neg_aligned),
            "direct_avg_empathy": avg("total_empathy_score", neg_direct),
            "aligned_excessive": sum(1 for s in neg_aligned if s.get("appropriateness") == "excessive"),
            "direct_excessive": sum(1 for s in neg_direct if s.get("appropriateness") == "excessive"),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 2b: Empathy Alignment Reasoning Analysis")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/empathy_alignment_eval")
    parser.add_argument("--min-context", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--chats", nargs="*", default=None)
    args = parser.parse_args()

    config = Experiment2bConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_context_messages=args.min_context,
        max_cases_per_chat=args.max_cases,
        chat_filter=args.chats,
    )

    run_empathy_alignment_experiment(config)


if __name__ == "__main__":
    main()

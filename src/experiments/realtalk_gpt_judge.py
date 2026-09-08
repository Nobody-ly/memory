"""REALTALK Appendix C GPT judge for completed persona predictions."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exp1_protocol import build_message_level_points, select_realtalk_splits
from ..utils import load_json


PAPER_TABLE2 = {
    "without_finetuning": {
        "reflectiveness_accuracy": 0.62,
        "grounding_accuracy": 0.40,
        "empathy_absolute_difference": 1.80,
    },
    "with_finetuning": {
        "reflectiveness_accuracy": 0.77,
        "grounding_accuracy": 0.62,
        "empathy_absolute_difference": 1.24,
    },
}

REFLECTIVENESS_PROMPT = """Reflectiveness Classification
You are an evaluator trained to determine if a speaker's language is reflective, indicating self-awareness. Reflective language is characterized by self-observation, perspective-taking, and intentionality. This means that the speaker is not only aware of their thoughts, feelings, or actions but also able to express this awareness clearly.

A reflective response often includes one or more of the following traits:
- Self-observation: The speaker describes their own emotional or cognitive state (e.g., "I feel uncertain about..." or "I'm aware that...").
- Perspective-taking: The speaker shows an understanding of how their actions or emotions affect others or acknowledges another person's perspective on the situation (e.g., "I understand that my response may seem...").
- Intentionality: The speaker explains the reasoning behind their behavior or decisions, revealing their underlying motivations or goals (e.g., "I decided to respond this way because...").

Example Statements
- "I realize I tend to get defensive when I receive feedback, and I think it's because I want to do well."
Reflective or Not Reflective: Reflective
Reason: This statement shows self-observation and insight into motivation.
- "I did what I thought was best for the project."
Reflective or Not Reflective: Not Reflective
Reason: While the speaker describes their decision, they don't analyze or acknowledge the emotions or motivations behind their choice or consider its impact on others.

Given this dialogue context:
{history}

Determine whether the speaker's last message ({turn}) is reflective or not.
Reflective language includes phrases like 'I feel...', 'I think...', or similar reflective expressions.
Respond only with 'True' for reflective or 'False' for not reflective."""

GROUNDING_PROMPT = """Grounding Act Classification
You are an evaluator trained to determine if a speaker's language demonstrates grounding, which reflects active engagement and a commitment to mutual understanding in conversation. Grounding acts are characterized by clarifying questions, follow-up inquiries, or statements that seek to confirm, clarify, or expand on shared information. These acts are essential for building common ground, ensuring that both participants have a clear understanding, and preventing misunderstandings.

A grounding response often includes one or more of the following traits:
- Clarifying questions: The speaker asks questions that seek clarification or further information about the other person's statements (e.g., "Could you explain that further?" or "What did you mean by...?").
- Follow-up inquiries: The speaker shows interest in exploring a point raised by the other person, prompting them to elaborate or continue sharing (e.g., "How did that make you feel?" or "Can you tell me more about...?").
- Confirmation checks: The speaker seeks to confirm their understanding of what the other person said (e.g., "So, you mean that...?" or "Are you saying that...?").

Example Statements
- "Can you tell me more about what happened at the event?"
Grounding or Not Grounding: Grounding
Reason: This is a follow-up question that prompts the other person to provide more information, demonstrating interest and a desire to deepen mutual understanding.
- "I completely understand your point."
Grounding or Not Grounding: Not Grounding
Reason: Although this statement indicates agreement, it does not actively seek further information or clarification and does not encourage continued dialogue.
- "So, you're saying that this new policy will impact the timeline?"
Grounding or Not Grounding: Grounding
Reason: This is a confirmation check, as the speaker seeks to ensure their understanding of the other person's statement.
- "It sounds like you've already made your decision."
Grounding or Not Grounding: Not Grounding
Reason: This statement reflects an observation rather than a clarifying or follow-up question, so it does not serve as a grounding act.

Dialogue context:
{history}

Determine whether the speaker's last message ({turn}) is grounding or not grounding.
Respond only with 'True' for grounding or 'False' for not grounding."""

EMPATHY_PROMPT = """Empathy Assessment
You are an evaluator assessing the level of empathy conveyed in a response, based on three core components: Emotional Reaction, Interpretation, and Exploration. For each component, provide a score from 0-2, where 0 indicates no presence, 1 indicates partial presence, and 2 indicates explicit presence.

Component 1: Emotional Reaction
Does the response express or allude to warmth, compassion, concern, or similar feelings of the responder towards the seeker?
- 0: No.
- 1: The response alludes to these feelings but they are not explicitly expressed.
- 2: The response has an explicit mention.

Component 2: Interpretation
Does the response communicate an understanding of the seeker's experiences and feelings? In what manner?
- 0: No.
- 1: Yes, the response communicates an understanding of the seeker's experiences and/or feelings. The response may contain conjectures or speculations, reflect back on or describe similar experiences of the responder or others, or paraphrase the seeker's experiences and/or feelings.
- 2: The response provides a deep, explicit understanding and validation of the seeker's feelings or experiences, potentially using multiple sub-categories.

Component 3: Exploration
Does the response attempt to explore the seeker's experiences and feelings?
- 0: No.
- 1: Exploration is present but generic.
- 2: Exploration is specific and delves into the particular experience or feeling.

Dialogue context:
{history}

Response to assess:
{turn}

Return only JSON:
{{"emotional_reaction": 0, "interpretation": 0, "exploration": 0}}"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_bool(text: str) -> bool:
    value = text.strip().strip("`'\". ").casefold()
    if value in {"true", "reflective", "grounding"}:
        return True
    if value in {"false", "not reflective", "not grounding"}:
        return False
    match = re.search(r"\b(true|false)\b", value)
    if match:
        return match.group(1) == "true"
    raise ValueError(f"cannot parse boolean judgment: {text[:120]!r}")


def _parse_empathy(text: str) -> dict[str, int]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("empathy judgment does not contain JSON")
    value = json.loads(match.group(0))
    fields = ("emotional_reaction", "interpretation", "exploration")
    if set(value) != set(fields):
        raise ValueError("empathy judgment has unexpected fields")
    normalized = {field: int(value[field]) for field in fields}
    if any(score < 0 or score > 2 for score in normalized.values()):
        raise ValueError("empathy scores must be in [0, 2]")
    return normalized


def _chat(base_url: str, api_key: str, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 180,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.load(response)
            content = data["choices"][0]["message"]["content"]
            return content, {
                "model": data.get("model"),
                "usage": data.get("usage", {}),
                "attempt": attempt,
            }
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    assert last_error is not None
    raise last_error


def _contexts(dataset_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    wanted = {row["speaker"] for row in rows}
    contexts: dict[str, str] = {}
    for split in select_realtalk_splits(dataset_dir, speaker_filter=sorted(wanted)):
        chat = load_json(str(dataset_dir / split["test_chat"]))
        points = build_message_level_points(
            chat,
            split["speaker"],
            test_sessions=3,
            merge_adjacent_bubbles=True,
        )
        for point in points:
            result_id = f"{_speaker_id(split['speaker'])}:{point['sample_id']}"
            within_session = [
                turn for turn in point["context_turns"]
                if turn["session_id"] == point["target_session"]
            ]
            contexts[result_id] = "\n".join(
                f"{turn['speaker']}: {turn['content']}"
                for turn in within_session
            )
    missing = [row["result_id"] for row in rows if row["result_id"] not in contexts]
    if missing:
        raise ValueError(f"failed to reconstruct contexts for {missing[:3]}")
    return contexts


def _speaker_id(speaker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", speaker.casefold()).strip("_")


def run(
    predictions: Path,
    dataset_dir: Path,
    output_dir: Path,
    model: str,
    reference_checkpoint: Path | None = None,
) -> dict[str, Any]:
    api_key = os.environ["REALTALK_JUDGE_API_KEY"]
    base_url = os.environ["REALTALK_JUDGE_BASE_URL"]
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    contexts = _contexts(dataset_dir, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {"judgments": {}, "errors": {}}
    reused_reference_keys: list[str] = []
    if reference_checkpoint is not None:
        source_checkpoint = json.loads(reference_checkpoint.read_text(encoding="utf-8"))
        source_judgments = source_checkpoint.get("judgments", {})
        for row in rows:
            for metric in ("reflectiveness", "grounding", "empathy"):
                key = f"{row['result_id']}:reference:{metric}"
                if key not in source_judgments:
                    raise ValueError(f"reference checkpoint is missing {key}")
                if key not in checkpoint["judgments"]:
                    checkpoint["judgments"][key] = source_judgments[key]
                reused_reference_keys.append(key)
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    raw_path = output_dir / "raw_responses.jsonl"

    for row in rows:
        result_id = row["result_id"]
        history = contexts[result_id]
        for side, turn in (("reference", row["ground_truth"]), ("candidate", row["generated_message"])):
            for metric, template, parser in (
                ("reflectiveness", REFLECTIVENESS_PROMPT, _parse_bool),
                ("grounding", GROUNDING_PROMPT, _parse_bool),
                ("empathy", EMPATHY_PROMPT, _parse_empathy),
            ):
                key = f"{result_id}:{side}:{metric}"
                if key in checkpoint["judgments"]:
                    continue
                prompt = template.format(history=history or "(none)", turn=turn)
                try:
                    content, audit = _chat(base_url, api_key, model, prompt)
                    value = parser(content)
                    checkpoint["judgments"][key] = {"value": value, "audit": audit}
                    checkpoint["errors"].pop(key, None)
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"key": key, "raw": content, "audit": audit, "recorded_at_utc": _now()}, ensure_ascii=False) + "\n")
                except Exception as exc:
                    checkpoint["errors"][key] = {"type": type(exc).__name__, "error": str(exc), "recorded_at_utc": _now()}
                checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

    scored = []
    for row in rows:
        result_id = row["result_id"]
        values = {}
        complete = True
        for side in ("reference", "candidate"):
            values[side] = {}
            for metric in ("reflectiveness", "grounding", "empathy"):
                item = checkpoint["judgments"].get(f"{result_id}:{side}:{metric}")
                if item is None:
                    complete = False
                    break
                values[side][metric] = item["value"]
        if not complete:
            continue
        ref_emp = sum(values["reference"]["empathy"].values())
        cand_emp = sum(values["candidate"]["empathy"].values())
        scored.append({
            "result_id": result_id,
            "speaker": row["speaker"],
            "reference": values["reference"],
            "candidate": values["candidate"],
            "metrics": {
                "reflectiveness_accuracy": float(values["reference"]["reflectiveness"] == values["candidate"]["reflectiveness"]),
                "grounding_accuracy": float(values["reference"]["grounding"] == values["candidate"]["grounding"]),
                "empathy_absolute_difference": abs(ref_emp - cand_emp),
            },
        })
    metric_names = ("reflectiveness_accuracy", "grounding_accuracy", "empathy_absolute_difference")
    by_speaker = {
        speaker: {
            name: round(statistics.mean(
                item["metrics"][name] for item in scored
                if item["speaker"] == speaker
            ), 6)
            for name in metric_names
        }
        for speaker in dict.fromkeys(item["speaker"] for item in scored)
    }
    speaker_macro = {
        name: {
            "mean": round(statistics.mean(
                values[name] for values in by_speaker.values()
            ), 6),
            "std_population": round(statistics.pstdev(
                values[name] for values in by_speaker.values()
            ), 6),
        }
        for name in metric_names
    } if by_speaker else {}
    summary = {
        "status": "complete" if len(scored) == len(rows) and not checkpoint["errors"] else "incomplete",
        "scope": "small_subset_diagnostic_not_table2_main_result",
        "judge_protocol": "realtalk_appendix_c_full_prompt_within_session_v3",
        "model_requested": model,
        "messages_input": len(rows),
        "messages_scored": len(scored),
        "judgments_expected": len(rows) * 6,
        "judgments_complete": len(checkpoint["judgments"]),
        "unresolved_errors": len(checkpoint["errors"]),
        "reference_judgments_reused": len(set(reused_reference_keys)),
        "reference_checkpoint": (
            str(reference_checkpoint.resolve()) if reference_checkpoint else None
        ),
        "aggregation_for_table2": "speaker_macro_mean_and_population_std",
        "by_speaker": by_speaker,
        "speaker_macro": speaker_macro,
        "message_micro": {name: round(statistics.mean(item["metrics"][name] for item in scored), 6) for name in metric_names} if scored else {},
        "paper_table2_reference": PAPER_TABLE2,
        "comparison_warning": (
            f"Diagnostic subset spanning {len({item['speaker'] for item in scored})} speakers; "
            "paper values use the complete protocol and are not directly comparable."
        ),
        "created_at_utc": _now(),
    }
    with (output_dir / "scored.jsonl").open("w", encoding="utf-8") as handle:
        for item in scored:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--reference-checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(
        args.predictions,
        args.dataset_dir,
        args.output_dir,
        args.model,
        reference_checkpoint=args.reference_checkpoint,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

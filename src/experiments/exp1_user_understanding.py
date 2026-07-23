"""Experiment 1: causal understanding of the current observed user state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from ..epistemic_decay import compute_portrait_entropy
from ..llm_client import LLMClient
from ..prompts.templates_en import (
    FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT,
    FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
)
from ..utils import load_json
from .exp1_protocol import (
    build_message_level_points,
    build_profile_corpus,
    canonical_speaker,
    message_speakers,
    select_realtalk_splits,
    stable_hash,
)
from .exp1_metrics import (
    build_metric_records,
    classification_report,
    paired_correctness_counts,
    speaker_macro_report,
)
from .exp1_schema import (
    EMOTION_LABELS,
    REFERENCE_JUDGMENT_SCHEMA,
    SENTIMENT_LABELS,
    STATE_RESPONSE_SCHEMA,
    normalize_reference_judgment,
    normalize_state,
)
from .experiment_utils import robust_parse_json
from .operation_checkpoint import OperationCheckpoint
from .realtalk_evaluator import RealTalkLabelEvaluator
from .result_provenance import build_run_manifest


METHODS = ("self_model", "flat_profile", "explicit_model")
PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")
STATE_MAX_TOKENS = 2048
REFERENCE_JUDGMENT_MAX_TOKENS = 1024
PRIMARY_METRICS = ("emotion_accuracy", "sentiment_accuracy")
SUPPLEMENTARY_METRICS = ("emotion_macro_f1", "sentiment_macro_f1")
EXTENDED_METRICS = ("topic_consistency",)
PAPER_HIGHER_IS_BETTER_METRICS = (
    "reflectiveness_accuracy",
    "grounding_accuracy",
)
PAPER_LOWER_IS_BETTER_METRICS = (
    "intimacy_absolute_difference",
    "empathy_absolute_difference",
)

REFERENCE_JUDGE_SYSTEM_PROMPT = """You label one observed dialogue message using the REALTALK evaluation definitions.

Reflective is true only when the speaker explicitly shows self-observation, perspective-taking, or explains the motivation behind their own thoughts, feelings, or actions.
Grounding is true when the message actively builds mutual understanding through a clarification, follow-up inquiry, confirmation check, or request to expand shared information.
Empathy uses EPITOME: emotional_reaction, interpretation, and exploration are each integer scores from 0 (absent) to 2 (explicit and specific).
Topic is a concise description of the main subject of the observed message."""

STATE_SYSTEM_PROMPTS = {
    "self_model": (
        "You are a companion agent using self-model based other modeling. "
        "Infer the current state expressed by the observed user by projecting "
        "your own general emotional and conversational model onto their situation."
    ),
    "flat_profile": (
        "You are a user-understanding evaluator. Infer the current state "
        "expressed by the observed user using the conversation and the flat "
        "user profile."
    ),
    "explicit_model": (
        "You are a user-understanding evaluator. Infer the current state "
        "expressed by the observed user using the conversation and the "
        "five-layer user profile."
    ),
}


@dataclass
class Exp1Config:
    dataset_dir: str = "dataset"
    output_dir: str = "data/exp1_user_understanding"
    profile_sessions: int = 3
    test_sessions: int = 3
    max_context_chars: int = 60000
    profile_max_tokens: int = 16000
    max_eval_points_per_speaker: int = 0
    operation_max_attempts: int = 3
    chat_filter: Optional[List[str]] = None
    speaker_filter: Optional[List[str]] = None
    continue_on_error: bool = True
    fresh: bool = False


def run_exp1(
    config: Exp1Config,
    llm: Optional[LLMClient] = None,
    label_evaluator: Optional[RealTalkLabelEvaluator] = None,
) -> Dict[str, Any]:
    splits = select_realtalk_splits(
        config.dataset_dir, config.chat_filter, config.speaker_filter
    )
    if not splits:
        raise ValueError("no REALTALK speaker splits matched the configuration")
    llm = llm or LLMClient()
    label_evaluator = label_evaluator or RealTalkLabelEvaluator()

    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    if config.fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
    signature = _run_signature(
        config, llm, splits, label_evaluator.metadata()
    )
    checkpoint = OperationCheckpoint(checkpoint_path, signature)
    started = perf_counter()
    run_failures: List[Dict[str, Any]] = []
    print(
        f"[Exp1] speakers={len(splits)} model={getattr(llm, 'model', None)} "
        f"profile_sessions={config.profile_sessions} "
        f"test_sessions={config.test_sessions} "
        f"resume_results={len(checkpoint.result_values())}"
    )

    dataset_dir = Path(config.dataset_dir)
    for split in splits:
        train_file = dataset_dir / split["train_chat"]
        test_file = dataset_dir / split["test_chat"]
        train_chat = load_json(str(train_file))
        test_chat = load_json(str(test_file))
        train_speaker = canonical_speaker(train_chat, split["speaker"])
        test_speaker = canonical_speaker(test_chat, split["speaker"])
        if train_speaker.casefold() != test_speaker.casefold():
            raise ValueError(
                f"speaker mismatch across split: {train_speaker!r} vs {test_speaker!r}"
            )
        target_speaker = test_speaker
        partner_speaker = next(
            speaker for speaker in message_speakers(test_chat)
            if speaker.casefold() != target_speaker.casefold()
        )
        profile_corpus = build_profile_corpus(
            train_chat, target_speaker, config.profile_sessions
        )
        points = build_message_level_points(
            test_chat,
            target_speaker,
            test_sessions=config.test_sessions,
            max_context_chars=config.max_context_chars,
            max_eval_points=config.max_eval_points_per_speaker,
        )
        print(
            f"[Exp1] speaker={target_speaker} train={train_file.name} "
            f"test={test_file.name} points={len(points)}"
        )
        flat_profile, flat_key = _cached_profile(
            checkpoint,
            llm,
            config,
            split,
            profile_corpus,
            target_speaker,
            "flat",
        )
        explicit_profile, explicit_key = _cached_profile(
            checkpoint,
            llm,
            config,
            split,
            profile_corpus,
            target_speaker,
            "explicit",
        )
        method_inputs = {
            "self_model": "",
            "flat_profile": json.dumps(flat_profile, ensure_ascii=False, indent=2),
            "explicit_model": json.dumps(
                explicit_profile, ensure_ascii=False, indent=2
            ),
        }

        for index, point in enumerate(points, start=1):
            result_id = (
                f"{_speaker_id(target_speaker)}:{test_file.stem}:"
                f"{point['sample_id']}"
            )
            if result_id in checkpoint.data["results"]:
                print(f"  [{index}/{len(points)}] resume {point['eval_id']}")
                continue
            sample_started = perf_counter()
            print(f"  [{index}/{len(points)}] evaluate {point['eval_id']}")
            try:
                reference = _cached_reference(
                    checkpoint, llm, label_evaluator, config, result_id,
                    target_speaker,
                    point["context_text"],
                    point["target_message"],
                )

                method_results: Dict[str, Any] = {}
                for method in METHODS:
                    prediction = _cached_prediction(
                        checkpoint,
                        llm,
                        config,
                        result_id,
                        method,
                        point["context_text"],
                        point["target_message"],
                        method_inputs[method],
                    )
                    method_results[method] = {
                        "prediction": prediction,
                        "scores": score_prediction(prediction, reference),
                    }

                result = {
                    "result_id": result_id,
                    "speaker": target_speaker,
                    "train_chat_file": train_file.name,
                    "test_chat_file": test_file.name,
                    "chat_file": test_file.name,
                    "eval_id": point["eval_id"],
                    "message_level_index": point["message_level_index"],
                    "target_session": point["target_session"],
                    "user_speaker": target_speaker,
                    "partner_speaker": partner_speaker,
                    "target_message": point["target_message"],
                    "target_dia_ids": point["target"].get("dia_ids", []),
                    "reference": reference,
                    "methods": method_results,
                    "profile": {
                        "source": "fixed_cross_conversation_ca",
                        "history_hash": profile_corpus["history_hash"],
                        "train_sessions": profile_corpus["sessions"],
                        "flat_cache_key": flat_key,
                        "explicit_cache_key": explicit_key,
                        "flat_characters": len(method_inputs["flat_profile"]),
                        "explicit_characters": len(method_inputs["explicit_model"]),
                        "explicit_portrait_entropy": compute_portrait_entropy(
                            explicit_profile
                        ),
                    },
                    "context": {
                        "source": "rolling_real_history_within_selected_cb",
                        "test_sessions": point["test_sessions"],
                        "actual_session_count": point["context_session_count"],
                        "semantic_turns": len(point["context_turns"]),
                        "characters": len(point["context_text"]),
                        "truncated": point["context_truncated"],
                        "history_hash": point["history_hash"],
                    },
                    "sample_elapsed_seconds": round(
                        perf_counter() - sample_started, 3
                    ),
                    "completed_at_utc": _now(),
                    "status": "complete",
                }
                checkpoint.store_result(result_id, result)
            except Exception as exc:
                failure = {
                    "result_id": result_id,
                    "speaker": target_speaker,
                    "train_chat_file": train_file.name,
                    "test_chat_file": test_file.name,
                    "eval_id": point["eval_id"],
                    "status": "excluded_incomplete_triplet",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "updated_at_utc": _now(),
                }
                checkpoint.store_excluded_result(result_id, failure)
                run_failures.append(failure)
                print(f"  [excluded] {type(exc).__name__}: {exc}")
                if not config.continue_on_error:
                    raise

    results = sorted(
        checkpoint.result_values(), key=lambda item: item["result_id"]
    )
    summary = aggregate_results(results)
    summary.update({
        "elapsed_seconds": round(perf_counter() - started, 3),
        "failed_samples_excluded_this_run": run_failures,
        "failed_sample_count_all_runs": sum(
            key.startswith("sample:") for key in checkpoint.data["failures"]
        ),
        "token_usage": _checkpoint_token_usage(checkpoint),
        "primary_aggregation": "speaker_macro",
        "realtalk_alignment": {
            "task_adaptation": (
                "REALTALK message-level persona simulation protocol adapted "
                "from next-message generation to observed current-state inference"
            ),
            "split": "speaker-specific Ca train/profile chat and Cb test chat from Table 8",
            "context": (
                "three-session paper setting; each Cb target receives all prior "
                "real merged turns in the selected test segment"
            ),
            "emotion_and_sentiment": "pinned REALTALK classifiers on human target messages",
            "reflectiveness_grounding_empathy": (
                "strict-schema LLM judgments using REALTALK definitions and "
                "the observed human target message with its prior context"
            ),
            "intimacy": "pinned REALTALK intimacy regressor on human target messages",
            "topic": (
                "Exp1-specific extension retained for analysis only; excluded "
                "from primary ranking and significance comparisons"
            ),
            "persona_consistency": (
                "separate diagnostic using absolute cross-conversation differences "
                "for Exp1-available EI attributes only"
            ),
        },
    })
    manifest = build_run_manifest(
        {**asdict(config), "fresh": False}, getattr(llm, "model", None)
    )
    manifest.update({
        "run_signature": signature,
        "reference_evaluator": label_evaluator.metadata(),
        "reference_judge": {
            "provider": "same_llm_client_as_method_predictions",
            "model": getattr(llm, "model", None),
            "temperature": 0.0,
            "max_tokens": REFERENCE_JUDGMENT_MAX_TOKENS,
            "note": (
                "Use an independently configured judge in the final paper run "
                "if the evaluation protocol requires generator-judge separation."
            ),
        },
        "response_schemas": {
            "method_prediction": STATE_RESPONSE_SCHEMA,
            "reference_judgment": REFERENCE_JUDGMENT_SCHEMA,
        },
        "method_prediction_max_tokens": STATE_MAX_TOKENS,
        "metric_protocol": _metric_protocol(),
        "output_contract": {
            "results.jsonl": "canonical complete per-sample triplets",
            "metric_records.jsonl": (
                "derived long-form records for offline statistics and audits"
            ),
            "summary.json": "aggregated metrics and diagnostics",
            "checkpoint.json": "resumable operation cache",
        },
        "network_retry_max_attempts": getattr(llm, "max_retries", None),
        "operation_retry_max_attempts": config.operation_max_attempts,
        "prompt_hashes": _prompt_hashes(),
        "realtalk_speaker_splits": splits,
    })
    _write_outputs(output_dir, results, summary, manifest)
    print(f"[Exp1] summary={json.dumps(summary.get('comparison', {}), indent=2)}")
    return summary


def _speaker_id(speaker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", speaker.casefold()).strip("_")


def _cached_profile(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp1Config,
    split: Dict[str, str],
    profile_corpus: Dict[str, Any],
    user_speaker: str,
    profile_type: str,
) -> tuple[Dict[str, Any], str]:
    prompt_hash = _prompt_hashes()[f"{profile_type}_profile"]
    key = ":".join((
        "profile",
        _speaker_id(user_speaker),
        Path(split["train_chat"]).stem,
        f"sessions_{config.profile_sessions}",
        profile_type,
        profile_corpus["history_hash"],
        str(getattr(llm, "model", "unknown")),
        prompt_hash,
    ))

    def operation() -> Dict[str, Any]:
        if profile_type == "flat":
            system_prompt = FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT.format(
                user_name=user_speaker
            )
            user_prompt = FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                user_name=user_speaker, corpus=profile_corpus["text"]
            )
        else:
            system_prompt = PROFILE_EXTRACTION_SYSTEM_PROMPT.format(
                user_name=user_speaker
            )
            user_prompt = PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                user_name=user_speaker, corpus=profile_corpus["text"]
            )
        return robust_parse_json(llm.chat(
            system_prompt,
            user_prompt,
            temperature=0.3,
            max_tokens=config.profile_max_tokens,
        ))

    validator = _validate_flat_profile if profile_type == "flat" else _validate_explicit_profile
    value = checkpoint.execute(
        key,
        operation,
        validator,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )
    return value, key


def _cached_reference(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    evaluator: RealTalkLabelEvaluator,
    config: Exp1Config,
    result_id: str,
    target_speaker: str,
    history_text: str,
    target_message: str,
) -> Dict[str, Any]:
    label_key = f"reference_labels:{result_id}:{stable_hash(target_message)}"
    labels = checkpoint.execute(
        label_key,
        lambda: evaluator.annotate(target_message),
        _validate_reference_labels,
        max_attempts=config.operation_max_attempts,
    )
    judgment_hash = stable_hash({
        "speaker": target_speaker,
        "history": history_text,
        "target": target_message,
        "system": REFERENCE_JUDGE_SYSTEM_PROMPT,
        "schema": REFERENCE_JUDGMENT_SCHEMA,
    })
    judgment_key = f"reference_judgment:{result_id}:{judgment_hash}"
    user_prompt = (
        f"CONVERSATION HISTORY:\n{history_text or '(none)'}\n\n"
        f"CURRENT OBSERVED MESSAGE BY {target_speaker}:\n{target_message}\n\n"
        "Judge only the current observed message, using its prior context."
    )
    judgment = checkpoint.execute(
        judgment_key,
        lambda: _reference_judgment_call(
            llm,
            REFERENCE_JUDGE_SYSTEM_PROMPT,
            user_prompt,
        ),
        normalize_reference_judgment,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )
    return {
        **labels,
        **judgment,
    }


def _cached_prediction(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp1Config,
    result_id: str,
    method: str,
    history_text: str,
    target_message: str,
    method_context: str,
) -> Dict[str, str]:
    input_hash = stable_hash({
        "history": history_text,
        "target": target_message,
        "method_context": method_context,
        "system": STATE_SYSTEM_PROMPTS[method],
        "schema": STATE_RESPONSE_SCHEMA,
    })
    key = f"prediction:{result_id}:{method}:{input_hash}"
    user_prompt = (
        f"CONVERSATION HISTORY:\n{history_text or '(none)'}\n\n"
        f"CURRENT OBSERVED USER MESSAGE:\n{target_message}\n\n"
    )
    if method != "self_model":
        profile_label = (
            "FLAT USER PROFILE"
            if method == "flat_profile"
            else "FIVE-LAYER USER PROFILE"
        )
        user_prompt += f"{profile_label}:\n{method_context}\n\n"
    user_prompt += "Infer the current state expressed in the observed user message."
    return checkpoint.execute(
        key,
        lambda: _state_call(llm, STATE_SYSTEM_PROMPTS[method], user_prompt),
        normalize_state,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _state_call(llm: LLMClient, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    raw = llm.chat(
        system_prompt,
        user_prompt,
        temperature=0.2,
        max_tokens=STATE_MAX_TOKENS,
        response_schema=STATE_RESPONSE_SCHEMA,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("strict schema response was not valid JSON") from exc


def _reference_judgment_call(
    llm: LLMClient, system_prompt: str, user_prompt: str
) -> Dict[str, Any]:
    raw = llm.chat(
        system_prompt,
        user_prompt,
        temperature=0.0,
        max_tokens=REFERENCE_JUDGMENT_MAX_TOKENS,
        response_schema=REFERENCE_JUDGMENT_SCHEMA,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("strict reference judgment was not valid JSON") from exc


def _validate_flat_profile(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or not value or value.get("error"):
        raise ValueError("flat profile is empty or invalid")
    return value


def _validate_explicit_profile(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("error"):
        raise ValueError("explicit profile is invalid")
    missing = [layer for layer in PROFILE_LAYERS if not isinstance(value.get(layer), dict)]
    if missing:
        raise ValueError(f"explicit profile is missing layers: {missing}")
    return value


def _validate_reference_labels(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("reference labels must be an object")
    emotion = str(value.get("emotion", "")).strip().lower()
    sentiment = str(value.get("sentiment", "")).strip().lower()
    intimacy = value.get("intimacy")
    if emotion not in EMOTION_LABELS:
        raise ValueError(f"unsupported reference emotion label: {emotion}")
    if sentiment not in SENTIMENT_LABELS:
        raise ValueError(f"unsupported reference sentiment label: {sentiment}")
    if isinstance(intimacy, bool) or not isinstance(intimacy, (int, float)):
        raise ValueError("reference intimacy must be numeric")
    intimacy = float(intimacy)
    if not 0 <= intimacy <= 1:
        raise ValueError("reference intimacy must be in [0, 1]")
    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "intimacy": intimacy,
    }


def score_prediction(
    prediction: Dict[str, Any], reference: Dict[str, Any]
) -> Dict[str, float]:
    return {
        "emotion_accuracy": float(prediction["emotion"] == reference["emotion"]),
        "sentiment_accuracy": float(
            prediction["sentiment"] == reference["sentiment"]
        ),
        "topic_consistency": round(
            _topic_overlap(prediction["topic"], reference["topic"]), 4
        ),
        "reflectiveness_accuracy": float(
            prediction["reflective"] == reference["reflective"]
        ),
        "grounding_accuracy": float(
            prediction["grounding"] == reference["grounding"]
        ),
        "intimacy_absolute_difference": round(
            abs(float(prediction["intimacy"]) - float(reference["intimacy"])), 4
        ),
        "empathy_absolute_difference": round(
            abs(_empathy_total(prediction["empathy"])
                - _empathy_total(reference["empathy"])),
            4,
        ),
    }


def _empathy_total(empathy: Dict[str, int]) -> int:
    return sum(
        int(empathy[field])
        for field in ("emotional_reaction", "interpretation", "exploration")
    )


def _topic_overlap(prediction: str, reference: str) -> float:
    predicted_words = set(re.findall(r"[\w']+", prediction.lower()))
    reference_words = set(re.findall(r"[\w']+", reference.lower()))
    return len(predicted_words & reference_words) / max(len(reference_words), 1)


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "comparison": {},
            "num_eval_points": 0,
            "num_speakers": 0,
            "persona_consistency_diagnostic": {},
            "metric_protocol": _metric_protocol(),
        }
    by_speaker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_speaker[result["speaker"]].append(result)

    comparison: Dict[str, Any] = {}
    for method in METHODS:
        per_speaker: List[Dict[str, float]] = []
        all_scores: List[Dict[str, float]] = []
        emotion_records: List[Dict[str, str]] = []
        sentiment_records: List[Dict[str, str]] = []
        for speaker, speaker_results in by_speaker.items():
            scores = [item["methods"][method]["scores"] for item in speaker_results]
            all_scores.extend(scores)
            emotion_records.extend({
                "speaker": speaker,
                "chat_file": item["chat_file"],
                "reference": item["reference"]["emotion"],
                "prediction": item["methods"][method]["prediction"]["emotion"],
            } for item in speaker_results)
            sentiment_records.extend({
                "speaker": speaker,
                "chat_file": item["chat_file"],
                "reference": item["reference"]["sentiment"],
                "prediction": item["methods"][method]["prediction"]["sentiment"],
            } for item in speaker_results)
            per_speaker.append({
                metric: _mean(score[metric] for score in scores)
                for metric in (
                    "emotion_accuracy",
                    "sentiment_accuracy",
                    "topic_consistency",
                    *PAPER_HIGHER_IS_BETTER_METRICS,
                    *PAPER_LOWER_IS_BETTER_METRICS,
                )
            })

        emotion_global = classification_report(
            [record["reference"] for record in emotion_records],
            [record["prediction"] for record in emotion_records],
            EMOTION_LABELS,
        )
        sentiment_global = classification_report(
            [record["reference"] for record in sentiment_records],
            [record["prediction"] for record in sentiment_records],
            SENTIMENT_LABELS,
        )
        emotion_speaker_macro = speaker_macro_report(
            emotion_records, EMOTION_LABELS
        )
        sentiment_speaker_macro = speaker_macro_report(
            sentiment_records, SENTIMENT_LABELS
        )
        speaker_macro = {
                "emotion_accuracy": round(
                    _mean(speaker["emotion_accuracy"] for speaker in per_speaker), 4
                ),
                "sentiment_accuracy": round(
                    _mean(speaker["sentiment_accuracy"] for speaker in per_speaker), 4
                ),
                "emotion_macro_f1": round(
                    emotion_speaker_macro["macro_f1"], 4
                ),
                "sentiment_macro_f1": round(
                    sentiment_speaker_macro["macro_f1"], 4
                ),
                "topic_consistency": round(
                    _mean(speaker["topic_consistency"] for speaker in per_speaker), 4
                ),
        }
        micro = {
                "emotion_accuracy": round(emotion_global["accuracy"], 4),
                "sentiment_accuracy": round(sentiment_global["accuracy"], 4),
                "emotion_macro_f1": round(emotion_global["macro_f1"], 4),
                "sentiment_macro_f1": round(sentiment_global["macro_f1"], 4),
                "topic_consistency": round(
                    _mean(score["topic_consistency"] for score in all_scores), 4
                ),
        }
        for metric in (
            *PAPER_HIGHER_IS_BETTER_METRICS,
            *PAPER_LOWER_IS_BETTER_METRICS,
        ):
            speaker_macro[metric] = round(
                _mean(speaker[metric] for speaker in per_speaker), 4
            )
            micro[metric] = round(
                _mean(score[metric] for score in all_scores), 4
            )
        comparison[method] = {
            "speaker_macro": speaker_macro,
            "micro": micro,
            "classification_details": {
                "emotion": {
                    "global": emotion_global,
                    "speaker_macro": emotion_speaker_macro,
                },
                "sentiment": {
                    "global": sentiment_global,
                    "speaker_macro": sentiment_speaker_macro,
                },
            },
            "num_evaluations": len(all_scores),
        }

    entropy_by_speaker = {
        speaker: round(_mean(
            item["profile"]["explicit_portrait_entropy"] for item in speaker_results
        ), 4)
        for speaker, speaker_results in by_speaker.items()
    }
    return {
        "comparison": comparison,
        "improvement_speaker_macro": _improvements(comparison),
        "extended_improvement_speaker_macro": _extended_improvements(comparison),
        "paired_outcomes": _paired_outcomes(results),
        "metric_protocol": _metric_protocol(),
        "num_eval_points": len(results),
        "num_speakers": len(by_speaker),
        "portrait_entropy": {
            "by_speaker": entropy_by_speaker,
            "macro_average": round(_mean(entropy_by_speaker.values()), 4),
        },
        "persona_consistency_diagnostic": {
            "status": "not_part_of_method_ranking",
            "reason": (
                "REALTALK persona consistency is a separate full-conversation "
                "dataset diagnostic, not a Cb current-state metric"
            ),
        },
    }


def _improvements(comparison: Dict[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for baseline in ("self_model", "flat_profile"):
        for metric in (
            PRIMARY_METRICS
            + SUPPLEMENTARY_METRICS
            + PAPER_HIGHER_IS_BETTER_METRICS
        ):
            values[f"explicit_vs_{baseline}_{metric}"] = round(
                comparison["explicit_model"]["speaker_macro"][metric]
                - comparison[baseline]["speaker_macro"][metric],
                4,
            )
        for metric in PAPER_LOWER_IS_BETTER_METRICS:
            values[f"explicit_vs_{baseline}_{metric}_reduction"] = round(
                comparison[baseline]["speaker_macro"][metric]
                - comparison["explicit_model"]["speaker_macro"][metric],
                4,
            )
    return values


def _extended_improvements(comparison: Dict[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for baseline in ("self_model", "flat_profile"):
        for metric in EXTENDED_METRICS:
            values[f"explicit_vs_{baseline}_{metric}"] = round(
                comparison["explicit_model"]["speaker_macro"][metric]
                - comparison[baseline]["speaker_macro"][metric],
                4,
            )
    return values


def _paired_outcomes(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        f"explicit_model_vs_{baseline}": {
            field: paired_correctness_counts(
                results, "explicit_model", baseline, field
            )
            for field in ("emotion", "sentiment", "reflective", "grounding")
        }
        for baseline in ("self_model", "flat_profile")
    }


def _metric_protocol() -> Dict[str, Any]:
    return {
        "primary_metrics": list(PRIMARY_METRICS),
        "supplementary_metrics": list(SUPPLEMENTARY_METRICS),
        "paper_aligned_higher_is_better_metrics": list(
            PAPER_HIGHER_IS_BETTER_METRICS
        ),
        "paper_aligned_lower_is_better_metrics": list(
            PAPER_LOWER_IS_BETTER_METRICS
        ),
        "extended_metrics": list(EXTENDED_METRICS),
        "primary_ranking_aggregation": "speaker_macro",
        "macro_f1": (
            "unweighted mean over labels present in reference or prediction; "
            "fixed-label value is also retained in classification_details"
        ),
        "topic_policy": (
            "retained unchanged for exploratory analysis; not used for primary "
            "ranking or paired outcome comparisons"
        ),
        "paper_alignment": (
            "emotion, sentiment, reflectiveness, grounding, intimacy, and "
            "EPITOME empathy preserve the raw values required by REALTALK-style "
            "evaluation; ROUGE and BERTScore are inapplicable because Exp1 "
            "classifies an observed message rather than generating one"
        ),
        "difference_direction": (
            "positive improvement means Explicit is better; accuracy deltas are "
            "Explicit minus baseline, absolute-difference reductions are baseline "
            "minus Explicit"
        ),
        "paired_outcomes": (
            "raw paired correctness contingency counts only; formal statistical "
            "test and confidence interval can be recomputed without API calls"
        ),
    }


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _checkpoint_token_usage(checkpoint: OperationCheckpoint) -> Dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    for operation in checkpoint.data["operations"].values():
        for key in total:
            total[key] += int(operation.get("token_usage", {}).get(key, 0))
    return total


def _prompt_hashes() -> Dict[str, str]:
    return {
        "flat_profile": stable_hash(
            FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT
            + FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
        ),
        "explicit_profile": stable_hash(
            PROFILE_EXTRACTION_SYSTEM_PROMPT + PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
        ),
        **{
            f"state_{method}": stable_hash(prompt)
            for method, prompt in STATE_SYSTEM_PROMPTS.items()
        },
        "reference_judgment": stable_hash(
            REFERENCE_JUDGE_SYSTEM_PROMPT
            + json.dumps(
                REFERENCE_JUDGMENT_SCHEMA,
                ensure_ascii=True,
                sort_keys=True,
            )
        ),
    }


def _run_signature(
    config: Exp1Config,
    llm: LLMClient,
    splits: List[Dict[str, str]],
    reference_metadata: Dict[str, Any],
) -> str:
    signature_config = asdict(config)
    for key in ("output_dir", "continue_on_error", "fresh"):
        signature_config.pop(key, None)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("exp1_metrics.py"),
        Path(__file__).with_name("exp1_protocol.py"),
        Path(__file__).with_name("exp1_schema.py"),
        Path(__file__).with_name("operation_checkpoint.py"),
        Path(__file__).with_name("realtalk_evaluator.py"),
    ]
    dataset_dir = Path(config.dataset_dir)
    chat_files = sorted({
        dataset_dir / split[key]
        for split in splits
        for key in ("train_chat", "test_chat")
    })
    return stable_hash({
        "schema_version": 2,
        "model": getattr(llm, "model", None),
        "enable_thinking": getattr(llm, "enable_thinking", None),
        "reference_evaluator": reference_metadata,
        "config": signature_config,
        "realtalk_speaker_splits": splits,
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "dataset_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in chat_files
        },
    })


def _write_outputs(
    output_dir: Path,
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    manifest: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output_dir / "results.jsonl",
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
    )
    metric_records = build_metric_records(results, METHODS)
    _atomic_text(
        output_dir / "metric_records.jsonl",
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in metric_records
        ),
    )
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(output_dir / "run_manifest.json", manifest)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 1: causal current-user-state understanding"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp1_user_understanding")
    parser.add_argument("--profile-sessions", type=int, default=3)
    parser.add_argument("--test-sessions", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--profile-max-tokens", type=int, default=16000)
    parser.add_argument(
        "--max-eval-points",
        type=int,
        default=0,
        help="Per speaker; 0 runs every merged target message.",
    )
    parser.add_argument("--operation-max-attempts", type=int, default=3)
    parser.add_argument("--chats", nargs="*", default=None)
    parser.add_argument("--speakers", nargs="*", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    run_exp1(Exp1Config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        profile_sessions=args.profile_sessions,
        test_sessions=args.test_sessions,
        max_context_chars=args.max_context_chars,
        profile_max_tokens=args.profile_max_tokens,
        max_eval_points_per_speaker=args.max_eval_points,
        operation_max_attempts=args.operation_max_attempts,
        chat_filter=args.chats,
        speaker_filter=args.speakers,
        continue_on_error=not args.fail_fast,
        fresh=args.fresh,
    ))


if __name__ == "__main__":
    main()

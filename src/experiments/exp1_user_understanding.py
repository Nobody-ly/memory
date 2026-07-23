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
    build_session_boundary_points,
    resolve_chat_roles,
    stable_hash,
)
from .exp1_metrics import (
    build_metric_records,
    chat_macro_report,
    classification_report,
    paired_correctness_counts,
)
from .exp1_schema import (
    EMOTION_LABELS,
    SENTIMENT_LABELS,
    STATE_RESPONSE_SCHEMA,
    normalize_state,
)
from .experiment_utils import load_chat_files, robust_parse_json
from .operation_checkpoint import OperationCheckpoint
from .realtalk_evaluator import RealTalkLabelEvaluator
from .result_provenance import build_run_manifest


METHODS = ("self_model", "flat_profile", "explicit_model")
PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")
PRIMARY_METRICS = ("emotion_accuracy", "sentiment_accuracy")
SUPPLEMENTARY_METRICS = ("emotion_macro_f1", "sentiment_macro_f1")
EXTENDED_METRICS = ("topic_consistency",)

STATE_SYSTEM_PROMPTS = {
    "self_model": (
        "You are a companion agent using self-model based other modeling. "
        "Infer the current state expressed by the observed user by projecting "
        "your own persona and perspective onto their situation."
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
    min_context_sessions: int = 2
    context_sessions: Optional[int] = 3
    max_context_chars: int = 60000
    profile_max_tokens: int = 8000
    max_eval_points_per_chat: int = 15
    operation_max_attempts: int = 3
    chat_filter: Optional[List[str]] = None
    continue_on_error: bool = True
    fresh: bool = False


def run_exp1(
    config: Exp1Config,
    llm: Optional[LLMClient] = None,
    label_evaluator: Optional[RealTalkLabelEvaluator] = None,
) -> Dict[str, Any]:
    chat_files = load_chat_files(config.dataset_dir, config.chat_filter)
    if not chat_files:
        raise ValueError("no Chat_*.json files matched the experiment configuration")
    llm = llm or LLMClient()
    label_evaluator = label_evaluator or RealTalkLabelEvaluator()

    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    if config.fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
    signature = _run_signature(
        config, llm, chat_files, label_evaluator.metadata()
    )
    checkpoint = OperationCheckpoint(checkpoint_path, signature)
    started = perf_counter()
    run_failures: List[Dict[str, Any]] = []
    print(
        f"[Exp1] chats={len(chat_files)} model={getattr(llm, 'model', None)} "
        f"context_sessions={config.context_sessions or 'all'} "
        f"resume_results={len(checkpoint.result_values())}"
    )

    for chat_file in chat_files:
        chat = load_json(str(chat_file))
        user_speaker, agent_speaker, role_warnings = resolve_chat_roles(chat)
        for warning in role_warnings:
            print(f"[Exp1 role warning] {chat_file.name}: {warning}")
        persona = _load_agent_persona(config.dataset_dir, agent_speaker)
        points = build_session_boundary_points(
            chat,
            user_speaker,
            min_context_sessions=config.min_context_sessions,
            context_sessions=config.context_sessions,
            max_context_chars=config.max_context_chars,
            max_eval_points=config.max_eval_points_per_chat,
        )
        print(
            f"[Exp1] {chat_file.name}: user={user_speaker} "
            f"agent={agent_speaker} points={len(points)}"
        )

        for index, point in enumerate(points, start=1):
            result_id = f"{chat_file.stem}:{point['sample_id']}"
            if result_id in checkpoint.data["results"]:
                print(f"  [{index}/{len(points)}] resume {point['eval_id']}")
                continue
            sample_started = perf_counter()
            print(f"  [{index}/{len(points)}] evaluate {point['eval_id']}")
            try:
                flat_profile, flat_key = _cached_profile(
                    checkpoint, llm, config, chat_file.stem, point, user_speaker, "flat"
                )
                explicit_profile, explicit_key = _cached_profile(
                    checkpoint, llm, config, chat_file.stem, point, user_speaker, "explicit"
                )
                reference = _cached_reference(
                    checkpoint, llm, label_evaluator, config, result_id,
                    point["target_message"],
                )

                method_inputs = {
                    "self_model": json.dumps(persona, ensure_ascii=False, indent=2),
                    "flat_profile": json.dumps(flat_profile, ensure_ascii=False, indent=2),
                    "explicit_model": json.dumps(
                        explicit_profile, ensure_ascii=False, indent=2
                    ),
                }
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
                    "chat_file": chat_file.name,
                    "eval_id": point["eval_id"],
                    "boundary_index": point["boundary_index"],
                    "target_session": point["target_session"],
                    "user_speaker": user_speaker,
                    "agent_speaker": agent_speaker,
                    "role_warnings": role_warnings,
                    "target_message": point["target_message"],
                    "target_dia_ids": point["target"].get("dia_ids", []),
                    "reference": reference,
                    "methods": method_results,
                    "profile": {
                        "history_hash": point["profile_history_hash"],
                        "completed_sessions": point["completed_sessions"],
                        "flat_cache_key": flat_key,
                        "explicit_cache_key": explicit_key,
                        "flat_characters": len(method_inputs["flat_profile"]),
                        "explicit_characters": len(method_inputs["explicit_model"]),
                        "explicit_portrait_entropy": compute_portrait_entropy(
                            explicit_profile
                        ),
                    },
                    "context": {
                        "configured_sessions": config.context_sessions,
                        "actual_session_count": point["context_session_count"],
                        "semantic_turns": len(point["context_turns"]),
                        "characters": len(point["context_text"]),
                        "truncated": point["context_truncated"],
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
                    "chat_file": chat_file.name,
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
        "primary_aggregation": "chat_macro",
        "realtalk_alignment": {
            "emotion_and_sentiment": "pinned REALTALK classifiers on human target messages",
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
        "response_schema": STATE_RESPONSE_SCHEMA,
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
    })
    _write_outputs(output_dir, results, summary, manifest)
    print(f"[Exp1] summary={json.dumps(summary.get('comparison', {}), indent=2)}")
    return summary


def _load_agent_persona(dataset_dir: str, agent_speaker: str) -> Dict[str, Any]:
    filename = re.sub(r"\s+", "_", agent_speaker.strip().lower()) + "_persona.json"
    path = Path(dataset_dir) / "output" / "agent" / filename
    if not path.exists():
        raise FileNotFoundError(f"agent persona is required but missing: {path}")
    persona = load_json(str(path))
    if not isinstance(persona, dict) or not persona:
        raise ValueError(f"agent persona is empty or invalid: {path}")
    return persona


def _cached_profile(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp1Config,
    chat_stem: str,
    point: Dict[str, Any],
    user_speaker: str,
    profile_type: str,
) -> tuple[Dict[str, Any], str]:
    prompt_hash = _prompt_hashes()[f"{profile_type}_profile"]
    key = ":".join((
        "profile",
        chat_stem,
        str(point["boundary_index"]),
        profile_type,
        point["profile_history_hash"],
        str(getattr(llm, "model", "unknown")),
        prompt_hash,
    ))

    def operation() -> Dict[str, Any]:
        if profile_type == "flat":
            system_prompt = FLAT_PROFILE_EXTRACTION_SYSTEM_PROMPT.format(
                user_name=user_speaker
            )
            user_prompt = FLAT_PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                user_name=user_speaker, corpus=point["profile_text"]
            )
        else:
            system_prompt = PROFILE_EXTRACTION_SYSTEM_PROMPT.format(
                user_name=user_speaker
            )
            user_prompt = PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                user_name=user_speaker, corpus=point["profile_text"]
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
    target_message: str,
) -> Dict[str, str]:
    label_key = f"reference_labels:{result_id}:{stable_hash(target_message)}"
    labels = checkpoint.execute(
        label_key,
        lambda: evaluator.annotate(target_message),
        _validate_reference_labels,
        max_attempts=config.operation_max_attempts,
    )
    topic_key = f"reference_topic:{result_id}:{stable_hash(target_message)}"
    topic_state = checkpoint.execute(
        topic_key,
        lambda: _state_call(
            llm,
            "You identify the current state expressed in one observed user message.",
            f"CURRENT USER MESSAGE:\n{target_message}\n\nIdentify its current state.",
        ),
        normalize_state,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )
    return {
        "emotion": labels["emotion"],
        "sentiment": labels["sentiment"],
        "topic": topic_state["topic"],
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
        f"CURRENT USER MESSAGE:\n{target_message}\n\n"
        f"{'AGENT PERSONA' if method == 'self_model' else 'USER PROFILE'}:\n"
        f"{method_context}\n\n"
        "Infer the current state expressed in the current user message."
    )
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
        max_tokens=512,
        response_schema=STATE_RESPONSE_SCHEMA,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("strict schema response was not valid JSON") from exc


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


def _validate_reference_labels(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("reference labels must be an object")
    normalized = normalize_state({
        "emotion": value.get("emotion"),
        "sentiment": value.get("sentiment"),
        "topic": "reference-placeholder",
    })
    return {"emotion": normalized["emotion"], "sentiment": normalized["sentiment"]}


def score_prediction(
    prediction: Dict[str, str], reference: Dict[str, str]
) -> Dict[str, float]:
    return {
        "emotion_accuracy": float(prediction["emotion"] == reference["emotion"]),
        "sentiment_accuracy": float(
            prediction["sentiment"] == reference["sentiment"]
        ),
        "topic_consistency": round(
            _topic_overlap(prediction["topic"], reference["topic"]), 4
        ),
    }


def _topic_overlap(prediction: str, reference: str) -> float:
    predicted_words = set(re.findall(r"[\w']+", prediction.lower()))
    reference_words = set(re.findall(r"[\w']+", reference.lower()))
    return len(predicted_words & reference_words) / max(len(reference_words), 1)


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "comparison": {},
            "num_eval_points": 0,
            "num_chats": 0,
            "persona_consistency_diagnostic": {},
            "metric_protocol": _metric_protocol(),
        }
    by_chat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_chat[result["chat_file"]].append(result)

    comparison: Dict[str, Any] = {}
    for method in METHODS:
        per_chat: List[Dict[str, float]] = []
        all_scores: List[Dict[str, float]] = []
        emotion_records: List[Dict[str, str]] = []
        sentiment_records: List[Dict[str, str]] = []
        for chat_results in by_chat.values():
            scores = [item["methods"][method]["scores"] for item in chat_results]
            all_scores.extend(scores)
            emotion_records.extend({
                "chat_file": item["chat_file"],
                "reference": item["reference"]["emotion"],
                "prediction": item["methods"][method]["prediction"]["emotion"],
            } for item in chat_results)
            sentiment_records.extend({
                "chat_file": item["chat_file"],
                "reference": item["reference"]["sentiment"],
                "prediction": item["methods"][method]["prediction"]["sentiment"],
            } for item in chat_results)
            per_chat.append({
                metric: _mean(score[metric] for score in scores)
                for metric in (
                    "emotion_accuracy", "sentiment_accuracy", "topic_consistency"
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
        emotion_chat_macro = chat_macro_report(emotion_records, EMOTION_LABELS)
        sentiment_chat_macro = chat_macro_report(
            sentiment_records, SENTIMENT_LABELS
        )
        comparison[method] = {
            "chat_macro": {
                "emotion_accuracy": round(
                    _mean(chat["emotion_accuracy"] for chat in per_chat), 4
                ),
                "sentiment_accuracy": round(
                    _mean(chat["sentiment_accuracy"] for chat in per_chat), 4
                ),
                "emotion_macro_f1": round(
                    emotion_chat_macro["macro_f1"], 4
                ),
                "sentiment_macro_f1": round(
                    sentiment_chat_macro["macro_f1"], 4
                ),
                "topic_consistency": round(
                    _mean(chat["topic_consistency"] for chat in per_chat), 4
                ),
            },
            "micro": {
                "emotion_accuracy": round(emotion_global["accuracy"], 4),
                "sentiment_accuracy": round(sentiment_global["accuracy"], 4),
                "emotion_macro_f1": round(emotion_global["macro_f1"], 4),
                "sentiment_macro_f1": round(sentiment_global["macro_f1"], 4),
                "topic_consistency": round(
                    _mean(score["topic_consistency"] for score in all_scores), 4
                ),
            },
            "classification_details": {
                "emotion": {
                    "global": emotion_global,
                    "chat_macro": emotion_chat_macro,
                },
                "sentiment": {
                    "global": sentiment_global,
                    "chat_macro": sentiment_chat_macro,
                },
            },
            "num_evaluations": len(all_scores),
        }

    entropy_by_chat = {
        chat: round(_mean(
            item["profile"]["explicit_portrait_entropy"] for item in chat_results
        ), 4)
        for chat, chat_results in by_chat.items()
    }
    return {
        "comparison": comparison,
        "improvement_chat_macro": _improvements(comparison),
        "extended_improvement_chat_macro": _extended_improvements(comparison),
        "paired_outcomes": _paired_outcomes(results),
        "metric_protocol": _metric_protocol(),
        "num_eval_points": len(results),
        "num_chats": len(by_chat),
        "portrait_entropy": {
            "by_chat": entropy_by_chat,
            "macro_average": round(_mean(entropy_by_chat.values()), 4),
        },
        "persona_consistency_diagnostic": _persona_consistency(results),
    }


def _improvements(comparison: Dict[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for baseline in ("self_model", "flat_profile"):
        for metric in PRIMARY_METRICS + SUPPLEMENTARY_METRICS:
            values[f"explicit_vs_{baseline}_{metric}"] = round(
                comparison["explicit_model"]["chat_macro"][metric]
                - comparison[baseline]["chat_macro"][metric],
                4,
            )
    return values


def _extended_improvements(comparison: Dict[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for baseline in ("self_model", "flat_profile"):
        for metric in EXTENDED_METRICS:
            values[f"explicit_vs_{baseline}_{metric}"] = round(
                comparison["explicit_model"]["chat_macro"][metric]
                - comparison[baseline]["chat_macro"][metric],
                4,
            )
    return values


def _paired_outcomes(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        f"explicit_model_vs_{baseline}": {
            field: paired_correctness_counts(
                results, "explicit_model", baseline, field
            )
            for field in ("emotion", "sentiment")
        }
        for baseline in ("self_model", "flat_profile")
    }


def _metric_protocol() -> Dict[str, Any]:
    return {
        "primary_metrics": list(PRIMARY_METRICS),
        "supplementary_metrics": list(SUPPLEMENTARY_METRICS),
        "extended_metrics": list(EXTENDED_METRICS),
        "primary_ranking_aggregation": "chat_macro",
        "macro_f1": (
            "unweighted mean over labels present in reference or prediction; "
            "fixed-label value is also retained in classification_details"
        ),
        "topic_policy": (
            "retained unchanged for exploratory analysis; not used for primary "
            "ranking or paired outcome comparisons"
        ),
        "paired_outcomes": (
            "raw paired correctness contingency counts only; formal statistical "
            "test and confidence interval can be recomputed without API calls"
        ),
    }


def _persona_consistency(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Absolute cross-chat EI differences for speakers represented in Exp1."""
    by_speaker_chat: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        by_speaker_chat[result["user_speaker"]][result["chat_file"]].append(
            result["reference"]
        )
    sentiment_value = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    participants: Dict[str, Any] = {}
    for speaker, chats in by_speaker_chat.items():
        if len(chats) != 2:
            continue
        chat_names = sorted(chats)
        left, right = (chats[name] for name in chat_names)
        left_distribution = _label_distribution(item["emotion"] for item in left)
        right_distribution = _label_distribution(item["emotion"] for item in right)
        labels = set(left_distribution) | set(right_distribution)
        emotion_total_variation = 0.5 * sum(
            abs(left_distribution.get(label, 0.0) - right_distribution.get(label, 0.0))
            for label in labels
        )
        left_sentiment = _mean(sentiment_value[item["sentiment"]] for item in left)
        right_sentiment = _mean(sentiment_value[item["sentiment"]] for item in right)
        participants[speaker] = {
            "chat_files": chat_names,
            "emotion_distribution_absolute_difference": round(
                emotion_total_variation, 4
            ),
            "sentiment_mean_absolute_difference": round(
                abs(left_sentiment - right_sentiment), 4
            ),
        }
    return {
        "scope": "human references for Exp1 target messages only",
        "not_used_for_method_ranking": True,
        "participants": participants,
    }


def _label_distribution(labels: Iterable[str]) -> Dict[str, float]:
    counts: Dict[str, int] = defaultdict(int)
    total = 0
    for label in labels:
        counts[label] += 1
        total += 1
    return {label: count / total for label, count in counts.items()} if total else {}


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
    }


def _run_signature(
    config: Exp1Config,
    llm: LLMClient,
    chat_files: List[Path],
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
    return stable_hash({
        "schema_version": 1,
        "model": getattr(llm, "model", None),
        "enable_thinking": getattr(llm, "enable_thinking", None),
        "reference_evaluator": reference_metadata,
        "config": signature_config,
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


def _parse_context_sessions(value: str) -> Optional[int]:
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("context sessions must be positive or 'all'")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 1: causal current-user-state understanding"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp1_user_understanding")
    parser.add_argument("--min-context-sessions", type=int, default=2)
    parser.add_argument("--context-sessions", type=_parse_context_sessions, default=3)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--profile-max-tokens", type=int, default=8000)
    parser.add_argument("--max-eval-points", type=int, default=15)
    parser.add_argument("--operation-max-attempts", type=int, default=3)
    parser.add_argument("--chats", nargs="*", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    run_exp1(Exp1Config(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        min_context_sessions=args.min_context_sessions,
        context_sessions=args.context_sessions,
        max_context_chars=args.max_context_chars,
        profile_max_tokens=args.profile_max_tokens,
        max_eval_points_per_chat=args.max_eval_points,
        operation_max_attempts=args.operation_max_attempts,
        chat_filter=args.chats,
        continue_on_error=not args.fail_fast,
        fresh=args.fresh,
    ))


if __name__ == "__main__":
    main()

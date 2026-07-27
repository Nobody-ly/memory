"""Experiment 2: causal prediction of the user's next conversational state."""
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

from ..llm_client import LLMClient
from ..prediction import (
    FUTURE_STATE_MAX_TOKENS,
    PREDICTION_SYSTEM_PROMPT,
    FutureStatePredictor,
    compute_prediction_error,
)
from ..prompts.templates_en import (
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
)
from ..utils import load_json
from .exp1_metrics import classification_report, speaker_macro_report
from .exp1_protocol import (
    build_message_level_points,
    build_profile_corpus,
    canonical_speaker,
    merge_consecutive_utterances,
    message_speakers,
    select_realtalk_splits,
    stable_hash,
)
from .persona_simulation import session_keys
from .exp1_schema import (
    EMOTION_LABELS,
    REFERENCE_JUDGMENT_SCHEMA,
    SENTIMENT_LABELS,
    normalize_reference_judgment,
)
from .exp1_user_understanding import REFERENCE_JUDGE_SYSTEM_PROMPT
from .exp2_schema import FUTURE_STATE_RESPONSE_SCHEMA, normalize_future_state
from .experiment_utils import robust_parse_json
from .operation_checkpoint import OperationCheckpoint
from .result_provenance import build_run_manifest


METHODS = (
    "llm_only",
    "dialogue_history",
    "user_profile",
    "full_framework",
)
PROFILE_LAYERS = ("core", "regulation", "cognition", "identity", "behavior")
HIGHER_IS_BETTER = (
    "emotion_accuracy",
    "sentiment_accuracy",
    "reflectiveness_accuracy",
    "grounding_accuracy",
)
LOWER_IS_BETTER = (
    "intimacy_absolute_difference",
    "empathy_absolute_difference",
)
EXTENDED_METRICS = ("topic_consistency",)


@dataclass
class Exp2Config:
    dataset_dir: str = "dataset"
    output_dir: str = "data/exp2_predictive_empathy"
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


def run_exp2(
    config: Exp2Config,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Run the REALTALK-aligned future-state prediction stage of Exp2."""
    splits = select_realtalk_splits(
        config.dataset_dir, config.chat_filter, config.speaker_filter
    )
    if not splits:
        raise ValueError("no REALTALK speaker splits matched the configuration")
    llm = llm or LLMClient()
    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    if config.fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
    signature = _run_signature(config, llm, splits)
    checkpoint = OperationCheckpoint(checkpoint_path, signature)
    predictors = {
        method: FutureStatePredictor(llm, mode=method) for method in METHODS
    }
    started = perf_counter()
    run_failures: List[Dict[str, Any]] = []
    print(
        f"[Exp2] speakers={len(splits)} model={getattr(llm, 'model', None)} "
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
        target_speaker = canonical_speaker(test_chat, split["speaker"])
        if train_speaker.casefold() != target_speaker.casefold():
            raise ValueError(
                f"speaker mismatch across split: {train_speaker!r} "
                f"vs {target_speaker!r}"
            )
        partner_speaker = next(
            speaker for speaker in message_speakers(test_chat)
            if speaker.casefold() != target_speaker.casefold()
        )
        profile_corpus = build_profile_corpus(
            train_chat, target_speaker, config.profile_sessions
        )
        profile, profile_key = _cached_profile(
            checkpoint, llm, config, split, profile_corpus, target_speaker
        )
        points = build_message_level_points(
            test_chat,
            target_speaker,
            test_sessions=config.test_sessions,
            max_context_chars=config.max_context_chars,
            max_eval_points=config.max_eval_points_per_speaker,
        )
        print(
            f"[Exp2] speaker={target_speaker} train={train_file.name} "
            f"test={test_file.name} points={len(points)}"
        )

        for index, point in enumerate(points, start=1):
            result_id = (
                f"{_speaker_id(target_speaker)}:{test_file.stem}:"
                f"{point['sample_id']}"
            )
            if result_id in checkpoint.data["results"]:
                print(f"  [{index}/{len(points)}] resume {point['eval_id']}")
                continue
            print(f"  [{index}/{len(points)}] evaluate {point['eval_id']}")
            sample_started = perf_counter()
            try:
                reference = _cached_reference(
                    checkpoint,
                    llm,
                    config,
                    result_id,
                    target_speaker,
                    point["context_text"],
                    point["target_message"],
                )
                latest_message, current_state = _cached_latest_observed_state(
                    checkpoint,
                    llm,
                    config,
                    result_id,
                    target_speaker,
                    point["context_turns"],
                )
                methods: Dict[str, Any] = {}
                for method in METHODS:
                    prediction = _cached_prediction(
                        checkpoint=checkpoint,
                        predictor=predictors[method],
                        llm=llm,
                        config=config,
                        result_id=result_id,
                        method=method,
                        latest_observed_message=latest_message,
                        context_turns=point["context_turns"],
                        profile=profile if method in {
                            "user_profile", "full_framework"
                        } else None,
                        current_state=(
                            current_state if method == "full_framework" else None
                        ),
                    )
                    methods[method] = {
                        "prediction": prediction,
                        "scores": compute_prediction_error(prediction, reference),
                    }
                response_reference = _next_partner_response(
                    test_chat,
                    point["target"]["turn_id"],
                    partner_speaker,
                    config.test_sessions,
                )
                result = {
                    "result_id": result_id,
                    "speaker": target_speaker,
                    "partner_speaker": partner_speaker,
                    "train_chat_file": train_file.name,
                    "test_chat_file": test_file.name,
                    "chat_file": test_file.name,
                    "eval_id": point["eval_id"],
                    "message_level_index": point["message_level_index"],
                    "target_session": point["target_session"],
                    "target_message": point["target_message"],
                    "target_dia_ids": point["target"].get("dia_ids", []),
                    "ground_truth_response": (
                        response_reference["content"]
                        if response_reference is not None else None
                    ),
                    "ground_truth_response_dia_ids": (
                        response_reference.get("dia_ids", [])
                        if response_reference is not None else []
                    ),
                    "reference": reference,
                    "latest_observed_user_message": latest_message,
                    "causal_current_state": current_state,
                    "methods": methods,
                    "profile": {
                        "source": "fixed_cross_conversation_ca",
                        "history_hash": profile_corpus["history_hash"],
                        "train_sessions": profile_corpus["sessions"],
                        "cache_key": profile_key,
                        "characters": len(
                            json.dumps(profile, ensure_ascii=False, indent=2)
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
                        "target_visible_to_predictors": False,
                        "turns": point["context_turns"],
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
                    "status": "excluded_incomplete_quartet",
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
            "task": (
                "predict the state of the next real merged user message from "
                "strictly prior dialogue history"
            ),
            "split": (
                "speaker-specific Ca profile chat and Cb test chat from "
                "REALTALK Table 8"
            ),
            "profile": (
                "fixed five-layer profile from the first three chronological "
                "Ca sessions; never updated with Cb"
            ),
            "context": (
                "first three Cb sessions with message-level rolling history; "
                "the human target is hidden from all predictors"
            ),
            "reference": (
                "the same configured Kimi model labels the hidden human target "
                "under the strict REALTALK-style schema"
            ),
            "paper_metrics": (
                "categorical attribute accuracy and continuous attribute "
                "absolute difference"
            ),
            "topic": (
                "Exp2-specific extension retained as exploratory output and "
                "excluded from primary ranking"
            ),
        },
    })
    manifest = build_run_manifest(
        {**asdict(config), "fresh": False}, getattr(llm, "model", None)
    )
    manifest.update({
        "run_signature": signature,
        "response_schemas": {
            "future_state": FUTURE_STATE_RESPONSE_SCHEMA,
            "reference_judgment": REFERENCE_JUDGMENT_SCHEMA,
        },
        "future_state_max_tokens": FUTURE_STATE_MAX_TOKENS,
        "metric_protocol": _metric_protocol(),
        "network_retry_max_attempts": getattr(llm, "max_retries", None),
        "operation_retry_max_attempts": config.operation_max_attempts,
        "realtalk_speaker_splits": splits,
        "prompt_hashes": {
            "prediction": stable_hash(PREDICTION_SYSTEM_PROMPT),
            "profile": stable_hash(
                PROFILE_EXTRACTION_SYSTEM_PROMPT
                + PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
            ),
            "reference": stable_hash(REFERENCE_JUDGE_SYSTEM_PROMPT),
        },
        "output_contract": {
            "results.jsonl": "complete per-sample four-method comparisons",
            "metric_records.jsonl": "long-form records for offline analysis",
            "summary.json": "speaker-macro and micro aggregate metrics",
            "checkpoint.json": "resumable operation cache",
            "run_manifest.json": "protocol, model, hashes, and retry metadata",
        },
        "response_generation_readiness": {
            "status": "raw_inputs_preserved",
            "fields": [
                "context.turns",
                "target_message",
                "ground_truth_response",
                "methods.*.prediction",
                "profile cache key",
            ],
            "note": (
                "ROUGE, BERTScore, response EI differences, and EPITOME are "
                "computed in the separate response-generation stage."
            ),
        },
    })
    _write_outputs(output_dir, results, summary, manifest)
    print(f"[Exp2] summary={json.dumps(summary.get('comparison', {}), indent=2)}")
    return summary


def _cached_profile(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2Config,
    split: Dict[str, str],
    profile_corpus: Dict[str, Any],
    speaker: str,
) -> tuple[Dict[str, Any], str]:
    prompt_hash = stable_hash(
        PROFILE_EXTRACTION_SYSTEM_PROMPT
        + PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE
    )
    key = ":".join((
        "profile",
        _speaker_id(speaker),
        Path(split["train_chat"]).stem,
        f"sessions_{config.profile_sessions}",
        "explicit",
        profile_corpus["history_hash"],
        str(getattr(llm, "model", "unknown")),
        prompt_hash,
    ))

    def operation() -> Dict[str, Any]:
        return robust_parse_json(llm.chat(
            PROFILE_EXTRACTION_SYSTEM_PROMPT.format(user_name=speaker),
            PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                user_name=speaker, corpus=profile_corpus["text"]
            ),
            temperature=0.3,
            max_tokens=config.profile_max_tokens,
        ))

    profile = checkpoint.execute(
        key,
        operation,
        _validate_profile,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )
    return profile, key


def _cached_reference(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2Config,
    result_id: str,
    speaker: str,
    history_text: str,
    target_message: str,
) -> Dict[str, Any]:
    input_hash = stable_hash({
        "speaker": speaker,
        "history": history_text,
        "target": target_message,
        "system": REFERENCE_JUDGE_SYSTEM_PROMPT,
        "schema": REFERENCE_JUDGMENT_SCHEMA,
    })
    key = f"reference_judgment:{result_id}:{input_hash}"
    prompt = (
        f"CONVERSATION HISTORY:\n{history_text or '(none)'}\n\n"
        f"CURRENT OBSERVED MESSAGE BY {speaker}:\n{target_message}\n\n"
        "Judge only the current observed message, using its prior context."
    )
    return checkpoint.execute(
        key,
        lambda: _reference_call(llm, prompt),
        normalize_reference_judgment,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _cached_latest_observed_state(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2Config,
    result_id: str,
    speaker: str,
    context_turns: List[Dict[str, Any]],
) -> tuple[str, Dict[str, Any]]:
    latest_index = next(
        (
            index for index in range(len(context_turns) - 1, -1, -1)
            if context_turns[index]["speaker"].casefold() == speaker.casefold()
        ),
        None,
    )
    if latest_index is None:
        return "(no prior user message)", {}
    latest = context_turns[latest_index]
    prior = context_turns[:latest_index]
    history_text = "\n".join(
        f"{turn['speaker']}: {turn['content']}" for turn in prior
    )
    input_hash = stable_hash({
        "speaker": speaker,
        "history": history_text,
        "observed": latest["content"],
        "schema": REFERENCE_JUDGMENT_SCHEMA,
    })
    key = f"causal_state:{result_id}:{input_hash}"
    prompt = (
        f"CONVERSATION HISTORY:\n{history_text or '(none)'}\n\n"
        f"CURRENT OBSERVED MESSAGE BY {speaker}:\n{latest['content']}\n\n"
        "Judge only the current observed message, using its prior context."
    )
    state = checkpoint.execute(
        key,
        lambda: _reference_call(llm, prompt),
        normalize_reference_judgment,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )
    return latest["content"], state


def _cached_prediction(
    checkpoint: OperationCheckpoint,
    predictor: FutureStatePredictor,
    llm: LLMClient,
    config: Exp2Config,
    result_id: str,
    method: str,
    latest_observed_message: str,
    context_turns: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]],
    current_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    method_history = [] if method == "llm_only" else context_turns
    input_hash = stable_hash({
        "method": method,
        "latest_observed_message": latest_observed_message,
        "history": method_history,
        "profile": profile,
        "current_state": current_state,
        "system": PREDICTION_SYSTEM_PROMPT,
        "schema": FUTURE_STATE_RESPONSE_SCHEMA,
    })
    key = f"prediction:{result_id}:{method}:{input_hash}"
    return checkpoint.execute(
        key,
        lambda: predictor.predict(
            user_message=latest_observed_message,
            conversation_history=method_history,
            user_profile=profile,
            current_state=current_state,
        ),
        normalize_future_state,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _reference_call(llm: LLMClient, prompt: str) -> Dict[str, Any]:
    raw = llm.chat(
        REFERENCE_JUDGE_SYSTEM_PROMPT,
        prompt,
        temperature=0.0,
        max_tokens=1024,
        response_schema=REFERENCE_JUDGMENT_SCHEMA,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("strict reference judgment was not valid JSON") from exc


def _validate_profile(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("error"):
        raise ValueError("explicit profile is invalid")
    missing = [
        layer for layer in PROFILE_LAYERS
        if not isinstance(value.get(layer), dict)
    ]
    if missing:
        raise ValueError(f"explicit profile is missing layers: {missing}")
    return value


def _next_partner_response(
    chat: Dict[str, Any],
    target_turn_id: str,
    partner_speaker: str,
    test_sessions: int,
) -> Optional[Dict[str, Any]]:
    selected = set(session_keys(chat)[:test_sessions])
    turns = [
        turn for turn in merge_consecutive_utterances(chat)
        if turn["session_id"] in selected
    ]
    target_index = next(
        (
            index for index, turn in enumerate(turns)
            if turn["turn_id"] == target_turn_id
        ),
        None,
    )
    if target_index is None or target_index + 1 >= len(turns):
        return None
    response = turns[target_index + 1]
    if response["speaker"].casefold() != partner_speaker.casefold():
        return None
    return response


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "comparison": {},
            "num_eval_points": 0,
            "num_speakers": 0,
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
                "reference": item["reference"]["emotion"],
                "prediction": item["methods"][method]["prediction"]["emotion"],
            } for item in speaker_results)
            sentiment_records.extend({
                "speaker": speaker,
                "reference": item["reference"]["sentiment"],
                "prediction": item["methods"][method]["prediction"]["sentiment"],
            } for item in speaker_results)
            per_speaker.append({
                metric: _mean(score[metric] for score in scores)
                for metric in (
                    *HIGHER_IS_BETTER,
                    *LOWER_IS_BETTER,
                    *EXTENDED_METRICS,
                )
            })
        emotion_global = classification_report(
            [item["reference"] for item in emotion_records],
            [item["prediction"] for item in emotion_records],
            EMOTION_LABELS,
        )
        sentiment_global = classification_report(
            [item["reference"] for item in sentiment_records],
            [item["prediction"] for item in sentiment_records],
            SENTIMENT_LABELS,
        )
        emotion_speaker = speaker_macro_report(emotion_records, EMOTION_LABELS)
        sentiment_speaker = speaker_macro_report(
            sentiment_records, SENTIMENT_LABELS
        )
        speaker_macro = {
            metric: round(_mean(item[metric] for item in per_speaker), 4)
            for metric in (
                *HIGHER_IS_BETTER,
                *LOWER_IS_BETTER,
                *EXTENDED_METRICS,
            )
        }
        speaker_macro.update({
            "emotion_macro_f1": round(emotion_speaker["macro_f1"], 4),
            "sentiment_macro_f1": round(sentiment_speaker["macro_f1"], 4),
        })
        micro = {
            metric: round(_mean(score[metric] for score in all_scores), 4)
            for metric in (
                *HIGHER_IS_BETTER,
                *LOWER_IS_BETTER,
                *EXTENDED_METRICS,
            )
        }
        micro.update({
            "emotion_macro_f1": round(emotion_global["macro_f1"], 4),
            "sentiment_macro_f1": round(sentiment_global["macro_f1"], 4),
        })
        comparison[method] = {
            "speaker_macro": speaker_macro,
            "micro": micro,
            "classification_details": {
                "emotion": {
                    "global": emotion_global,
                    "speaker_macro": emotion_speaker,
                },
                "sentiment": {
                    "global": sentiment_global,
                    "speaker_macro": sentiment_speaker,
                },
            },
            "num_evaluations": len(all_scores),
        }
    return {
        "comparison": comparison,
        "full_framework_improvement": _improvements(comparison),
        "num_eval_points": len(results),
        "num_speakers": len(by_speaker),
        "metric_protocol": _metric_protocol(),
    }


def _improvements(comparison: Dict[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    full = comparison["full_framework"]["speaker_macro"]
    for baseline in METHODS[:-1]:
        other = comparison[baseline]["speaker_macro"]
        for metric in (
            *HIGHER_IS_BETTER,
            "emotion_macro_f1",
            "sentiment_macro_f1",
        ):
            values[f"full_vs_{baseline}_{metric}"] = round(
                full[metric] - other[metric], 4
            )
        for metric in LOWER_IS_BETTER:
            values[f"full_vs_{baseline}_{metric}_reduction"] = round(
                other[metric] - full[metric], 4
            )
    return values


def _metric_protocol() -> Dict[str, Any]:
    return {
        "primary_metrics": [
            "emotion_accuracy",
            "sentiment_accuracy",
            "emotion_macro_f1",
            "sentiment_macro_f1",
            "intimacy_absolute_difference",
        ],
        "paper_aligned_auxiliary_metrics": [
            "reflectiveness_accuracy",
            "grounding_accuracy",
            "empathy_absolute_difference",
        ],
        "extended_metrics": list(EXTENDED_METRICS),
        "primary_ranking_aggregation": "speaker_macro",
        "categorical_policy": "strict label equality; no semantic-match credit",
        "continuous_policy": "mean absolute difference",
        "topic_policy": (
            "lexical overlap retained for exploratory continuity only; excluded "
            "from primary rankings"
        ),
        "content_similarity": (
            "ROUGE and BERTScore apply to the response-generation stage, not "
            "structured future-state prediction"
        ),
        "composite_score": "none; every metric is reported separately",
    }


def _build_metric_records(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for result in results:
        for method in METHODS:
            records.append({
                "result_id": result["result_id"],
                "speaker": result["speaker"],
                "chat_file": result["chat_file"],
                "eval_id": result["eval_id"],
                "method": method,
                "reference": result["reference"],
                "prediction": result["methods"][method]["prediction"],
                "scores": result["methods"][method]["scores"],
            })
    return records


def _run_signature(
    config: Exp2Config,
    llm: LLMClient,
    splits: List[Dict[str, str]],
) -> str:
    signature_config = asdict(config)
    for key in (
        "output_dir",
        "continue_on_error",
        "fresh",
        "max_eval_points_per_speaker",
    ):
        signature_config.pop(key, None)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("exp1_protocol.py"),
        Path(__file__).with_name("exp2_schema.py"),
        Path(__file__).with_name("operation_checkpoint.py"),
        Path(__file__).parents[1] / "prediction.py",
    ]
    dataset_dir = Path(config.dataset_dir)
    chat_files = sorted({
        dataset_dir / split[key]
        for split in splits
        for key in ("train_chat", "test_chat")
    })
    return stable_hash({
        "schema_version": 1,
        "model": getattr(llm, "model", None),
        "enable_thinking": getattr(llm, "enable_thinking", None),
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


def _checkpoint_token_usage(checkpoint: OperationCheckpoint) -> Dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    for operation in checkpoint.data["operations"].values():
        for key in total:
            total[key] += int(operation.get("token_usage", {}).get(key, 0))
    return total


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
    _atomic_text(
        output_dir / "metric_records.jsonl",
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in _build_metric_records(results)
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


def _speaker_id(speaker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", speaker.casefold()).strip("_")


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: causal future-state prediction"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/exp2_predictive_empathy")
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
    run_exp2(Exp2Config(
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

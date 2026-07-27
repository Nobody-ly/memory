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
    BACKGROUND_REASONING_SYSTEM_PROMPT,
    BACKGROUND_REASONING_USER_PROMPT_TEMPLATE,
    DDIRECT_RESPONSE_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT,
    EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE,
    PERSONA_EXTRACTION_SYSTEM_PROMPT,
    PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE,
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    PROFILE_EXTRACTION_USER_PROMPT_TEMPLATE,
)
from ..utils import load_json
from .exp1_metrics import classification_report, speaker_macro_report
from .exp1_protocol import (
    REALTALK_PERSONA_SPLITS,
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
from .empathy_alignment_analysis import EmpathyAlignmentReasoner
from .exp2_framework import (
    FRAMEWORK_STATE_MAX_TOKENS,
    FrameworkStateReasoner,
    latest_complete_exchange,
)
from .exp2_generation import (
    BERTSCORE_MODEL,
    BERTSCORE_NUM_LAYERS,
    RESPONSE_MAX_TOKENS,
    Exp2ResponseGenerator,
    add_batched_bertscore,
    compute_response_scores,
)
from .exp2_schema import (
    FRAMEWORK_STATE_RESPONSE_SCHEMA,
    FUTURE_STATE_RESPONSE_SCHEMA,
    normalize_framework_state,
    normalize_future_state,
)
from .experiment_utils import robust_parse_json
from .operation_checkpoint import OperationCheckpoint
from .realtalk_evaluator import RealTalkLabelEvaluator
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
)
LOWER_IS_BETTER = (
    "intimacy_absolute_difference",
)
EXTENDED_METRICS = ("topic_consistency",)
RESPONSE_HIGHER_IS_BETTER = (
    "style_similarity",
    "rouge_l",
    "lexical_overlap",
    "reflectiveness_accuracy",
    "grounding_accuracy",
    "sentiment_accuracy",
    "emotion_accuracy",
)
RESPONSE_LOWER_IS_BETTER = (
    "intimacy_absolute_difference",
    "empathy_absolute_difference",
)


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
    run_generation: bool = True
    compute_bertscore: bool = False
    fresh: bool = False


def run_exp2(
    config: Exp2Config,
    llm: Optional[LLMClient] = None,
    label_evaluator: Optional[RealTalkLabelEvaluator] = None,
) -> Dict[str, Any]:
    """Run causal prediction and response generation for Experiment 2."""
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
    signature = _run_signature(config, llm, splits)
    checkpoint = OperationCheckpoint(checkpoint_path, signature)
    predictors = {
        method: FutureStatePredictor(llm, mode=method) for method in METHODS
    }
    response_generators = {
        method: Exp2ResponseGenerator(llm, method=method) for method in METHODS
    }
    state_reasoner = FrameworkStateReasoner(llm)
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
        persona_corpus, persona_source = _partner_persona_corpus(
            dataset_dir,
            partner_speaker,
            config.profile_sessions,
        )
        persona, persona_key = _cached_persona(
            checkpoint,
            llm,
            config,
            partner_speaker,
            persona_corpus,
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

        rolling_framework_state: Dict[str, Any] = {}
        for index, point in enumerate(points, start=1):
            result_id = (
                f"{_speaker_id(target_speaker)}:{test_file.stem}:"
                f"{point['sample_id']}"
            )
            if result_id in checkpoint.data["results"]:
                saved = checkpoint.data["results"][result_id]
                rolling_framework_state = (
                    saved.get("framework_state") or rolling_framework_state
                )
                print(f"  [{index}/{len(points)}] resume {point['eval_id']}")
                continue
            print(f"  [{index}/{len(points)}] evaluate {point['eval_id']}")
            sample_started = perf_counter()
            try:
                recent_exchange, observed_user_input, observed_partner_reply = (
                    latest_complete_exchange(
                        point["context_turns"],
                        target_speaker,
                        partner_speaker,
                    )
                )
                framework_state = _cached_framework_state(
                    checkpoint,
                    state_reasoner,
                    llm,
                    config,
                    result_id,
                    recent_exchange,
                    (
                        point["context_turns"][:-len(recent_exchange)]
                        if recent_exchange else point["context_turns"]
                    ),
                    observed_user_input,
                    observed_partner_reply,
                    profile,
                    persona,
                    rolling_framework_state,
                )
                reference = _cached_message_ei(
                    checkpoint,
                    llm,
                    label_evaluator,
                    config,
                    result_id,
                    "future_user_message",
                    target_speaker,
                    point["context_text"],
                    point["target_message"],
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
                        recent_exchange=recent_exchange,
                        context_turns=point["context_turns"],
                        profile=profile if method in {
                            "user_profile", "full_framework"
                        } else None,
                        framework_state=(
                            framework_state
                            if method == "full_framework" else None
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
                empathy_alignment: Dict[str, Any] = {}
                response_reference_ei: Optional[Dict[str, Any]] = None
                if config.run_generation and response_reference is not None:
                    empathy_alignment = _cached_empathy_alignment(
                        checkpoint,
                        llm,
                        config,
                        result_id,
                        point,
                        profile,
                        persona,
                        framework_state,
                    )
                    response_reference_ei = _cached_message_ei(
                        checkpoint,
                        llm,
                        label_evaluator,
                        config,
                        result_id,
                        "ground_truth_partner_response",
                        partner_speaker,
                        (
                            point["context_text"]
                            + "\n"
                            + f"{target_speaker}: {point['target_message']}"
                        ),
                        response_reference["content"],
                    )
                    for method in METHODS:
                        guidance = None
                        if method == "full_framework":
                            guidance = {
                                "state": framework_state,
                                "empathy_alignment": empathy_alignment,
                            }
                        generated = _cached_generated_response(
                            checkpoint,
                            response_generators[method],
                            llm,
                            config,
                            result_id,
                            method,
                            point,
                            target_speaker,
                            partner_speaker,
                            profile if method in {
                                "user_profile", "full_framework"
                            } else None,
                            persona,
                            guidance,
                        )
                        generated_ei = _cached_message_ei(
                            checkpoint,
                            llm,
                            label_evaluator,
                            config,
                            result_id,
                            f"generated_{method}",
                            partner_speaker,
                            (
                                point["context_text"]
                                + "\n"
                                + f"{target_speaker}: {point['target_message']}"
                            ),
                            generated,
                        )
                        methods[method]["generation"] = {
                            "response": generated,
                            "response_ei": generated_ei,
                            "scores": compute_response_scores(
                                reference_text=response_reference["content"],
                                candidate_text=generated,
                                reference_ei=response_reference_ei,
                                candidate_ei=generated_ei,
                            ),
                        }
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
                    "ground_truth_response_ei": response_reference_ei,
                    "reference": reference,
                    "recent_complete_exchange": recent_exchange,
                    "framework_state": framework_state,
                    "empathy_alignment": empathy_alignment,
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
                    "agent_persona": {
                        "speaker": partner_speaker,
                        "source": persona_source,
                        "history_hash": persona_corpus["history_hash"],
                        "train_sessions": persona_corpus["sessions"],
                        "cache_key": persona_key,
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
                rolling_framework_state = framework_state or rolling_framework_state
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

    results = sorted(checkpoint.result_values(), key=_result_sort_key)
    if config.compute_bertscore:
        add_batched_bertscore(results)
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
                "pinned REALTALK classifiers label emotion, sentiment, and "
                "intimacy; Kimi supplies only non-classifier attributes"
            ),
            "generation": (
                "all four methods answer the same observed target message; "
                "the human partner's next merged message is the reference"
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
            "framework_state": FRAMEWORK_STATE_RESPONSE_SCHEMA,
            "reference_judgment": REFERENCE_JUDGMENT_SCHEMA,
        },
        "future_state_max_tokens": FUTURE_STATE_MAX_TOKENS,
        "framework_state_max_tokens": FRAMEWORK_STATE_MAX_TOKENS,
        "response_max_tokens": RESPONSE_MAX_TOKENS,
        "label_evaluator": label_evaluator.metadata(),
        "bertscore": {
            "model_type": BERTSCORE_MODEL,
            "num_layers": BERTSCORE_NUM_LAYERS,
            "idf": False,
            "rescale_with_baseline": False,
        },
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
            "persona": stable_hash(
                PERSONA_EXTRACTION_SYSTEM_PROMPT
                + PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE
            ),
            "framework_state": stable_hash(
                BACKGROUND_REASONING_SYSTEM_PROMPT
                + BACKGROUND_REASONING_USER_PROMPT_TEMPLATE
            ),
            "empathy_alignment": stable_hash(
                EMPATHY_ALIGNMENT_REASONING_SYSTEM_PROMPT
                + EMPATHY_ALIGNMENT_REASONING_USER_PROMPT_TEMPLATE
            ),
            "response_generation": stable_hash(
                DDIRECT_RESPONSE_SYSTEM_PROMPT
            ),
        },
        "output_contract": {
            "results.jsonl": "complete per-sample four-method comparisons",
            "metric_records.jsonl": "long-form records for offline analysis",
            "summary.json": "speaker-macro and micro aggregate metrics",
            "checkpoint.json": "resumable operation cache",
            "run_manifest.json": "protocol, model, hashes, and retry metadata",
        },
        "response_generation": {
            "enabled": config.run_generation,
            "bertscore_enabled": config.compute_bertscore,
            "fields": [
                "context.turns",
                "target_message",
                "ground_truth_response",
                "methods.*.prediction",
                "methods.*.generation.response",
                "methods.*.generation.response_ei",
                "methods.*.generation.scores",
            ],
            "note": (
                "Raw response pairs and EI labels are retained even when "
                "BERTScore is deferred."
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


def _partner_persona_corpus(
    dataset_dir: Path,
    partner_speaker: str,
    profile_sessions: int,
) -> tuple[Dict[str, Any], str]:
    split = next(
        (
            item for item in REALTALK_PERSONA_SPLITS
            if item["speaker"].casefold() == partner_speaker.casefold()
        ),
        None,
    )
    if split is None:
        raise ValueError(
            f"no REALTALK persona split is defined for {partner_speaker}"
        )
    train_file = dataset_dir / split["train_chat"]
    if not train_file.exists():
        raise FileNotFoundError(
            f"agent persona corpus is missing: {train_file.name}"
        )
    train_chat = load_json(str(train_file))
    speaker = canonical_speaker(train_chat, partner_speaker)
    return (
        build_profile_corpus(train_chat, speaker, profile_sessions),
        f"fixed_cross_conversation_ca:{train_file.name}",
    )


def _cached_persona(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2Config,
    speaker: str,
    corpus: Dict[str, Any],
) -> tuple[Dict[str, Any], str]:
    system_prompt = PERSONA_EXTRACTION_SYSTEM_PROMPT.format(agent_name=speaker)
    user_prompt = PERSONA_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        agent_name=speaker,
        corpus=corpus["text"],
    )
    prompt_hash = stable_hash(system_prompt + user_prompt)
    key = ":".join((
        "persona",
        _speaker_id(speaker),
        f"sessions_{config.profile_sessions}",
        corpus["history_hash"],
        str(getattr(llm, "model", "unknown")),
        prompt_hash,
    ))
    persona = checkpoint.execute(
        key,
        lambda: robust_parse_json(llm.chat(
            system_prompt,
            user_prompt,
            temperature=0.3,
            max_tokens=2048,
        )),
        lambda value: _validate_persona(value, speaker),
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )
    return persona, key


def _cached_message_ei(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    label_evaluator: RealTalkLabelEvaluator,
    config: Exp2Config,
    result_id: str,
    role: str,
    speaker: str,
    history_text: str,
    message: str,
) -> Dict[str, Any]:
    input_hash = stable_hash({
        "role": role,
        "speaker": speaker,
        "history": history_text,
        "message": message,
        "system": REFERENCE_JUDGE_SYSTEM_PROMPT,
        "schema": REFERENCE_JUDGMENT_SCHEMA,
        "classifier": label_evaluator.metadata(),
    })
    key = f"message_ei:{result_id}:{role}:{input_hash}"
    prompt = (
        f"CONVERSATION HISTORY:\n{history_text or '(none)'}\n\n"
        f"CURRENT OBSERVED MESSAGE BY {speaker}:\n{message}\n\n"
        "Judge only the current observed message, using its prior context."
    )

    def operation() -> Dict[str, Any]:
        judgment = _reference_call(llm, prompt)
        fixed_labels = label_evaluator.annotate(message)
        return {
            **judgment,
            **fixed_labels,
        }

    return checkpoint.execute(
        key,
        operation,
        normalize_reference_judgment,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _cached_framework_state(
    checkpoint: OperationCheckpoint,
    reasoner: FrameworkStateReasoner,
    llm: LLMClient,
    config: Exp2Config,
    result_id: str,
    recent_exchange: List[Dict[str, Any]],
    prior_context: List[Dict[str, Any]],
    user_input: str,
    assistant_response: str,
    profile: Dict[str, Any],
    persona: Dict[str, Any],
    previous_state: Dict[str, Any],
) -> Dict[str, Any]:
    if not recent_exchange or not user_input or not assistant_response:
        return previous_state
    input_hash = stable_hash({
        "exchange": recent_exchange,
        "prior_context": prior_context,
        "profile": profile,
        "persona": persona,
        "previous_state": previous_state,
        "schema": FRAMEWORK_STATE_RESPONSE_SCHEMA,
    })
    key = f"framework_state:{result_id}:{input_hash}"
    return checkpoint.execute(
        key,
        lambda: reasoner.derive(
            user_input=user_input,
            assistant_response=assistant_response,
            static_profile=profile,
            previous_state=previous_state,
            previous_context=prior_context[-10:],
            agent_persona=persona,
        ),
        normalize_framework_state,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _cached_prediction(
    checkpoint: OperationCheckpoint,
    predictor: FutureStatePredictor,
    llm: LLMClient,
    config: Exp2Config,
    result_id: str,
    method: str,
    recent_exchange: List[Dict[str, Any]],
    context_turns: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]],
    framework_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    method_history = [] if method == "llm_only" else context_turns
    method_exchange = recent_exchange if method == "llm_only" else []
    effective_state = framework_state or None
    effective_method = method
    if method == "dialogue_history" and not method_history:
        effective_method = "llm_only"
    elif method == "full_framework" and not effective_state:
        effective_method = "user_profile"
    input_hash = stable_hash({
        "effective_method": effective_method,
        "recent_exchange": method_exchange,
        "history": method_history,
        "profile": profile,
        "framework_state": effective_state,
        "system": PREDICTION_SYSTEM_PROMPT,
        "schema": FUTURE_STATE_RESPONSE_SCHEMA,
    })
    key = f"prediction:{result_id}:{effective_method}:{input_hash}"
    return checkpoint.execute(
        key,
        lambda: predictor.predict(
            recent_exchange=method_exchange,
            conversation_history=method_history,
            user_profile=profile,
            current_state=effective_state,
        ),
        normalize_future_state,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _cached_empathy_alignment(
    checkpoint: OperationCheckpoint,
    llm: LLMClient,
    config: Exp2Config,
    result_id: str,
    point: Dict[str, Any],
    profile: Dict[str, Any],
    persona: Dict[str, Any],
    framework_state: Dict[str, Any],
) -> Dict[str, Any]:
    input_hash = stable_hash({
        "target_message": point["target_message"],
        "context": point["context_turns"],
        "profile": profile,
        "persona": persona,
        "framework_state": framework_state,
        "interaction_count": point["message_level_index"],
    })
    key = f"empathy_alignment:{result_id}:{input_hash}"

    def operation() -> Dict[str, Any]:
        reasoner = EmpathyAlignmentReasoner(
            llm,
            interaction_count=point["message_level_index"],
        )
        return reasoner.reason(
            point["target_message"],
            point["context_turns"],
            profile,
            persona,
            framework_state.get("current_state", {}),
        )

    return checkpoint.execute(
        key,
        operation,
        _validate_empathy_alignment,
        max_attempts=config.operation_max_attempts,
        usage_supplier=lambda: dict(getattr(llm, "token_usage", {})),
    )


def _cached_generated_response(
    checkpoint: OperationCheckpoint,
    generator: Exp2ResponseGenerator,
    llm: LLMClient,
    config: Exp2Config,
    result_id: str,
    method: str,
    point: Dict[str, Any],
    user_speaker: str,
    agent_speaker: str,
    profile: Optional[Dict[str, Any]],
    persona: Dict[str, Any],
    framework_guidance: Optional[Dict[str, Any]],
) -> str:
    input_hash = stable_hash({
        "method": method,
        "target_message": point["target_message"],
        "context": point["context_turns"],
        "profile": profile,
        "persona": persona,
        "framework_guidance": framework_guidance,
    })
    key = f"generated_response:{result_id}:{method}:{input_hash}"
    return checkpoint.execute(
        key,
        lambda: generator.generate(
            user_message=point["target_message"],
            context_turns=point["context_turns"],
            profile=profile,
            persona=persona,
            framework_guidance=framework_guidance,
            agent_speaker=agent_speaker,
            user_speaker=user_speaker,
        ),
        _validate_generated_response,
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


def _validate_persona(value: Any, speaker: str) -> Dict[str, Any]:
    required = (
        "name",
        "personality",
        "tone",
        "interaction_principles",
        "expression_patterns",
    )
    if not isinstance(value, dict) or not all(key in value for key in required):
        raise ValueError(f"persona for {speaker} is missing required fields")
    if not str(value.get("personality", "")).strip():
        raise ValueError(f"persona for {speaker} has no personality")
    return {key: value[key] for key in required}


def _validate_empathy_alignment(value: Any) -> Dict[str, Any]:
    required = {
        "understanding",
        "prediction",
        "exploration",
        "alignment",
        "empathy_state",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("empathy alignment is incomplete")
    if not isinstance(value["empathy_state"], dict):
        raise ValueError("empathy alignment has no empathy_state")
    return value


def _validate_generated_response(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("generated response is empty")
    return value.strip()


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
        prediction = _aggregate_prediction(by_speaker, method)
        generation = _aggregate_generation(by_speaker, method)
        comparison[method] = {
            "prediction": prediction,
            "generation": generation,
        }
    return {
        "comparison": comparison,
        "full_framework_improvement": _improvements(comparison),
        "prediction_trend": _prediction_trend(results),
        "num_eval_points": len(results),
        "num_speakers": len(by_speaker),
        "metric_protocol": _metric_protocol(),
    }


def _aggregate_prediction(
    by_speaker: Dict[str, List[Dict[str, Any]]],
    method: str,
) -> Dict[str, Any]:
    metrics = (*HIGHER_IS_BETTER, *LOWER_IS_BETTER, *EXTENDED_METRICS)
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
            for metric in metrics
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
    sentiment_speaker = speaker_macro_report(sentiment_records, SENTIMENT_LABELS)
    speaker_macro = {
        metric: round(_mean(item[metric] for item in per_speaker), 4)
        for metric in metrics
    }
    speaker_macro.update({
        "emotion_macro_f1": round(emotion_speaker["macro_f1"], 4),
        "sentiment_macro_f1": round(sentiment_speaker["macro_f1"], 4),
    })
    micro = {
        metric: round(_mean(score[metric] for score in all_scores), 4)
        for metric in metrics
    }
    micro.update({
        "emotion_macro_f1": round(emotion_global["macro_f1"], 4),
        "sentiment_macro_f1": round(sentiment_global["macro_f1"], 4),
    })
    return {
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


def _aggregate_generation(
    by_speaker: Dict[str, List[Dict[str, Any]]],
    method: str,
) -> Dict[str, Any]:
    per_speaker: List[Dict[str, float]] = []
    all_scores: List[Dict[str, float]] = []
    metric_names = [
        *RESPONSE_HIGHER_IS_BETTER,
        *RESPONSE_LOWER_IS_BETTER,
    ]
    has_bertscore = any(
        "bertscore_f1" in (
            item["methods"][method].get("generation", {}).get("scores", {})
        )
        for items in by_speaker.values()
        for item in items
    )
    if has_bertscore:
        metric_names.append("bertscore_f1")
    for speaker_results in by_speaker.values():
        scores = [
            item["methods"][method]["generation"]["scores"]
            for item in speaker_results
            if item["methods"][method].get("generation")
        ]
        if not scores:
            continue
        all_scores.extend(scores)
        per_speaker.append({
            metric: _mean(score[metric] for score in scores if metric in score)
            for metric in metric_names
        })
    return {
        "speaker_macro": {
            metric: round(_mean(item[metric] for item in per_speaker), 4)
            for metric in metric_names
        },
        "micro": {
            metric: round(
                _mean(score[metric] for score in all_scores if metric in score),
                4,
            )
            for metric in metric_names
        },
        "num_evaluations": len(all_scores),
    }


def _prediction_trend(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    trend: Dict[str, Any] = {}
    for method in METHODS:
        by_index: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        for item in results:
            by_index[int(item["message_level_index"])].append(
                item["methods"][method]["scores"]
            )
        trend[method] = [
            {
                "message_level_index": index,
                "num_samples": len(scores),
                "emotion_accuracy": round(
                    _mean(score["emotion_accuracy"] for score in scores), 4
                ),
                "sentiment_accuracy": round(
                    _mean(score["sentiment_accuracy"] for score in scores), 4
                ),
                "intimacy_absolute_difference": round(
                    _mean(
                        score["intimacy_absolute_difference"]
                        for score in scores
                    ),
                    4,
                ),
            }
            for index, scores in sorted(by_index.items())
        ]
    return trend


def _improvements(comparison: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {"prediction": {}, "generation": {}}
    full_prediction = comparison["full_framework"]["prediction"]["speaker_macro"]
    full_generation = comparison["full_framework"]["generation"]["speaker_macro"]
    for baseline in METHODS[:-1]:
        other_prediction = comparison[baseline]["prediction"]["speaker_macro"]
        for metric in (
            *HIGHER_IS_BETTER,
            "emotion_macro_f1",
            "sentiment_macro_f1",
        ):
            values["prediction"][f"full_vs_{baseline}_{metric}"] = round(
                full_prediction[metric] - other_prediction[metric], 4
            )
        for metric in LOWER_IS_BETTER:
            values["prediction"][
                f"full_vs_{baseline}_{metric}_reduction"
            ] = round(other_prediction[metric] - full_prediction[metric], 4)

        other_generation = comparison[baseline]["generation"]["speaker_macro"]
        for metric in RESPONSE_HIGHER_IS_BETTER:
            if metric in full_generation and metric in other_generation:
                values["generation"][f"full_vs_{baseline}_{metric}"] = round(
                    full_generation[metric] - other_generation[metric], 4
                )
        if "bertscore_f1" in full_generation and "bertscore_f1" in other_generation:
            values["generation"][
                f"full_vs_{baseline}_bertscore_f1"
            ] = round(
                full_generation["bertscore_f1"]
                - other_generation["bertscore_f1"],
                4,
            )
        for metric in RESPONSE_LOWER_IS_BETTER:
            if metric in full_generation and metric in other_generation:
                values["generation"][
                    f"full_vs_{baseline}_{metric}_reduction"
                ] = round(other_generation[metric] - full_generation[metric], 4)
    return values


def _metric_protocol() -> Dict[str, Any]:
    return {
        "prediction_primary_metrics": [
            "emotion_accuracy",
            "sentiment_accuracy",
            "emotion_macro_f1",
            "sentiment_macro_f1",
            "intimacy_absolute_difference",
        ],
        "prediction_extended_metrics": list(EXTENDED_METRICS),
        "generation_paper_aligned_metrics": [
            "style_similarity",
            "rouge_l",
            "bertscore_f1",
            "reflectiveness_accuracy",
            "grounding_accuracy",
            "sentiment_accuracy",
            "emotion_accuracy",
            "intimacy_absolute_difference",
            "empathy_absolute_difference",
        ],
        "primary_ranking_aggregation": "speaker_macro",
        "categorical_policy": "strict label equality; no semantic-match credit",
        "continuous_policy": "mean absolute difference",
        "topic_policy": (
            "lexical overlap retained for exploratory continuity only; excluded "
            "from primary rankings"
        ),
        "bertscore_policy": (
            "computed in one optional batch after generation; raw response "
            "pairs are always preserved for deterministic recomputation"
        ),
        "composite_score": "none; every metric is reported separately",
    }


def _build_metric_records(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for result in results:
        for method in METHODS:
            records.append({
                "stage": "prediction",
                "result_id": result["result_id"],
                "speaker": result["speaker"],
                "chat_file": result["chat_file"],
                "eval_id": result["eval_id"],
                "method": method,
                "reference": result["reference"],
                "prediction": result["methods"][method]["prediction"],
                "scores": result["methods"][method]["scores"],
            })
            generation = result["methods"][method].get("generation")
            if generation:
                records.append({
                    "stage": "generation",
                    "result_id": result["result_id"],
                    "speaker": result["speaker"],
                    "chat_file": result["chat_file"],
                    "eval_id": result["eval_id"],
                    "method": method,
                    "reference_text": result["ground_truth_response"],
                    "reference_ei": result["ground_truth_response_ei"],
                    "generated_text": generation["response"],
                    "generated_ei": generation["response_ei"],
                    "scores": generation["scores"],
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
        "compute_bertscore",
    ):
        signature_config.pop(key, None)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("exp1_protocol.py"),
        Path(__file__).with_name("exp2_schema.py"),
        Path(__file__).with_name("exp2_framework.py"),
        Path(__file__).with_name("exp2_generation.py"),
        Path(__file__).with_name("empathy_alignment_analysis.py"),
        Path(__file__).with_name("operation_checkpoint.py"),
        Path(__file__).parents[1] / "prediction.py",
    ]
    dataset_dir = Path(config.dataset_dir)
    chat_files = {
        dataset_dir / split[key]
        for split in splits
        for key in ("train_chat", "test_chat")
    }
    for split in splits:
        test_chat = load_json(str(dataset_dir / split["test_chat"]))
        target = canonical_speaker(test_chat, split["speaker"])
        partner = next(
            speaker for speaker in message_speakers(test_chat)
            if speaker.casefold() != target.casefold()
        )
        partner_split = next(
            item for item in REALTALK_PERSONA_SPLITS
            if item["speaker"].casefold() == partner.casefold()
        )
        chat_files.add(dataset_dir / partner_split["train_chat"])
    chat_files = sorted(chat_files)
    return stable_hash({
        "schema_version": 2,
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


def _result_sort_key(item: Dict[str, Any]) -> tuple[str, str, int]:
    return (
        item["test_chat_file"],
        item["speaker"].casefold(),
        int(item["message_level_index"]),
    )


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
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="Skip the response-generation stage.",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Compute BERTScore in one batch after response generation.",
    )
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
        run_generation=not args.prediction_only,
        compute_bertscore=args.bertscore,
        fresh=args.fresh,
    ))


if __name__ == "__main__":
    main()

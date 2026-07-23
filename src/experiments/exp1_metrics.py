"""Offline-recomputable classification metrics for Experiment 1."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def classification_report(
    references: Sequence[str],
    predictions: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Return accuracy, macro-F1, support, and a fixed-label confusion matrix."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have equal length")
    if len(set(labels)) != len(labels):
        raise ValueError("classification labels must be unique")

    label_list = list(labels)
    label_to_index = {label: index for index, label in enumerate(label_list)}
    matrix = [[0 for _ in label_list] for _ in label_list]

    for reference, prediction in zip(references, predictions):
        if reference not in label_to_index:
            raise ValueError(f"unknown reference label: {reference}")
        if prediction not in label_to_index:
            raise ValueError(f"unknown prediction label: {prediction}")
        matrix[label_to_index[reference]][label_to_index[prediction]] += 1

    total = len(references)
    correct = sum(matrix[index][index] for index in range(len(label_list)))
    per_class: Dict[str, Dict[str, float | int]] = {}
    active_f1: List[float] = []
    fixed_f1: List[float] = []

    for index, label in enumerate(label_list):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        fixed_f1.append(f1)
        if support or predicted:
            active_f1.append(f1)
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
            "predicted": predicted,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(_mean(active_f1), 6),
        "macro_f1_fixed_labels": round(_mean(fixed_f1), 6),
        "macro_f1_definition": (
            "unweighted mean over labels present in reference or prediction"
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "rows": "reference",
            "columns": "prediction",
            "labels": label_list,
            "matrix": matrix,
        },
    }


def chat_macro_report(
    records: Sequence[Dict[str, str]],
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Average independently computed chat-level classification metrics."""
    by_chat: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        by_chat.setdefault(record["chat_file"], []).append(record)

    per_chat: Dict[str, Dict[str, Any]] = {}
    for chat_file, chat_records in sorted(by_chat.items()):
        per_chat[chat_file] = classification_report(
            [record["reference"] for record in chat_records],
            [record["prediction"] for record in chat_records],
            labels,
        )

    return {
        "num_chats": len(per_chat),
        "accuracy": round(
            _mean(report["accuracy"] for report in per_chat.values()), 6
        ),
        "macro_f1": round(
            _mean(report["macro_f1"] for report in per_chat.values()), 6
        ),
        "per_chat": per_chat,
    }


def paired_correctness_counts(
    results: Sequence[Dict[str, Any]],
    candidate: str,
    baseline: str,
    label_field: str,
) -> Dict[str, int | float]:
    """Preserve paired outcome counts needed for later significance tests."""
    both_correct = 0
    candidate_only = 0
    baseline_only = 0
    both_wrong = 0

    for result in results:
        reference = result["reference"][label_field]
        candidate_correct = (
            result["methods"][candidate]["prediction"][label_field] == reference
        )
        baseline_correct = (
            result["methods"][baseline]["prediction"][label_field] == reference
        )
        if candidate_correct and baseline_correct:
            both_correct += 1
        elif candidate_correct:
            candidate_only += 1
        elif baseline_correct:
            baseline_only += 1
        else:
            both_wrong += 1

    total = both_correct + candidate_only + baseline_only + both_wrong
    return {
        "total_pairs": total,
        "both_correct": both_correct,
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only,
        "both_wrong": both_wrong,
        "accuracy_delta": round(
            (candidate_only - baseline_only) / total, 6
        ) if total else 0.0,
    }


def build_metric_records(
    results: Sequence[Dict[str, Any]],
    methods: Iterable[str],
) -> List[Dict[str, Any]]:
    """Build a compact long-form table without discarding the source results."""
    records: List[Dict[str, Any]] = []
    for result in results:
        for method in methods:
            prediction = result["methods"][method]["prediction"]
            reference = result["reference"]
            records.append({
                "result_id": result["result_id"],
                "chat_file": result["chat_file"],
                "eval_id": result["eval_id"],
                "boundary_index": result["boundary_index"],
                "target_session": result["target_session"],
                "user_speaker": result.get("user_speaker"),
                "target_dia_ids": result.get("target_dia_ids", []),
                "target_message": result.get("target_message"),
                "method": method,
                "reference_emotion": reference["emotion"],
                "predicted_emotion": prediction["emotion"],
                "emotion_correct": prediction["emotion"] == reference["emotion"],
                "reference_sentiment": reference["sentiment"],
                "predicted_sentiment": prediction["sentiment"],
                "sentiment_correct": (
                    prediction["sentiment"] == reference["sentiment"]
                ),
                "reference_topic": reference["topic"],
                "predicted_topic": prediction["topic"],
                "topic_consistency": result["methods"][method]["scores"][
                    "topic_consistency"
                ],
                "completed_profile_sessions": result.get("profile", {}).get(
                    "completed_sessions"
                ),
                "profile_history_hash": result.get("profile", {}).get(
                    "history_hash"
                ),
                "profile_characters": (
                    result.get("profile", {}).get("explicit_characters")
                    if method == "explicit_model"
                    else result.get("profile", {}).get("flat_characters")
                    if method == "flat_profile"
                    else None
                ),
                "portrait_entropy": result.get("profile", {}).get(
                    "explicit_portrait_entropy"
                ),
                "context_session_count": result.get("context", {}).get(
                    "actual_session_count"
                ),
                "context_semantic_turns": result.get("context", {}).get(
                    "semantic_turns"
                ),
                "context_characters": result.get("context", {}).get(
                    "characters"
                ),
                "context_truncated": result.get("context", {}).get("truncated"),
            })
    return records


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0

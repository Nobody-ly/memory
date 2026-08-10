from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import ChatBackend, OpenAICompatibleChatBackend
from .dataset import PersonaEmpDataset
from .generation import _parse_json_object, prompt_hash


TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
LEVELS = ("low", "medium", "high")
PAPER_BIG_FIVE_MODEL = "deepseek-v4-flash"
PAPER_TOTAL_USERS = 2780
PAPER_TOTAL_QUERIES = 5463
PAPER_TEST_TARGET_USERS = 278
PAPER_TEST_TARGET_QUERIES = 546
PAPER_TEST_TARGET_CATEGORIES = {
    "ES": 273,
    "HEQ": 186,
    "SS": 87,
}
BIG_FIVE_SYSTEM_PROMPT = """You annotate a grounded user persona with the Big
Five personality model. Assign low, medium, or high for each trait. Use only
the supplied persona evidence and avoid adding facts. Return the requested
JSON and no commentary."""
BIG_FIVE_USER_PROMPT = """User persona:
{persona}
"""
BIG_FIVE_SCHEMA = {
    "name": "personaemp_big_five",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            trait: {"type": "string", "enum": list(LEVELS)}
            for trait in TRAITS
        },
        "required": list(TRAITS),
        "additionalProperties": False,
    },
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def random_user_split(
    user_ids: list[str],
    *,
    seed: int = 42,
    test_ratio: float = 0.1,
) -> tuple[list[str], list[str]]:
    if len(user_ids) < 2:
        raise ValueError("at least two users are required")
    unique = sorted(set(user_ids))
    if len(unique) != len(user_ids):
        raise ValueError("user_ids must be unique")
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_ratio))
    test = sorted(shuffled[:test_count])
    train = sorted(shuffled[test_count:])
    return train, test


class BigFiveCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.values: dict[str, dict[str, str]] = {}
        if path.is_file():
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    cache_key = str(value.get("cache_key") or "")
                    if cache_key:
                        self.values[cache_key] = dict(value["labels"])

    def save(
        self,
        cache_key: str,
        user_id: str,
        labels: dict[str, str],
        provenance: dict[str, str],
    ) -> None:
        if cache_key in self.values:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as destination:
            destination.write(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "user_id": user_id,
                        "labels": labels,
                        "provenance": provenance,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            destination.flush()
            os.fsync(destination.fileno())
        self.values[cache_key] = labels


class BigFiveLabeler:
    def __init__(self, backend: ChatBackend, cache: BigFiveCache) -> None:
        if backend.model != PAPER_BIG_FIVE_MODEL:
            raise ValueError(
                "PersonaEmp Appendix A.1 requires "
                f"{PAPER_BIG_FIVE_MODEL}; got {backend.model}"
            )
        self.backend = backend
        self.cache = cache

    def label(self, user_id: str, persona: str) -> dict[str, str]:
        provenance = {
            "model": self.backend.model,
            "system_prompt_sha256": prompt_hash(BIG_FIVE_SYSTEM_PROMPT),
            "user_prompt_sha256": prompt_hash(BIG_FIVE_USER_PROMPT),
            "schema_sha256": prompt_hash(
                json.dumps(BIG_FIVE_SCHEMA, sort_keys=True)
            ),
            "persona_sha256": prompt_hash(persona),
        }
        cache_key = hashlib.sha256(
            json.dumps(
                {"user_id": user_id, **provenance},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if cache_key in self.cache.values:
            return self.cache.values[cache_key]
        result = self.backend.chat(
            BIG_FIVE_SYSTEM_PROMPT,
            BIG_FIVE_USER_PROMPT.format(persona=persona),
            temperature=0.0,
            max_tokens=250,
            response_schema=BIG_FIVE_SCHEMA,
        )
        value = _parse_json_object(result.content)
        labels = {trait: str(value.get(trait) or "") for trait in TRAITS}
        if any(level not in LEVELS for level in labels.values()):
            raise ValueError(f"invalid Big Five labels for user {user_id}")
        self.cache.save(cache_key, user_id, labels, provenance)
        return labels


@dataclass(frozen=True)
class OODSplit:
    selected_k: int
    silhouette: float
    held_out_cluster: int
    assignments: dict[str, int]
    train_users: tuple[str, ...]
    test_users: tuple[str, ...]
    leave_one_cluster_out: dict[str, tuple[str, ...]]


def _encode(labels: dict[str, dict[str, str]], users: list[str]) -> Any:
    import numpy as np

    mapping = {"low": 0, "medium": 1, "high": 2}
    return np.asarray(
        [[mapping[labels[user][trait]] for trait in TRAITS] for user in users],
        dtype=int,
    )


def _centroid_distance(left: Any, right: Any) -> float:
    return sum(a != b for a, b in zip(left, right)) / len(left)


def build_ood_split(
    labels: dict[str, dict[str, str]],
    *,
    seed: int = 42,
    k_min: int = 2,
    k_max: int = 8,
) -> OODSplit:
    try:
        from kmodes.kmodes import KModes
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise RuntimeError("kmodes and scikit-learn are required for OOD split") from exc

    users = sorted(labels)
    matrix = _encode(labels, users)
    candidates: list[tuple[float, int, Any, Any]] = []
    upper = min(k_max, len(users) - 1)
    for k in range(k_min, upper + 1):
        model = KModes(
            n_clusters=k,
            init="Huang",
            n_init=20,
            random_state=seed,
            verbose=0,
        )
        assignments = model.fit_predict(matrix)
        unique = set(int(value) for value in assignments)
        if len(unique) < 2 or len(unique) >= len(users):
            continue
        score = float(silhouette_score(matrix, assignments, metric="hamming"))
        candidates.append((score, k, assignments, model.cluster_centroids_))
    if not candidates:
        raise RuntimeError("no valid KModes clustering candidate")
    score, selected_k, assignments, centroids = sorted(
        candidates,
        key=lambda value: (-value[0], value[1]),
    )[0]
    mean_distances = []
    for index, centroid in enumerate(centroids):
        others = [
            _centroid_distance(centroid, other)
            for other_index, other in enumerate(centroids)
            if other_index != index
        ]
        mean_distances.append(sum(others) / len(others))
    held_out = sorted(
        range(selected_k),
        key=lambda index: (-mean_distances[index], index),
    )[0]
    assignment_map = {
        user: int(cluster) for user, cluster in zip(users, assignments)
    }
    test_users = tuple(
        user for user in users if assignment_map[user] == held_out
    )
    train_users = tuple(
        user for user in users if assignment_map[user] != held_out
    )
    loco = {
        str(cluster): tuple(
            user for user in users if assignment_map[user] == cluster
        )
        for cluster in range(selected_k)
    }
    return OODSplit(
        selected_k=selected_k,
        silhouette=score,
        held_out_cluster=held_out,
        assignments=assignment_map,
        train_users=train_users,
        test_users=test_users,
        leave_one_cluster_out=loco,
    )


def _subset(dataset: PersonaEmpDataset, user_ids: set[str]) -> list[dict[str, Any]]:
    return [
        session
        for session in dataset.raw_sessions
        if str(session.get("session_id") or session.get("original_sid") or "")
        in user_ids
    ]


def _user_id(session: dict[str, Any]) -> str:
    return str(session.get("session_id") or session.get("original_sid") or "")


def _canonical_category(value: Any) -> str:
    aliases = {
        "emotional support": "ES",
        "es": "ES",
        "high-eq interaction": "HEQ",
        "hei": "HEQ",
        "heq": "HEQ",
        "social strategy": "SS",
        "ss": "SS",
    }
    text = str(value or "unknown").strip()
    return aliases.get(text.lower(), text)


def _test_set_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(
        _canonical_category(query.get("category"))
        for record in records
        for query in record.get("queries") or []
        if isinstance(query, dict)
    )
    query_counts = [
        len(record.get("queries") or [])
        for record in records
        if isinstance(record, dict)
    ]
    return {
        "users": len(records),
        "queries": sum(query_counts),
        "category_distribution": dict(sorted(categories.items())),
        "queries_per_user": {
            "minimum": min(query_counts) if query_counts else 0,
            "maximum": max(query_counts) if query_counts else 0,
            "mean": (
                round(sum(query_counts) / len(query_counts), 6)
                if query_counts
                else 0
            ),
        },
    }


def fixed_random_test_users(
    user_ids: list[str],
    *,
    target_users: int = PAPER_TEST_TARGET_USERS,
    seed: int = 42,
) -> list[str]:
    unique = sorted(set(user_ids))
    if len(unique) != len(user_ids):
        raise ValueError("user_ids must be unique")
    if len(unique) < target_users:
        raise ValueError(
            f"need at least {target_users} eligible users; got {len(unique)}"
        )
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:target_users])


def build_target_sized_ood_split(
    labels: dict[str, dict[str, str]],
    *,
    target_users: int = PAPER_TEST_TARGET_USERS,
    seed: int = 42,
    k_min: int = 2,
    k_max: int = 10,
) -> OODSplit:
    """Choose a complete KModes cluster nearest the paper-scale test size."""
    try:
        from kmodes.kmodes import KModes
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise RuntimeError("kmodes and scikit-learn are required for OOD split") from exc

    users = sorted(labels)
    if len(users) < 2:
        raise ValueError("at least two labelled users are required")
    matrix = _encode(labels, users)
    candidates: list[tuple[Any, ...]] = []
    upper = min(k_max, len(users) - 1)
    for k in range(k_min, upper + 1):
        model = KModes(
            n_clusters=k,
            init="Huang",
            n_init=20,
            random_state=seed,
            verbose=0,
        )
        assignments = model.fit_predict(matrix)
        unique_clusters = sorted(set(int(value) for value in assignments))
        if len(unique_clusters) < 2 or len(unique_clusters) >= len(users):
            continue
        silhouette = float(
            silhouette_score(matrix, assignments, metric="hamming")
        )
        centroids = model.cluster_centroids_
        for cluster in unique_clusters:
            size = sum(int(value) == cluster for value in assignments)
            centroid = centroids[cluster]
            distances = [
                _centroid_distance(centroid, other)
                for index, other in enumerate(centroids)
                if index != cluster
            ]
            isolation = sum(distances) / len(distances)
            candidates.append(
                (
                    abs(size - target_users),
                    -isolation,
                    -silhouette,
                    k,
                    cluster,
                    assignments,
                )
            )
    if not candidates:
        raise RuntimeError("no valid KModes clustering candidate")
    _, _, negative_silhouette, selected_k, held_out, assignments = sorted(
        candidates,
        key=lambda value: value[:5],
    )[0]
    assignment_map = {
        user: int(cluster) for user, cluster in zip(users, assignments)
    }
    test_users = tuple(
        user for user in users if assignment_map[user] == held_out
    )
    train_users = tuple(
        user for user in users if assignment_map[user] != held_out
    )
    loco = {
        str(cluster): tuple(
            user for user in users if assignment_map[user] == cluster
        )
        for cluster in range(selected_k)
    }
    return OODSplit(
        selected_k=selected_k,
        silhouette=-negative_silhouette,
        held_out_cluster=held_out,
        assignments=assignment_map,
        train_users=train_users,
        test_users=test_users,
        leave_one_cluster_out=loco,
    )


def build_test_only_split_artifacts(
    dataset: PersonaEmpDataset,
    output_dir: Path,
    labeler: BigFiveLabeler,
    *,
    target_users: int = PAPER_TEST_TARGET_USERS,
    seed: int = 42,
) -> dict[str, Any]:
    """Build paper-scale Random/OOD tests without materialising train data."""
    user_ids = [_user_id(session) for session in dataset.raw_sessions]
    if any(not value for value in user_ids):
        raise ValueError("every final record must have a stable user id")
    random_users = fixed_random_test_users(
        user_ids,
        target_users=target_users,
        seed=seed,
    )
    personas = {
        _user_id(session): str(
            (session.get("persona") or {}).get("persona_profile")
            if isinstance(session.get("persona"), dict)
            else session.get("persona") or ""
        )
        for session in dataset.raw_sessions
    }
    if any(not personas[user_id].strip() for user_id in user_ids):
        raise ValueError("every final record must retain a generated persona")
    labels = {
        user_id: labeler.label(user_id, personas[user_id])
        for user_id in sorted(user_ids)
    }
    ood = build_target_sized_ood_split(
        labels,
        target_users=target_users,
        seed=seed,
    )
    random_records = _subset(dataset, set(random_users))
    ood_records = _subset(dataset, set(ood.test_users))
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "random_test": output_dir / "random_test.json",
        "ood_test": output_dir / "ood_test.json",
        "big_five_labels": output_dir / "big_five_labels.json",
    }
    _atomic_json(files["random_test"], random_records)
    _atomic_json(files["ood_test"], ood_records)
    _atomic_json(files["big_five_labels"], labels)
    manifest = {
        "protocol": "personaemp_public_task1_test_only_split_v1",
        "dataset_sha256": dataset.fingerprint,
        "paper_scale_reference": {
            "total_users": PAPER_TOTAL_USERS,
            "total_queries": PAPER_TOTAL_QUERIES,
            "test_target_users": target_users,
            "test_target_queries": PAPER_TEST_TARGET_QUERIES,
            "test_target_categories": PAPER_TEST_TARGET_CATEGORIES,
            "target_basis": "absolute 10-percent paper scale; no train artifact",
        },
        "candidate_pool": _test_set_summary(dataset.raw_sessions),
        "random": {
            "seed": seed,
            "selection": "fixed-count seeded user sample",
            "test_users": random_users,
            "summary": _test_set_summary(random_records),
        },
        "ood": {
            "traits": list(TRAITS),
            "levels": list(LEVELS),
            "label_model": labeler.backend.model,
            "big_five_prompt_status": "reconstructed_from_paper_description",
            "prompt_sha256": prompt_hash(BIG_FIVE_SYSTEM_PROMPT),
            "k_search": [2, 10],
            "selected_k": ood.selected_k,
            "silhouette_hamming": ood.silhouette,
            "held_out_rule": "complete cluster nearest absolute target size; ties prefer isolation then silhouette",
            "held_out_cluster": ood.held_out_cluster,
            "assignments": ood.assignments,
            "test_users": list(ood.test_users),
            "leave_one_cluster_out": {
                cluster: list(users)
                for cluster, users in ood.leave_one_cluster_out.items()
            },
            "summary": _test_set_summary(ood_records),
        },
        "train_artifacts_created": False,
        "files": {name: str(path) for name, path in files.items()},
    }
    _atomic_json(output_dir / "split_manifest.json", manifest)
    return manifest


def build_random_split_artifacts(
    dataset: PersonaEmpDataset,
    output_dir: Path,
) -> dict[str, Any]:
    user_ids = [
        str(session.get("session_id") or session.get("original_sid") or "")
        for session in dataset.raw_sessions
    ]
    random_train, random_test = random_user_split(user_ids)
    files = {
        "random_train": output_dir / "random_train.json",
        "random_test": output_dir / "random_test.json",
    }
    _atomic_json(files["random_train"], _subset(dataset, set(random_train)))
    _atomic_json(files["random_test"], _subset(dataset, set(random_test)))
    manifest = {
        "protocol": "personaemp_public_random_split_v1",
        "dataset_sha256": dataset.fingerprint,
        "random": {
            "seed": 42,
            "ratio": "9:1",
            "train_users": random_train,
            "test_users": random_test,
        },
        "ood": {
            "status": "not_built",
            "reason": "Big Five label model credentials were not supplied",
        },
        "files": {name: str(path) for name, path in files.items()},
    }
    _atomic_json(output_dir / "split_manifest.json", manifest)
    return manifest


def build_split_artifacts(
    dataset: PersonaEmpDataset,
    output_dir: Path,
    labeler: BigFiveLabeler,
) -> dict[str, Any]:
    random_manifest = build_random_split_artifacts(dataset, output_dir)
    random_train = random_manifest["random"]["train_users"]
    random_test = random_manifest["random"]["test_users"]
    user_ids = [*random_train, *random_test]
    personas = {
        str(session.get("session_id") or session.get("original_sid") or ""):
        str((session.get("persona") or {}).get("persona_profile")
            if isinstance(session.get("persona"), dict)
            else session.get("persona") or "")
        for session in dataset.raw_sessions
    }
    labels = {
        user_id: labeler.label(user_id, personas[user_id])
        for user_id in sorted(user_ids)
    }
    ood = build_ood_split(labels)
    files = {
        "random_train": output_dir / "random_train.json",
        "random_test": output_dir / "random_test.json",
        "ood_train": output_dir / "ood_train.json",
        "ood_test": output_dir / "ood_test.json",
    }
    _atomic_json(files["random_train"], _subset(dataset, set(random_train)))
    _atomic_json(files["random_test"], _subset(dataset, set(random_test)))
    _atomic_json(files["ood_train"], _subset(dataset, set(ood.train_users)))
    _atomic_json(files["ood_test"], _subset(dataset, set(ood.test_users)))
    manifest = {
        "protocol": "personaemp_public_split_v1",
        "dataset_sha256": dataset.fingerprint,
        "random": {
            "seed": 42,
            "ratio": "9:1",
            "train_users": random_train,
            "test_users": random_test,
        },
        "ood": {
            "traits": list(TRAITS),
            "levels": list(LEVELS),
            "label_model": labeler.backend.model,
            "label_model_requirement": PAPER_BIG_FIVE_MODEL,
            "big_five_prompt_status": "reconstructed_from_paper_description",
            "prompt_sha256": prompt_hash(BIG_FIVE_SYSTEM_PROMPT),
            "k_search": [2, 8],
            "selected_k": ood.selected_k,
            "silhouette_hamming": ood.silhouette,
            "held_out_rule": "largest_mean_inter_centroid_hamming_distance",
            "held_out_cluster": ood.held_out_cluster,
            "assignments": ood.assignments,
            "train_users": list(ood.train_users),
            "test_users": list(ood.test_users),
            "leave_one_cluster_out": {
                cluster: list(users)
                for cluster, users in ood.leave_one_cluster_out.items()
            },
        },
        "files": {name: str(path) for name, path in files.items()},
    }
    _atomic_json(output_dir / "split_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build PersonaEmp Random and Big-Five OOD splits."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-prefix", default="PERSONAEMP_BIG5")
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Build the user-level 9:1 Random split without Big Five/OOD.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    dataset = PersonaEmpDataset.load(args.dataset)
    if args.random_only:
        manifest = build_random_split_artifacts(dataset, args.output_dir)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    backend = OpenAICompatibleChatBackend.from_env(args.env_prefix)
    labeler = BigFiveLabeler(
        backend,
        BigFiveCache(args.output_dir / "cache" / "big_five.jsonl"),
    )
    manifest = build_split_artifacts(dataset, args.output_dir, labeler)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

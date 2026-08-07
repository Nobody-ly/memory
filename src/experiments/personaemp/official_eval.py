from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any


OFFICIAL_COMMIT = "b555447f267b8057039aab39a4be44725718ea7f"
DIMENSIONS = ("resonation", "expression", "reception")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_prediction_alignment(
    dataset_path: Path,
    prediction_path: Path,
) -> tuple[int, int]:
    dataset = load_json(dataset_path)
    predictions = load_json(prediction_path)
    if not isinstance(dataset, list) or not isinstance(predictions, list):
        raise ValueError("dataset and predictions must both be JSON lists")
    if len(dataset) != len(predictions):
        raise ValueError(
            f"session count mismatch: dataset={len(dataset)} "
            f"predictions={len(predictions)}"
        )

    query_count = 0
    for index, (session, prediction) in enumerate(zip(dataset, predictions)):
        dataset_session_id = str(
            session.get("session_id") or session.get("original_sid") or ""
        )
        prediction_session_id = str(prediction.get("session_id") or "")
        if dataset_session_id != prediction_session_id:
            raise ValueError(
                f"session[{index}] id mismatch: "
                f"{dataset_session_id!r} != {prediction_session_id!r}"
            )

        queries = session.get("queries", [])
        responses = prediction.get("responses", [])
        if len(queries) != len(responses):
            raise ValueError(
                f"session[{index}] query count mismatch: "
                f"dataset={len(queries)} predictions={len(responses)}"
            )
        for query_index, (query, response) in enumerate(zip(queries, responses)):
            query_id = str(query.get("query_id") or "")
            response_query_id = str(response.get("query_id") or "")
            if query_id != response_query_id:
                raise ValueError(
                    f"session[{index}].query[{query_index}] id mismatch: "
                    f"{query_id!r} != {response_query_id!r}"
                )
            if not str(response.get("response") or "").strip():
                raise ValueError(
                    f"empty response for session={dataset_session_id} query={query_id}"
                )
            query_count += 1
    return len(dataset), query_count


def validate_criteria_alignment(
    dataset_path: Path,
    criteria_path: Path,
    limit: int | None = None,
) -> int:
    dataset = load_json(dataset_path)
    criteria_data = load_json(criteria_path)
    if not isinstance(dataset, list) or not isinstance(criteria_data, list):
        raise ValueError("dataset and criteria must both be JSON lists")
    if len(dataset) != len(criteria_data):
        raise ValueError(
            f"criteria session count mismatch: dataset={len(dataset)} "
            f"criteria={len(criteria_data)}"
        )

    checked = 0
    for session_index, (session, criteria_session) in enumerate(
        zip(dataset, criteria_data)
    ):
        dataset_session_id = str(
            session.get("session_id") or session.get("original_sid") or ""
        )
        criteria_session_id = str(criteria_session.get("session_id") or "")
        if dataset_session_id != criteria_session_id:
            raise ValueError(
                f"criteria session[{session_index}] id mismatch: "
                f"{dataset_session_id!r} != {criteria_session_id!r}"
            )

        queries = session.get("queries", [])
        criteria_rows = criteria_session.get("criterias", [])
        if len(queries) != len(criteria_rows):
            raise ValueError(
                f"criteria session[{session_index}] query count mismatch: "
                f"dataset={len(queries)} criteria={len(criteria_rows)}"
            )
        for query_index, (query, criteria) in enumerate(
            zip(queries, criteria_rows)
        ):
            if limit is not None and checked >= limit:
                return checked
            query_id = str(query.get("query_id") or "")
            criteria_query_id = str(criteria.get("query_id") or "")
            if query_id != criteria_query_id:
                raise ValueError(
                    f"criteria session[{session_index}].query[{query_index}] "
                    f"id mismatch: {query_id!r} != {criteria_query_id!r}"
                )
            missing = [
                dimension
                for dimension in DIMENSIONS
                if not str(criteria.get(dimension) or "").strip()
            ]
            if missing:
                raise ValueError(
                    f"incomplete criteria for session={dataset_session_id} "
                    f"query={query_id}: {', '.join(missing)}"
                )
            checked += 1
    return checked


def summarize_official_results(results_path: Path) -> dict[str, Any]:
    records = load_json(results_path)
    if not isinstance(records, list):
        raise ValueError("official result root must be a list")

    scores: dict[str, list[float]] = {dimension: [] for dimension in DIMENSIONS}
    invalid: list[dict[str, str]] = []
    for record in records:
        for dimension in DIMENSIONS:
            value = (record.get(dimension) or {}).get("score")
            if isinstance(value, (int, float)) and 1 <= float(value) <= 5:
                scores[dimension].append(float(value))
            else:
                invalid.append(
                    {
                        "session_id": str(record.get("session_id") or ""),
                        "query_id": str(record.get("query_id") or ""),
                        "dimension": dimension,
                    }
                )

    means = {
        dimension: round(mean(values), 6) if values else None
        for dimension, values in scores.items()
    }
    valid_means = [value for value in means.values() if value is not None]
    raw_average = round(mean(valid_means), 6) if valid_means else None
    return {
        "records": len(records),
        "valid_scores": {
            dimension: len(values) for dimension, values in scores.items()
        },
        "invalid_scores": invalid,
        "resonation": means["resonation"],
        "expression": means["expression"],
        "reception": means["reception"],
        "average_raw_1_to_5": raw_average,
        "average_normalized_0_to_1": (
            round(raw_average / 5.0, 6) if raw_average is not None else None
        ),
    }


def _valid_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and 1 <= float(value) <= 5


def _merge_retry_results(
    previous: list[dict[str, Any]],
    retried: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep accepted scores and fill only cells missing from an earlier run."""
    previous_by_id = {
        (str(row.get("session_id") or ""), str(row.get("query_id") or "")): row
        for row in previous
    }
    retried_ids = {
        (str(row.get("session_id") or ""), str(row.get("query_id") or ""))
        for row in retried
    }
    if len(previous_by_id) != len(previous) or set(previous_by_id) != retried_ids:
        raise ValueError("cannot merge judge retries with different result identities")

    merged = json.loads(json.dumps(retried, ensure_ascii=False))
    for row in merged:
        identity = (
            str(row.get("session_id") or ""),
            str(row.get("query_id") or ""),
        )
        old_row = previous_by_id[identity]
        for dimension in DIMENSIONS:
            old_dimension = old_row.get(dimension) or {}
            if _valid_score(old_dimension.get("score")):
                row[dimension] = old_dimension
    return merged


def _compatible_previous_results(
    output: Path,
    *,
    judge_model: str,
    input_hashes: dict[str, str],
) -> list[dict[str, Any]] | None:
    summary_path = output.with_suffix(".summary.json")
    if not output.is_file() or not summary_path.is_file():
        return None
    try:
        summary = load_json(summary_path)
        summary_inputs = summary.get("inputs") or {}
        if summary.get("judge_model") != judge_model:
            return None
        for key in ("dataset_sha256", "predictions_sha256", "criteria_sha256"):
            if summary_inputs.get(key) != input_hashes.get(key):
                return None
        records = load_json(output)
        return records if isinstance(records, list) else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _completed_judge_output(
    output: Path,
    *,
    expected_records: int,
    judge_model: str,
    input_hashes: dict[str, str],
) -> bool:
    summary_path = output.with_suffix(".summary.json")
    if not output.is_file() or not summary_path.is_file():
        return False
    try:
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            return False
        if int(summary.get("records", -1)) != expected_records:
            return False
        if summary.get("judge_model") != judge_model:
            return False
        if summary.get("inputs") != input_hashes:
            return False
        if summary.get("invalid_scores"):
            return False
        valid_scores = summary.get("valid_scores") or {}
        return all(
            int(valid_scores.get(dimension, -1)) == expected_records
            for dimension in DIMENSIONS
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def verify_official_checkout(repository: Path) -> None:
    if _git_head(repository) != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"official checkout must be pinned to {OFFICIAL_COMMIT}"
        )
    for relative_path in (
        "evaluation/prepare_criteria.py",
        "evaluation/eval.py",
        "evaluation/api_call.py",
        "evaluation/utils.py",
    ):
        if not (repository / relative_path).is_file():
            raise RuntimeError(f"official checkout is missing {relative_path}")


def _api_environment(prefix: str, target_prefix: str) -> dict[str, str]:
    environment = dict(os.environ)
    missing: list[str] = []
    for name in ("API_KEY", "BASE_URL", "MODEL"):
        source_name = f"{prefix}_{name}"
        value = os.getenv(source_name, "").strip()
        if not value:
            missing.append(source_name)
        else:
            environment[f"{target_prefix}_{name}"] = value
    if missing:
        raise RuntimeError(
            "missing evaluation environment variables: " + ", ".join(missing)
        )
    return environment


def run_prepare_criteria(args: argparse.Namespace) -> None:
    repository = args.official_repo.resolve()
    verify_official_checkout(repository)
    environment = _api_environment(args.env_prefix, "OPENAI")
    command = [
        str(args.python),
        str(repository / "evaluation" / "prepare_criteria.py"),
        "--input",
        str(args.dataset.resolve()),
        "--output",
        str(args.output.resolve()),
        "--model",
        environment["OPENAI_MODEL"],
        "--concurrency",
        str(args.concurrency),
        "--temperature",
        str(args.temperature),
        "--max-retries",
        str(args.max_retries),
        "--resume",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    subprocess.run(
        command,
        cwd=repository / "evaluation",
        env=environment,
        check=True,
    )
    if args.limit is not None and args.limit % len(DIMENSIONS) != 0:
        raise ValueError(
            "--limit counts criteria jobs and must be a multiple of 3 "
            "to complete whole queries"
        )
    expected_queries = (
        args.limit // len(DIMENSIONS)
        if args.limit is not None
        else None
    )
    validate_criteria_alignment(
        args.dataset,
        args.output,
        expected_queries,
    )
    manifest = {
        "protocol": "personaemp_official_fixed_criteria_v1",
        "official_commit": OFFICIAL_COMMIT,
        "model": environment["OPENAI_MODEL"],
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": _sha256(args.dataset),
        },
        "criteria": {
            "path": str(args.output.resolve()),
            "sha256": _sha256(args.output),
        },
        "temperature": args.temperature,
        "limit_jobs": args.limit,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_judge(args: argparse.Namespace) -> None:
    repository = args.official_repo.resolve()
    verify_official_checkout(repository)
    _session_count, query_count = validate_prediction_alignment(
        args.dataset,
        args.predictions,
    )
    validate_criteria_alignment(args.dataset, args.criteria, args.limit)
    environment = _api_environment(args.env_prefix, "EVAL")
    expected_records = min(args.limit, query_count) if args.limit else query_count
    input_hashes = {
        "dataset_sha256": _sha256(args.dataset),
        "predictions_sha256": _sha256(args.predictions),
        "criteria_sha256": _sha256(args.criteria),
        "results_sha256": _sha256(args.output) if args.output.is_file() else "",
    }
    if getattr(args, "resume", False) and _completed_judge_output(
        args.output,
        expected_records=expected_records,
        judge_model=environment["EVAL_MODEL"],
        input_hashes=input_hashes,
    ):
        return
    previous_results = _compatible_previous_results(
        args.output,
        judge_model=environment["EVAL_MODEL"],
        input_hashes=input_hashes,
    )
    command = [
        str(args.python),
        str(repository / "evaluation" / "eval.py"),
        "--dataset_path",
        str(args.dataset.resolve()),
        "--predict_path",
        str(args.predictions.resolve()),
        "--criteria_path",
        str(args.criteria.resolve()),
        "--eval_model",
        environment["EVAL_MODEL"],
        "--eval_batch_size",
        str(args.concurrency),
        "--eval_temperature",
        str(args.temperature),
    ]
    if args.limit is not None:
        command.extend(["--eval_num", str(args.limit)])
    subprocess.run(
        command,
        cwd=repository / "evaluation",
        env=environment,
        check=True,
    )

    generated_path = args.predictions.with_name(
        f"{args.predictions.stem}_fix_criteria_res.json"
    )
    if not generated_path.is_file():
        raise RuntimeError(
            f"official evaluator did not create expected output: {generated_path}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_results = load_json(generated_path)
    if not isinstance(generated_results, list):
        raise ValueError("official judge result root must be a list")
    if previous_results is not None:
        generated_results = _merge_retry_results(
            previous_results,
            generated_results,
        )
    args.output.write_text(
        json.dumps(generated_results, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    summary = summarize_official_results(args.output)
    summary["judge_model"] = environment["EVAL_MODEL"]
    summary["judge_name"] = args.judge_name
    summary["official_commit"] = OFFICIAL_COMMIT
    summary["inputs"] = {
        "dataset_sha256": _sha256(args.dataset),
        "predictions_sha256": _sha256(args.predictions),
        "criteria_sha256": _sha256(args.criteria),
        "results_sha256": _sha256(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if summary["invalid_scores"]:
        raise RuntimeError(
            f"judge returned {len(summary['invalid_scores'])} invalid scores; "
            "resume will retry only the missing cells"
        )


def _judge_spec(value: str) -> tuple[str, str]:
    name, separator, prefix = value.partition(":")
    if not separator or not name.strip() or not prefix.strip():
        raise ValueError("--judge must use NAME:ENV_PREFIX")
    return name.strip(), prefix.strip()


def run_suite(args: argparse.Namespace) -> None:
    methods = ("base_model", "memory", "rag", "ours")
    judges = [_judge_spec(value) for value in args.judge]
    outputs: dict[str, dict[str, str]] = {}
    for judge_name, env_prefix in judges:
        judge_outputs: dict[str, str] = {}
        for method in methods:
            predictions = args.predictions_dir / f"{method}.json"
            if not predictions.is_file():
                raise FileNotFoundError(
                    f"missing {method} predictions: {predictions}"
                )
            output = (
                args.output_dir
                / args.split_name
                / judge_name
                / f"{method}.json"
            )
            run_judge(
                argparse.Namespace(
                    official_repo=args.official_repo,
                    dataset=args.dataset,
                    predictions=predictions,
                    criteria=args.criteria,
                    output=output,
                    judge_name=judge_name,
                    env_prefix=env_prefix,
                    python=args.python,
                    concurrency=args.concurrency,
                    temperature=args.temperature,
                    limit=args.limit,
                    resume=args.resume,
                )
            )
            judge_outputs[method] = str(output)
        outputs[judge_name] = judge_outputs
    manifest = {
        "protocol": "personaemp_official_dual_judge_suite_v1",
        "official_commit": OFFICIAL_COMMIT,
        "split": args.split_name,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "criteria": str(args.criteria.resolve()),
        "criteria_sha256": _sha256(args.criteria),
        "judges": outputs,
        "judge_models": {
            judge_name: os.getenv(f"{env_prefix}_MODEL", "").strip()
            for judge_name, env_prefix in judges
        },
        "methods": list(methods),
    }
    manifest_path = args.output_dir / args.split_name / "evaluation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pinned wrapper around the official PersonaEmp evaluator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    criteria = subparsers.add_parser("prepare-criteria")
    criteria.add_argument("--official-repo", type=Path, required=True)
    criteria.add_argument("--dataset", type=Path, required=True)
    criteria.add_argument("--output", type=Path, required=True)
    criteria.add_argument("--env-prefix", default="PERSONAEMP_CRITERIA")
    criteria.add_argument("--python", type=Path, default=Path(sys.executable))
    criteria.add_argument("--concurrency", type=int, default=8)
    criteria.add_argument("--temperature", type=float, default=0.2)
    criteria.add_argument("--max-retries", type=int, default=6)
    criteria.add_argument("--limit", type=int, default=None)
    criteria.set_defaults(handler=run_prepare_criteria)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--official-repo", type=Path, required=True)
    judge.add_argument("--dataset", type=Path, required=True)
    judge.add_argument("--predictions", type=Path, required=True)
    judge.add_argument("--criteria", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--judge-name", required=True)
    judge.add_argument("--env-prefix", required=True)
    judge.add_argument("--python", type=Path, default=Path(sys.executable))
    judge.add_argument("--concurrency", type=int, default=8)
    judge.add_argument("--temperature", type=float, default=0.3)
    judge.add_argument("--limit", type=int, default=None)
    judge.add_argument("--resume", action="store_true")
    judge.set_defaults(handler=run_judge)

    suite = subparsers.add_parser("suite")
    suite.add_argument("--official-repo", type=Path, required=True)
    suite.add_argument("--dataset", type=Path, required=True)
    suite.add_argument("--predictions-dir", type=Path, required=True)
    suite.add_argument("--criteria", type=Path, required=True)
    suite.add_argument("--output-dir", type=Path, required=True)
    suite.add_argument("--split-name", choices=("random", "ood"), required=True)
    suite.add_argument(
        "--judge",
        action="append",
        required=True,
        help="NAME:ENV_PREFIX; pass once for Qwen and once for DeepSeek.",
    )
    suite.add_argument("--python", type=Path, default=Path(sys.executable))
    suite.add_argument("--concurrency", type=int, default=8)
    suite.add_argument("--temperature", type=float, default=0.3)
    suite.add_argument("--limit", type=int, default=None)
    suite.add_argument("--resume", action="store_true")
    suite.set_defaults(handler=run_suite)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

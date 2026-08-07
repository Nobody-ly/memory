from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import OpenAICompatibleChatBackend
from .dataset import PersonaEmpDataset
from .generation import (
    BaseModelGenerator,
    DeepEmpathyGenerator,
    JsonCache,
    MemoryGenerator,
    MemorySummaryBuilder,
    ProfileBuilder,
    ProfileCache,
    RAGGenerator,
    RAG_ENCODER_MODEL,
    RAGRetriever,
    SentenceTransformerEncoder,
)
from .runner import PersonaEmpRunner, RunConfiguration

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PersonaEmp Exp1 predictions without changing core prompts."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("base_model", "memory", "rag", "ours"),
        default=("base_model", "memory", "rag", "ours"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--balanced-per-category",
        type=int,
        default=None,
        help="Select this many ES, HEQ and SS queries for a balanced pilot.",
    )
    parser.add_argument(
        "--dataset-provenance",
        default="unknown",
        choices=(
            "official",
            "regenerated",
            "public_reconstruction",
            "paper_case_pilot",
            "unknown",
        ),
    )
    parser.add_argument(
        "--expected-table1-dataset-sha256",
        default=None,
        help=(
            "Only a matching explicit fingerprint permits direct comparison "
            "with PersonaEmp Table 1."
        ),
    )
    parser.add_argument(
        "--env-prefix",
        default="PERSONAEMP_GENERATOR",
        help="Environment variable prefix for the OpenAI-compatible generator.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate schema and print dataset metadata without calling a model.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    dataset = PersonaEmpDataset.load(args.dataset)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "path": str(dataset.path),
                    "sha256": dataset.fingerprint,
                    "sessions": len(dataset.raw_sessions),
                    "queries": len(dataset.samples),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    backend = OpenAICompatibleChatBackend.from_env(args.env_prefix)
    profile_cache = ProfileCache(args.output_dir / "cache" / "profiles")
    profile_builder = ProfileBuilder(backend, profile_cache)
    summary_builder = MemorySummaryBuilder(
        backend,
        JsonCache(args.output_dir / "cache" / "memory_summaries"),
    )
    generators = {
        "ours": DeepEmpathyGenerator(backend, profile_builder),
        "base_model": BaseModelGenerator(backend),
        "memory": MemoryGenerator(backend, summary_builder),
    }
    if "rag" in args.methods:
        retriever = RAGRetriever(
            SentenceTransformerEncoder(RAG_ENCODER_MODEL),
            JsonCache(args.output_dir / "cache" / "rag_embeddings"),
        )
        generators["rag"] = RAGGenerator(backend, retriever)
    selected_generators = {
        method: generators[method] for method in args.methods
    }

    repository_root = Path(__file__).resolve().parents[3]
    runner = PersonaEmpRunner(
        repository_root=repository_root,
        dataset=dataset,
        output_dir=args.output_dir,
        config=RunConfiguration(
            methods=tuple(args.methods),
            limit=args.limit,
            dataset_provenance=args.dataset_provenance,
            expected_table1_dataset_sha256=args.expected_table1_dataset_sha256,
            generator_model=backend.model,
            generator_base_url=backend.base_url,
            generator_enable_thinking=backend.enable_thinking,
            balanced_per_category=args.balanced_per_category,
        ),
        generators=selected_generators,
    )
    print(json.dumps(runner.run(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line interface for the evaluation platform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rageval import paths
from rageval.config import EvalConfig
from rageval.dataset import load_dataset
from rageval.runner import run_evaluation
from rageval.schema import RunResult


def _print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print rows as a simple aligned table."""
    if not rows:
        return

    widths = {
        column: max(
            len(column),
            *(len(str(row.get(column, ""))) for row in rows),
        )
        for column in columns
    }

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))

    for row in rows:
        print(
            "  ".join(
                str(row.get(column, "")).ljust(widths[column]) for column in columns
            )
        )


def _format_aggregates(aggregates: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"metric": name, "value": f"{value:.4f}" if isinstance(value, float) else value}
        for name, value in sorted(aggregates.items())
    ]


def _format_cases(run: RunResult) -> list[dict[str, Any]]:
    rows = []
    for case in run.cases:
        rows.append(
            {
                "id": case.test_id,
                "answerable": "yes" if case.answerable else "no",
                "correct": case.metrics.get("answer_correct", ""),
                "hit@k": case.metrics.get("hit_at_k", ""),
                "mrr": case.metrics.get("reciprocal_rank", ""),
                "abstained": "yes" if case.abstained else "no",
                "question": case.question[:52],
            }
        )
    return rows


def _build_config(args: argparse.Namespace) -> EvalConfig:
    config = EvalConfig.load(args.config)
    config.apply_overrides(args.overrides or [])

    if getattr(args, "dataset", None):
        config.dataset_path = Path(args.dataset)
    if getattr(args, "run_name", None):
        config.run_name = args.run_name
    if getattr(args, "no_cache", False):
        config.use_cache = False
    return config


def command_run(args: argparse.Namespace) -> int:
    config = _build_config(args)
    run = run_evaluation(config)

    if args.json:
        print(json.dumps(run.to_dict(), indent=2, default=str))
        return 0

    print(f"Run id:       {run.run_id}")
    print(f"Dataset:      v{run.dataset_version} ({run.dataset_hash})")
    print(f"Commit:       {run.git_commit or 'unknown'}")
    print(f"Generator:    {config.generation.provider}")
    print(f"Retriever:    {config.retrieval.retriever} (top_k={config.retrieval.top_k})")
    print()

    print("AGGREGATES")
    _print_table(_format_aggregates(run.aggregates), ["metric", "value"])
    print()

    print("CASES")
    _print_table(
        _format_cases(run),
        ["id", "answerable", "correct", "hit@k", "mrr", "abstained", "question"],
    )
    print()

    errors = [case for case in run.cases if case.error]
    if errors:
        print(f"{len(errors)} case(s) failed:")
        for case in errors:
            print(f"  case {case.test_id}: {case.error}")
        print()

    print(f"Reports written to {Path(config.reports_dir).resolve()}")
    return 1 if errors else 0


def command_cases(args: argparse.Namespace) -> int:
    config = _build_config(args)
    dataset = load_dataset(Path(config.dataset_path))

    for warning in dataset.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if dataset.warnings:
        print()

    rows = [
        {
            "id": case.id,
            "answerable": "yes" if case.answerable else "no",
            "difficulty": case.difficulty,
            "sentences": ",".join(str(s) for s in case.relevant_sentence_ids),
            "question": case.question,
        }
        for case in dataset.cases
    ]
    _print_table(rows, ["id", "answerable", "difficulty", "sentences", "question"])
    print()
    print(
        f"{len(dataset)} case(s), version {dataset.version}, hash {dataset.content_hash}"
    )
    return 0


def command_config(args: argparse.Namespace) -> int:
    config = _build_config(args)
    print(json.dumps(config.to_dict(), indent=2))
    return 0


def command_retrieve(args: argparse.Namespace) -> int:
    from rageval.runner import build_retriever, load_knowledge_base

    config = _build_config(args)
    document = load_knowledge_base(Path(config.knowledge_base_path))
    retriever = build_retriever(config, document)

    chunks = retriever.retrieve(
        args.question,
        config.retrieval.top_k,
        config.retrieval.min_score,
    )

    if not chunks:
        print("No chunks passed the relevance floor.")
        return 0

    for chunk in chunks:
        print(f"chunk {chunk.chunk_id}  score={chunk.score:.3f}  "
              f"sentences={list(chunk.sentence_ids)}")
        print(f"  {chunk.text}")
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Evaluate a retrieval-augmented generation pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--config",
            type=Path,
            help="Path to a JSON configuration file.",
        )
        subparser.add_argument(
            "--set",
            dest="overrides",
            action="append",
            metavar="SECTION.FIELD=VALUE",
            help="Override a configuration value. Repeatable.",
        )

    run_parser = sub.add_parser("run", help="Run the evaluation suite.")
    add_common(run_parser)
    run_parser.add_argument("--dataset", type=Path, help="Override the dataset path.")
    run_parser.add_argument("--run-name", help="Name for this run.")
    run_parser.add_argument("--no-cache", action="store_true", help="Disable the cache.")
    run_parser.add_argument("--json", action="store_true", help="Print the raw report.")
    run_parser.set_defaults(func=command_run)

    cases_parser = sub.add_parser("cases", help="List the dataset cases.")
    add_common(cases_parser)
    cases_parser.add_argument("--dataset", type=Path, help="Override the dataset path.")
    cases_parser.set_defaults(func=command_cases)

    config_parser = sub.add_parser("config", help="Show the effective configuration.")
    add_common(config_parser)
    config_parser.set_defaults(func=command_config)

    retrieve_parser = sub.add_parser(
        "retrieve", help="Show what the retriever returns for a question."
    )
    add_common(retrieve_parser)
    retrieve_parser.add_argument("question", help="The question to retrieve for.")
    retrieve_parser.set_defaults(func=command_retrieve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

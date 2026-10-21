"""Compare the retrievers on the same dataset.

Running the three retrievers one after another and reading two reports side
by side is tedious, and easy to get wrong. This runs all of them against the
same dataset, at their own calibrated relevance floors, and prints the
metrics that actually distinguish them.

The reports are written to a temporary directory so that a comparison never
clobbers the baseline in ``reports/``.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path
from typing import Any

from rageval.config import EvalConfig
from rageval.runner import run_evaluation

RETRIEVERS = ("lexical", "dense", "hybrid")

#: The metrics worth comparing. The score-based ``retrieval_relevance`` is
#: left out on purpose: it is a raw number in each retriever's own scale, so
#: putting 0.74 next to 0.47 side by side would invite a false comparison.
METRICS = (
    ("hit_at_k", "hit@k"),
    ("recall_at_k", "recall@k"),
    ("precision_at_k", "precision@k"),
    ("reciprocal_rank", "MRR"),
    ("answer_correct", "answer_correct"),
    ("token_f1", "token_f1"),
    ("unanswerable_abstention_rate", "abstention (unanswerable)"),
    ("unanswerable_rejection_rate", "rejection (unanswerable)"),
    ("answerable_rejection_rate", "rejection (answerable)"),
    ("abstention_correct", "abstention_correct"),
    ("hallucination_rate", "hallucination_rate"),
    ("error_count", "errors"),
)


def compare(config: EvalConfig) -> dict[str, dict[str, Any]]:
    """Run every retriever against the same dataset and collect the results."""
    results: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        for name in RETRIEVERS:
            run_config = copy.deepcopy(config)
            run_config.retrieval.retriever = name
            run_config.reports_dir = workspace
            run_config.cache_dir = workspace / "cache"
            run_config.run_name = name
            run = run_evaluation(run_config)
            results[name] = {
                "aggregates": run.aggregates,
                "min_score": run_config.retrieval.resolved_min_score,
            }

    return results


def render(results: dict[str, dict[str, Any]]) -> str:
    """Format the comparison as an aligned table."""
    names = list(RETRIEVERS)
    label_width = max(len(label) for _, label in METRICS) + 2
    column_width = 10

    lines = [
        "metric".ljust(label_width)
        + "".join(name.rjust(column_width) for name in names),
        "-" * (label_width + column_width * len(names)),
    ]
    for key, label in METRICS:
        row = label.ljust(label_width)
        for name in names:
            value = results.get(name, {}).get("aggregates", {}).get(key, 0.0)
            row += f"{value:>{column_width}.4f}"
        lines.append(row)

    lines.append("")
    floors = "  ".join(
        f"{name}={results.get(name, {}).get('min_score', 0.0)}" for name in names
    )
    lines.append(f"relevance floors: {floors}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rageval.compare",
        description="Compare the lexical, dense and hybrid retrievers.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON configuration file.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="SECTION.FIELD=VALUE",
        help="Override a configuration value. Repeatable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = EvalConfig.load(args.config)
    config.apply_overrides(args.overrides or [])

    print(render(compare(config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

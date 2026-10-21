"""Compare the pipeline's arms on the same dataset.

Running two arms one after another and reading two reports side by side is
tedious, and easy to get wrong. This runs them against the same dataset and
prints the metrics that actually distinguish them.

There are two axes:

* ``retriever`` compares the lexical, dense and hybrid retrievers against
  each other, each at its own calibrated relevance floor.
* ``generator`` compares the extractive baseline against the configured
  language model. This is the axis that answers the question the whole
  project exists to ask: does a language model answer better than copying
  the best matching sentence, and what does it give up to do so?

The reports are written to a temporary directory so that a comparison never
clobbers the baseline in ``reports/``.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path
from typing import Any, Callable

from rageval.config import EvalConfig
from rageval.runner import run_evaluation

RETRIEVERS = ("lexical", "dense", "hybrid")

#: The metrics worth comparing on the retriever axis. The score-based
#: ``retrieval_relevance`` is left out on purpose: it is a raw number in
#: each retriever's own scale, so putting 0.74 next to 0.47 side by side
#: would invite a false comparison.
RETRIEVER_METRICS = (
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

#: The metrics worth comparing on the generator axis. The retrieval
#: metrics are left out because both arms retrieve with the same retriever,
#: so they are identical by construction and would only take up space.
GENERATOR_METRICS = (
    ("answer_correct", "answer_correct"),
    ("token_recall", "token_recall"),
    ("token_f1", "token_f1"),
    ("faithfulness", "faithfulness"),
    ("answer_relevance", "answer_relevance"),
    ("abstention_correct", "abstention_correct"),
    ("unanswerable_abstention_rate", "abstention (unanswerable)"),
    ("hallucination_rate", "hallucination_rate"),
    ("abstention_rate", "abstention (all)"),
    ("mean_latency_ms", "mean latency ms"),
    ("error_count", "errors"),
)

def _run_arm(
    config: EvalConfig,
    workspace: Path,
    name: str,
    configure: Callable[[EvalConfig], None],
) -> dict[str, Any]:
    """Run one arm of a comparison and collect its results."""
    run_config = copy.deepcopy(config)
    configure(run_config)
    run_config.reports_dir = workspace
    run_config.run_name = name
    run = run_evaluation(run_config)
    return {
        "aggregates": run.aggregates,
        "min_score": run_config.retrieval.resolved_min_score,
        "generator": run_config.generation.model,
    }


def compare_retrievers(config: EvalConfig) -> dict[str, dict[str, Any]]:
    """Run every retriever against the same dataset and collect the results.

    Each retriever uses its own calibrated relevance floor, because the
    score scales are not comparable and a shared floor would quietly favour
    one of them.
    """
    results: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        for name in RETRIEVERS:
            results[name] = _run_arm(
                config,
                workspace,
                name,
                lambda run_config, chosen=name: setattr(
                    run_config.retrieval, "retriever", chosen
                ),
            )

    return results


def compare_generators(config: EvalConfig) -> dict[str, dict[str, Any]]:
    """Compare the extractive baseline against the configured language model.

    The cache is switched off for this comparison. A cached answer is a
    lookup rather than a generation, and a comparison that reported the
    lookup time as the cost of an arm would be measuring the cache instead
    of the model.
    """
    generation = config.generation
    if generation.provider == "extractive" or not generation.model:
        raise ValueError(
            "The generator comparison needs a language model to compare "
            "against. Set generation.provider=ollama and generation.model, "
            "or pass --model, for example: "
            "--axis generator --model qwen3:1.7b"
        )

    results: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        for name, provider, model in (
            ("extractive", "extractive", "extractive"),
            ("llm", generation.provider, generation.model),
        ):
            results[name] = _run_arm(
                config,
                workspace,
                name,
                lambda run_config, p=provider, m=model: (
                    setattr(run_config.generation, "provider", p),
                    setattr(run_config.generation, "model", m),
                    setattr(run_config, "use_cache", False),
                ),
            )

    return results


def render(
    results: dict[str, dict[str, Any]],
    metrics: tuple[tuple[str, str], ...],
    names: tuple[str, ...],
    footnote: str = "",
    show_floors: bool = False,
) -> str:
    """Format the comparison as an aligned table."""
    label_width = max(len(label) for _, label in metrics) + 2
    column_width = 12

    lines = [
        "metric".ljust(label_width)
        + "".join(name.rjust(column_width) for name in names),
        "-" * (label_width + column_width * len(names)),
    ]
    for key, label in metrics:
        row = label.ljust(label_width)
        for name in names:
            value = results.get(name, {}).get("aggregates", {}).get(key, 0.0)
            row += f"{value:>{column_width}.4f}"
        lines.append(row)

    if show_floors:
        floors = "  ".join(
            f"{name}={results.get(name, {}).get('min_score', 0.0)}" for name in names
        )
        lines.append("")
        lines.append(f"relevance floors: {floors}")

    if footnote:
        lines.append("")
        lines.append(footnote)

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rageval.compare",
        description="Compare the retrievers or the generators on one dataset.",
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
    parser.add_argument(
        "--axis",
        choices=("retriever", "generator"),
        default="retriever",
        help="What to compare. Default: retriever.",
    )
    parser.add_argument(
        "--model",
        help="The language model to compare against the baseline, "
        "for example qwen3:1.7b. Implies --axis generator.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = EvalConfig.load(args.config)
    config.apply_overrides(args.overrides or [])

    if args.model:
        config.generation.provider = "ollama"
        config.generation.model = args.model
        args.axis = "generator"

    if args.axis == "generator":
        results = compare_generators(config)
        print(
            render(
                results,
                GENERATOR_METRICS,
                ("extractive", "llm"),
                footnote=(
                    "the cache is off for this comparison, so the latency is "
                    "the cost of a real generation"
                ),
            )
        )
        return 0

    results = compare_retrievers(config)
    print(render(results, RETRIEVER_METRICS, RETRIEVERS, show_floors=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

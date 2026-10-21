"""Reading evaluation reports for the dashboard.

The dashboard is a view over reports that already exist on disk. Nothing here
computes a metric or calls a model: every number it shows is read from the
JSON a run wrote, so what a reader sees in the browser is exactly what the
run produced, and the deployed app needs no model server.

Keeping the logic here rather than in the Streamlit file means it can be
tested without Streamlit installed, which matters because the rest of the
project has no runtime dependencies at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rageval.paths import REPORTS_DIR

#: The name of the file that holds the judge validation report.
JUDGE_VALIDATION_FILE = "judge_validation.json"


@dataclass(frozen=True)
class MetricInfo:
    """What a metric is and how to read it."""

    key: str
    label: str
    description: str
    #: ``True`` when a bigger number is better, ``False`` when smaller is
    #: better, and ``None`` when the metric is neither.
    higher_is_better: bool | None = None
    #: ``"percent"`` renders the value as a percentage.
    kind: str = "number"


@dataclass(frozen=True)
class RunSummary:
    """One report, reduced to what a list of runs needs to show."""

    run_id: str
    path: Path
    created_at: str
    retriever: str
    generator: str
    judge: str
    n_cases: int
    metrics: dict[str, float]

    @property
    def label(self) -> str:
        """A short description of the configuration behind the run."""
        parts = [self.retriever, self.generator]
        if self.judge:
            parts.append(f"judged by {self.judge}")
        return " · ".join(part for part in parts if part)


#: The metrics shown as headline cards, in the order they are shown.
HEADLINE_METRICS: tuple[MetricInfo, ...] = (
    MetricInfo(
        "answer_correct",
        "Answer correct",
        "The share of graded answers the token rule accepts.",
        higher_is_better=True,
        kind="percent",
    ),
    MetricInfo(
        "judge_correct",
        "Judge correct",
        "The share of graded answers the judge accepts.",
        higher_is_better=True,
        kind="percent",
    ),
    MetricInfo(
        "judge_token_agreement",
        "Judge and token rule agree",
        "How often the judge and the token rule reach the same verdict.",
        higher_is_better=True,
        kind="percent",
    ),
    MetricInfo(
        "token_f1",
        "Token F1",
        "How much of the reference and the answer overlap.",
        higher_is_better=True,
    ),
    MetricInfo(
        "faithfulness",
        "Faithfulness",
        "How well the answer is supported by the retrieved context.",
        higher_is_better=True,
        kind="percent",
    ),
    MetricInfo(
        "hallucination_rate",
        "Hallucination rate",
        "The share of answers that assert something the context does not say.",
        higher_is_better=False,
        kind="percent",
    ),
    MetricInfo(
        "abstention_correct",
        "Abstention correct",
        "Answered exactly when the question was answerable.",
        higher_is_better=True,
        kind="percent",
    ),
    MetricInfo(
        "hit_at_k",
        "Hit at k",
        "The share of answerable questions the retriever found evidence for.",
        higher_is_better=True,
        kind="percent",
    ),
    MetricInfo(
        "precision_at_k",
        "Precision at k",
        "How much of the retrieved text is relevant to the question.",
        higher_is_better=True,
    ),
    MetricInfo(
        "mean_latency_ms",
        "Latency",
        "The mean time a case took, in milliseconds.",
        higher_is_better=False,
    ),
)

#: The metrics a reader is most likely to look for, by name.
HEADLINE_KEYS: tuple[str, ...] = tuple(metric.key for metric in HEADLINE_METRICS)

#: Every metric the reports carry, with a short description for the ones
#: that are not in the headline set.
OTHER_METRICS: dict[str, str] = {
    "abstained": "The share of cases the generator declined to answer.",
    "abstention_rate": "The share of cases that abstained.",
    "answer_relevance": "How directly the answer addresses the question.",
    "answerable_cases": "How many cases had an answerable question.",
    "answerable_rejection_rate": "Answerable questions the pipeline turned away.",
    "cache_hits": "How many answers came from the cache rather than the model.",
    "error_count": "How many cases failed outright.",
    "exact_match": "The share of answers that match the reference exactly.",
    "judge_completeness": "The judge's completeness score, from 1 to 5.",
    "judge_correctness": "The judge's correctness score, from 1 to 5.",
    "judge_groundedness": "The judge's groundedness score, from 1 to 5.",
    "judge_relevance": "The judge's relevance score, from 1 to 5.",
    "mean_latency_ms": "The mean time a case took, in milliseconds.",
    "numeric_match": "The share of answers that match the reference numerically.",
    "recall_at_k": "How much of the relevant text the retriever returned.",
    "reciprocal_rank": "How high up the retrieved text the relevant part was.",
    "retrieval_rejected": "The share of questions the retriever turned away.",
    "retrieval_relevance": "The retriever's own relevance score.",
    "token_recall": "How much of the reference the answer covers.",
    "total_cases": "How many cases the run covered.",
    "unanswerable_abstention_rate": "Unanswerable questions the generator declined.",
    "unanswerable_cases": "How many cases had an unanswerable question.",
    "unanswerable_rejection_rate": "Unanswerable questions the pipeline turned away.",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON report and return it."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _describe_config(run: dict[str, Any]) -> tuple[str, str, str]:
    """Pull the retriever, the generator and the judge out of a report."""
    config = run.get("config", {}) or {}
    retrieval = config.get("retrieval", {}) or {}
    generation = config.get("generation", {}) or {}
    judge = config.get("judge", {}) or {}

    retriever = str(retrieval.get("retriever", "unknown"))
    provider = str(generation.get("provider", "unknown"))
    model = str(generation.get("model", ""))
    generator = f"{provider} ({model})" if model else provider

    if judge.get("enabled"):
        judge_model = str(judge.get("model", ""))
        judge_name = (
            f"{judge.get('provider')} ({judge_model})"
            if judge_model
            else str(judge.get("provider", "judge"))
        )
    else:
        judge_name = ""

    return retriever, generator, judge_name


def summarise(run: dict[str, Any], path: Path) -> RunSummary:
    """Reduce a report to a :class:`RunSummary`."""
    retriever, generator, judge = _describe_config(run)
    aggregates = run.get("aggregates", {}) or {}
    metrics = {
        key: value for key, value in aggregates.items() if isinstance(value, (int, float))
    }
    return RunSummary(
        run_id=str(run.get("run_id", path.stem)),
        path=path,
        created_at=str(run.get("created_at", "")),
        retriever=retriever,
        generator=generator,
        judge=judge,
        n_cases=int(metrics.get("total_cases", len(run.get("cases", [])))),
        metrics=metrics,
    )


def discover_runs(directory: Path | None = None) -> list[RunSummary]:
    """Every run report in a directory, newest first.

    ``latest.json`` is a copy of the most recent run rather than a run of
    its own, so a run is listed once: if both files hold the same run, the
    one with the proper name is kept and the copy is dropped. That matters
    for the deployed copy, where the copy is the only file there is and
    the run it holds still has to be listed.

    A report that cannot be read is skipped too: one broken file should
    not empty the dashboard.
    """
    reports_dir = Path(directory) if directory is not None else REPORTS_DIR
    if not reports_dir.is_dir():
        return []

    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(reports_dir.glob("*.json")):
        if path.name == JUDGE_VALIDATION_FILE:
            continue
        try:
            run = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if "run_id" not in run:
            continue
        run_id = str(run["run_id"])
        existing = found.get(run_id)
        if existing is None or (
            existing[0].name == "latest.json" and path.name != "latest.json"
        ):
            found[run_id] = (path, run)

    summaries = [summarise(run, path) for path, run in found.values()]
    summaries.sort(key=lambda item: item.created_at, reverse=True)
    return summaries


def load_run(path: Path) -> dict[str, Any]:
    """Load one report."""
    return read_json(Path(path))


def case_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    """One flat row per case, ready for a table.

    The numbers are pulled out of the metrics dictionary rather than being
    recomputed, so the table cannot disagree with the report.
    """
    rows: list[dict[str, Any]] = []
    for case in run.get("cases", []):
        metrics = case.get("metrics", {}) or {}
        judge = case.get("judge", {}) or {}
        rows.append(
            {
                "id": case.get("test_id"),
                "question": case.get("question", ""),
                "answerable": case.get("answerable"),
                "answer": case.get("answer", ""),
                "abstained": bool(case.get("abstained")),
                "reference": case.get("reference_answer", ""),
                "token_correct": metrics.get("answer_correct"),
                "judge_correct": judge.get("correct"),
                "judge_correctness": judge.get("correctness"),
                "judge_relevance": judge.get("relevance"),
                "judge_groundedness": judge.get("groundedness"),
                "judge_completeness": judge.get("completeness"),
                "judge_reason": judge.get("reason", ""),
                "token_recall": metrics.get("token_recall"),
                "token_f1": metrics.get("token_f1"),
                "faithfulness": metrics.get("faithfulness"),
                "hit_at_k": metrics.get("hit_at_k"),
                "top_score": max(
                    (chunk.get("score", 0.0) for chunk in case.get("retrieved", [])),
                    default=0.0,
                ),
                "chunks": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "score": chunk.get("score", 0.0),
                        "text": chunk.get("text", ""),
                        "sentence_ids": chunk.get("sentence_ids", []),
                    }
                    for chunk in case.get("retrieved", [])
                ],
                "error": case.get("error"),
            }
        )
    return rows


def judge_validation_path(directory: Path | None = None) -> Path:
    """Where the judge validation report lives."""
    reports_dir = Path(directory) if directory is not None else REPORTS_DIR
    return reports_dir / JUDGE_VALIDATION_FILE


def load_judge_validation(directory: Path | None = None) -> dict[str, Any] | None:
    """Load the judge validation report, if there is one.

    ``None`` rather than an exception: the dashboard has to work in a
    checkout where the validation has not been run.
    """
    path = judge_validation_path(directory)
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def kappa_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The judge's agreement with the labels, over both subsets.

    Each row is one rater on one subset, so a table can be drawn straight
    from it without the reader having to know how the report is nested.
    """
    rows: list[dict[str, Any]] = []
    for subset, title in (("all", "All items"), ("held_out", "Held out")):
        block = report.get(subset) or {}
        if not block:
            continue
        for rater, label in (("judge", "The judge"), ("token_rule", "The token rule")):
            values = block.get(rater) or {}
            if not values:
                continue
            rows.append(
                {
                    "subset": title,
                    "n_items": block.get("n_items", 0),
                    "rater": label,
                    "agreement": values.get("observed_agreement", 0.0),
                    "kappa": values.get("cohen_kappa", 0.0),
                    "band": values.get("interpretation", ""),
                }
            )
    return rows


def judge_settings(report: dict[str, Any]) -> dict[str, Any]:
    """How the judge was configured for the validation run."""
    return report.get("judge") or {}


def self_consistency(report: dict[str, Any]) -> dict[str, Any]:
    """How stable the judge was with itself.

    Three numbers, each answering a different question. Sample agreement is
    whether the judge's own samples reached the same verdict. Order
    consistency is whether it reached the same verdict when the answer and
    the reference were swapped. The score kappa is whether the 1 to 5
    scores lined up between the judge and the token rule's ranking.
    """
    settings = judge_settings(report)
    block = report.get("all") or {}
    return {
        "sample_agreement": settings.get("sample_agreement"),
        "order_consistency": settings.get("order_consistency"),
        "score_kappa": block.get("score_kappa"),
        "n_samples": settings.get("n_samples"),
        "swap_order": settings.get("swap_order"),
        "threshold": settings.get("threshold"),
        "judge_name": settings.get("name", ""),
    }


def mean_judge_scores(report: dict[str, Any]) -> dict[str, float]:
    """The judge's mean score on each rubric dimension, from 1 to 5."""
    scores = judge_settings(report).get("mean_scores") or {}
    return {
        dimension: float(value)
        for dimension, value in scores.items()
        if isinstance(value, (int, float))
    }


def judge_vs_token_row(report: dict[str, Any]) -> dict[str, Any] | None:
    """How often the judge and the token rule agree with each other."""
    block = report.get("all") or {}
    values = block.get("judge_vs_token_rule") or {}
    if not values:
        return None
    return {
        "n_items": block.get("n_items", 0),
        "agreement": values.get("observed_agreement", 0.0),
        "kappa": values.get("cohen_kappa", 0.0),
        "band": values.get("interpretation", ""),
    }


def metric_comparison(runs: list[RunSummary], key: str) -> dict[str, float]:
    """One metric across every run, for a chart.

    Runs that did not measure the metric are left out rather than shown as
    zero, which would look like a measurement of zero rather than of
    nothing.
    """
    comparison: dict[str, float] = {}
    for run in runs:
        value = run.metrics.get(key)
        if isinstance(value, (int, float)):
            comparison[run.run_id] = float(value)
    return comparison


def run_details(run: dict[str, Any]) -> list[tuple[str, str]]:
    """The facts about a run worth showing next to its metrics.

    Everything here comes out of the report, so the dashboard cannot
    describe a run differently from the way it was produced.
    """
    config = run.get("config", {}) or {}
    retrieval = config.get("retrieval", {}) or {}
    generation = config.get("generation", {}) or {}
    judge = config.get("judge", {}) or {}

    details = [
        ("Run id", str(run.get("run_id", ""))),
        ("Created", str(run.get("created_at", ""))),
        ("Dataset version", str(run.get("dataset_version", ""))),
        ("Dataset hash", str(run.get("dataset_hash", "")[:12])),
        ("Git commit", str(run.get("git_commit", "")[:8])),
        ("Retriever", str(retrieval.get("retriever", ""))),
        ("Top k", str(retrieval.get("top_k", ""))),
        ("Generator", str(generation.get("provider", ""))),
        ("Model", str(generation.get("model", ""))),
        ("Seed", str(generation.get("seed", ""))),
    ]
    if judge.get("enabled"):
        details.append(("Judge", f"{judge.get('provider')} ({judge.get('model')})"))
        details.append(("Judge threshold", str(judge.get("threshold", ""))))
    else:
        details.append(("Judge", "not enabled"))
    return details


def judge_disagreements(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The items where a rater disagreed with the hand labels."""
    rows: list[dict[str, Any]] = []
    for item in report.get("items", []):
        human = item.get("human_label", "")
        judge = item.get("judge_label", "")
        token = item.get("token_rule_label", "")
        if judge == human and token == human:
            continue
        rows.append(
            {
                "test_id": item.get("test_id"),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "human": human,
                "judge": judge,
                "token_rule": token,
                "judge_wrong": judge != human,
                "token_wrong": token != human,
                "used_for_tuning": bool(item.get("used_for_tuning")),
                "note": item.get("note", ""),
            }
        )
    return rows


def format_metric(value: Any, kind: str = "number") -> str:
    """Render a metric the way the dashboard shows it."""
    if value is None:
        return "—"
    if not isinstance(value, (int, float)):
        return str(value)
    if kind == "percent":
        return f"{value * 100:.1f}%"
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def judge_verdict_label(value: Any) -> str:
    """Render a verdict as a word rather than a boolean."""
    if value is None:
        return "—"
    return "correct" if value else "incorrect"

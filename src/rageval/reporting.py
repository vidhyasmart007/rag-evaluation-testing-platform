"""Writing evaluation reports to disk.

Each run produces a JSON file with the full detail and a CSV file with one
row per case, plus a ``latest`` copy of both. Reports are written per run
rather than appended to, so a report always describes exactly one
configuration.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rageval.config import EvalConfig
from rageval.schema import RunResult

_CSV_BASE_FIELDS = [
    "test_id",
    "question",
    "reference_answer",
    "answerable",
    "answer",
    "abstained",
    "retrieved_chunk_ids",
    "retrieved_sentence_ids",
    "top_score",
    "generator",
    "cached",
    "latency_ms",
    "error",
]


def _metric_names(run: RunResult) -> list[str]:
    names: set[str] = set()
    for case in run.cases:
        names.update(case.metrics)
    return sorted(names)


def write_reports(run: RunResult, config: EvalConfig) -> tuple[Path, Path]:
    """Write the JSON and CSV reports for a run and return their paths."""
    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / f"{run.run_id}.json"
    csv_path = reports_dir / f"{run.run_id}.csv"

    json_path.write_text(
        json.dumps(run.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    metric_names = _metric_names(run)
    fieldnames = _CSV_BASE_FIELDS + metric_names

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for case in run.cases:
            row = {
                "test_id": case.test_id,
                "question": case.question,
                "reference_answer": case.reference_answer,
                "answerable": case.answerable,
                "answer": case.answer,
                "abstained": case.abstained,
                "retrieved_chunk_ids": ";".join(str(c.chunk_id) for c in case.retrieved),
                "retrieved_sentence_ids": ";".join(
                    str(sid) for c in case.retrieved for sid in c.sentence_ids
                ),
                "top_score": max((c.score for c in case.retrieved), default=0.0),
                "generator": case.generator,
                "cached": case.cached,
                "latency_ms": case.latency_ms,
                "error": case.error or "",
            }
            for name in metric_names:
                row[name] = case.metrics.get(name, "")
            writer.writerow(row)

    (reports_dir / "latest.json").write_text(
        json.dumps(run.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    (reports_dir / "latest.csv").write_text(
        csv_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    return json_path, csv_path

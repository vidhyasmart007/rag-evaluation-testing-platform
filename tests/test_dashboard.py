"""Tests for the dashboard's reading logic.

The dashboard module deliberately holds no Streamlit code, so these tests
run without Streamlit installed and without a model server. They check the
one thing that matters about a dashboard: that it shows what the report
actually said.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rageval import dashboard

RUN = {
    "run_id": "example",
    "created_at": "2026-01-01T00:00:00",
    "dataset_version": "v2",
    "dataset_hash": "abcdef0123456789",
    "git_commit": "1234567890abcdef",
    "config": {
        "retrieval": {"retriever": "lexical", "top_k": 3},
        "generation": {"provider": "ollama", "model": "qwen3:1.7b", "seed": 42},
        "judge": {"enabled": True, "provider": "ollama", "model": "llama3.2:3b"},
    },
    "aggregates": {"total_cases": 2, "answer_correct": 0.5, "token_f1": 0.75},
    "cases": [
        {
            "test_id": "c1",
            "question": "Where is the office?",
            "reference_answer": "Bengaluru",
            "answerable": True,
            "answer": "Bengaluru",
            "abstained": False,
            "metrics": {"answer_correct": True, "token_f1": 1.0, "faithfulness": 1.0},
            "judge": {
                "correct": True,
                "correctness": 5,
                "relevance": 5,
                "groundedness": 5,
                "completeness": 4,
                "reason": "matches the reference",
            },
            "retrieved": [
                {
                    "chunk_id": 0,
                    "score": 0.75,
                    "text": "The office is in Bengaluru.",
                    "sentence_ids": [0],
                }
            ],
            "error": None,
        },
        {
            "test_id": "c2",
            "question": "What is the CEO's favourite colour?",
            "reference_answer": "",
            "answerable": False,
            "answer": "",
            "abstained": True,
            "metrics": {"answer_correct": True, "token_f1": 0.0, "faithfulness": 1.0},
            "judge": None,
            "retrieved": [],
            "error": None,
        },
    ],
}

VALIDATION = {
    "n_items": 3,
    "judge": {
        "name": "ollama:llama3.2:3b",
        "n_samples": 3,
        "swap_order": True,
        "threshold": 4,
        "mean_scores": {"correctness": 3.8, "relevance": 3.9},
        "sample_agreement": 0.98,
        "order_consistency": 0.97,
    },
    "all": {
        "n_items": 3,
        "judge": {
            "observed_agreement": 0.9,
            "cohen_kappa": 0.8,
            "interpretation": "almost perfect",
        },
        "token_rule": {
            "observed_agreement": 0.7,
            "cohen_kappa": 0.5,
            "interpretation": "substantial",
        },
        "judge_vs_token_rule": {
            "observed_agreement": 0.8,
            "cohen_kappa": 0.6,
            "interpretation": "substantial",
        },
        "score_kappa": 0.91,
    },
    "held_out": {
        "n_items": 2,
        "judge": {"observed_agreement": 1.0, "cohen_kappa": 1.0},
        "token_rule": {"observed_agreement": 0.5, "cohen_kappa": 0.0},
    },
    "items": [
        {
            "test_id": 3,
            "question": "Which language?",
            "answer": "Python",
            "human_label": "correct",
            "judge_label": "correct",
            "token_rule_label": "incorrect",
            "used_for_tuning": True,
            "note": "The token rule is too strict here.",
        },
        {
            "test_id": 4,
            "question": "Which city?",
            "answer": "Bengaluru",
            "human_label": "correct",
            "judge_label": "correct",
            "token_rule_label": "correct",
            "used_for_tuning": False,
            "note": "",
        },
    ],
}


@pytest.fixture()
def reports_dir(tmp_path: Path) -> Path:
    """A directory holding one run, a copy of it and a validation report."""
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "example.json").write_text(json.dumps(RUN), encoding="utf-8")
    (directory / "latest.json").write_text(json.dumps(RUN), encoding="utf-8")
    (directory / "judge_validation.json").write_text(
        json.dumps(VALIDATION), encoding="utf-8"
    )
    return directory


def test_the_run_list_leaves_out_the_copy_of_the_latest_run(reports_dir: Path):
    """latest.json is the same run under another name, not a run of its own."""
    runs = dashboard.discover_runs(reports_dir)

    assert [run.run_id for run in runs] == ["example"]


def test_the_copy_is_still_listed_when_it_is_the_only_copy(tmp_path: Path):
    """The deployed checkout has only latest.json, and it must still show.

    A checkout where the copy is the only file has to list the run it
    holds, or the dashboard opens empty on the public deployment.
    """
    directory = tmp_path / "deployed"
    directory.mkdir()
    (directory / "latest.json").write_text(json.dumps(RUN), encoding="utf-8")

    runs = dashboard.discover_runs(directory)

    assert [run.run_id for run in runs] == ["example"]


def test_two_runs_with_different_ids_are_both_listed(reports_dir: Path):
    """Deduplication is by run id, not by file name."""
    path = reports_dir / "example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "second"
    (reports_dir / "second.json").write_text(json.dumps(payload), encoding="utf-8")

    runs = dashboard.discover_runs(reports_dir)

    assert sorted(run.run_id for run in runs) == ["example", "second"]


def test_a_broken_report_does_not_empty_the_dashboard(reports_dir: Path):
    """One unreadable file should cost one run, not all of them."""
    (reports_dir / "broken.json").write_text("{not json", encoding="utf-8")

    runs = dashboard.discover_runs(reports_dir)

    assert [run.run_id for run in runs] == ["example"]


def test_a_file_that_is_not_a_report_is_skipped(reports_dir: Path):
    """A JSON file without a run id is not a run."""
    (reports_dir / "notes.json").write_text(json.dumps({"note": "hi"}), encoding="utf-8")

    runs = dashboard.discover_runs(reports_dir)

    assert [run.run_id for run in runs] == ["example"]


def test_a_missing_reports_directory_is_not_an_error(tmp_path: Path):
    """The dashboard has to work in a checkout with no reports in it."""
    assert dashboard.discover_runs(tmp_path / "nowhere") == []


def test_a_run_is_described_by_its_configuration(reports_dir: Path):
    """The summary says what produced the run, not just its name."""
    run = dashboard.discover_runs(reports_dir)[0]

    assert run.retriever == "lexical"
    assert run.generator == "ollama (qwen3:1.7b)"
    assert run.judge == "ollama (llama3.2:3b)"
    assert run.n_cases == 2
    assert run.label == "lexical · ollama (qwen3:1.7b) · judged by ollama (llama3.2:3b)"


def test_a_run_without_a_judge_does_not_claim_one(reports_dir: Path):
    """Saying nothing is better than saying "judged by" nothing."""
    path = reports_dir / "example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["judge"]["enabled"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    run = dashboard.discover_runs(reports_dir)[0]

    assert run.judge == ""
    assert "judged by" not in run.label


def test_the_case_table_reads_the_numbers_from_the_report(reports_dir: Path):
    """The table must not disagree with the report it came from."""
    report = dashboard.load_run(reports_dir / "example.json")

    rows = dashboard.case_rows(report)

    assert [row["id"] for row in rows] == ["c1", "c2"]
    assert rows[0]["token_correct"] is True
    assert rows[0]["judge_correct"] is True
    assert rows[0]["judge_correctness"] == 5
    assert rows[0]["judge_reason"] == "matches the reference"
    assert rows[0]["top_score"] == 0.75


def test_a_case_that_was_not_judged_is_not_given_a_verdict(reports_dir: Path):
    """None means the judge never reached a verdict, not that it said no."""
    report = dashboard.load_run(reports_dir / "example.json")

    rows = dashboard.case_rows(report)

    assert rows[1]["judge_correct"] is None
    assert dashboard.judge_verdict_label(rows[1]["judge_correct"]) == "—"


def test_the_validation_report_is_optional(reports_dir: Path, tmp_path: Path):
    """The dashboard has to work without a validation run."""
    assert dashboard.load_judge_validation(reports_dir) is not None
    assert dashboard.load_judge_validation(tmp_path / "empty") is None


def test_a_broken_validation_report_is_treated_as_missing(reports_dir: Path):
    """A report that cannot be read is no better than one that is absent."""
    (reports_dir / "judge_validation.json").write_text("{", encoding="utf-8")

    assert dashboard.load_judge_validation(reports_dir) is None


def test_the_kappa_table_has_one_row_per_rater_and_subset(reports_dir: Path):
    """A table can be drawn from the rows without knowing the report shape."""
    report = dashboard.load_judge_validation(reports_dir)

    rows = dashboard.kappa_rows(report)

    assert [(row["subset"], row["rater"]) for row in rows] == [
        ("All items", "The judge"),
        ("All items", "The token rule"),
        ("Held out", "The judge"),
        ("Held out", "The token rule"),
    ]
    assert rows[0]["kappa"] == 0.8
    assert rows[0]["n_items"] == 3


def test_the_judge_reports_how_stable_it_was_with_itself(reports_dir: Path):
    """The three self-consistency numbers answer three different questions."""
    report = dashboard.load_judge_validation(reports_dir)

    consistency = dashboard.self_consistency(report)

    assert consistency["sample_agreement"] == 0.98
    assert consistency["order_consistency"] == 0.97
    assert consistency["score_kappa"] == 0.91
    assert consistency["judge_name"] == "ollama:llama3.2:3b"


def test_the_mean_rubric_scores_are_reported_as_numbers(reports_dir: Path):
    """They are read from the report, so they arrive as numbers already."""
    report = dashboard.load_judge_validation(reports_dir)

    scores = dashboard.mean_judge_scores(report)

    assert scores == {"correctness": 3.8, "relevance": 3.9}


def test_only_the_disagreements_are_listed(reports_dir: Path):
    """An item both raters got right is not a disagreement."""
    report = dashboard.load_judge_validation(reports_dir)

    rows = dashboard.judge_disagreements(report)

    assert len(rows) == 1
    assert rows[0]["test_id"] == 3
    assert rows[0]["token_wrong"] is True
    assert rows[0]["judge_wrong"] is False
    assert rows[0]["used_for_tuning"] is True


def test_a_metric_is_compared_across_the_runs_that_measured_it(reports_dir: Path):
    """A run that did not measure a metric is left out, not shown as zero."""
    path = reports_dir / "example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "second"
    payload["aggregates"].pop("token_f1")
    (reports_dir / "second.json").write_text(json.dumps(payload), encoding="utf-8")

    runs = dashboard.discover_runs(reports_dir)
    comparison = dashboard.metric_comparison(runs, "token_f1")

    assert comparison == {"example": 0.75}


def test_a_run_is_described_by_the_facts_about_it(reports_dir: Path):
    """The details shown next to the metrics come from the report."""
    report = dashboard.load_run(reports_dir / "example.json")

    details = dict(dashboard.run_details(report))

    assert details["Retriever"] == "lexical"
    assert details["Model"] == "qwen3:1.7b"
    assert details["Judge"] == "ollama (llama3.2:3b)"


def test_a_percentage_is_shown_as_a_percentage():
    """A share of 0.9524 is ninety five percent, not 0.9524."""
    assert dashboard.format_metric(0.9524, "percent") == "95.2%"


def test_a_plain_number_is_shown_without_a_rounding_tail():
    """0.8 stays 0.8 and does not become 0.8000."""
    assert dashboard.format_metric(0.8) == "0.8"
    assert dashboard.format_metric(1.0) == "1"


def test_a_missing_metric_is_shown_as_a_dash():
    """A missing value is not a zero."""
    assert dashboard.format_metric(None) == "—"

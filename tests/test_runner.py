"""End to end tests for the evaluation runner."""

from __future__ import annotations

import csv
import json

import pytest

from rageval.config import EvalConfig
from rageval.runner import aggregate, build_generator, build_retriever, run_evaluation


def test_a_run_produces_a_result_for_every_case(config):
    run = run_evaluation(config)

    assert len(run.cases) == 3
    assert run.run_id == "test_run"
    assert run.dataset_version == "2.0.0"
    assert run.aggregates["total_cases"] == 3.0


def test_a_run_writes_a_json_report_and_a_csv_report(config):
    run = run_evaluation(config)

    json_report = config.reports_dir / "test_run.json"
    csv_report = config.reports_dir / "test_run.csv"
    assert json_report.exists()
    assert csv_report.exists()

    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert payload["run_id"] == run.run_id
    assert len(payload["cases"]) == 3

    with csv_report.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    assert rows[0]["question"] == "Where is the company office?"


def test_the_latest_report_is_refreshed(config):
    run_evaluation(config)
    payload = json.loads((config.reports_dir / "latest.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "test_run"


def test_the_report_records_the_configuration_it_ran_with(config):
    run = run_evaluation(config)
    assert run.config["retrieval"]["top_k"] == config.retrieval.top_k
    assert run.config["generation"]["provider"] == "extractive"


def test_answerable_cases_are_scored_against_their_reference(config):
    run = run_evaluation(config)
    scored = {case.test_id: case for case in run.cases}

    assert scored[1].metrics["answer_correct"] == 1.0
    assert scored[2].metrics["numeric_match"] == 1.0
    assert scored[1].metrics["hit_at_k"] == 1.0


def test_an_unanswerable_case_is_expected_to_abstain(config):
    run = run_evaluation(config)
    case = next(case for case in run.cases if case.test_id == 3)
    assert case.abstained
    assert case.metrics["abstention_correct"] == 1.0


def test_metrics_that_do_not_apply_are_left_out(config):
    """An unanswerable case has no reference, so it has no correctness score."""
    run = run_evaluation(config)
    case = next(case for case in run.cases if case.test_id == 3)
    assert "answer_correct" not in case.metrics
    assert "token_f1" not in case.metrics


def test_aggregates_average_only_the_cases_that_were_measured(config):
    run = run_evaluation(config)
    # Two answerable cases, one of which is unanswerable at the answer level.
    assert run.aggregates["answerable_cases"] == 2.0
    assert run.aggregates["unanswerable_cases"] == 1.0


def test_aggregate_of_an_empty_run_is_empty_but_safe():
    aggregates = aggregate([])
    assert aggregates["total_cases"] == 0.0
    assert aggregates["error_count"] == 0.0


def test_a_missing_knowledge_base_is_reported(tmp_path, dataset_path):
    config = EvalConfig(
        dataset_path=dataset_path,
        knowledge_base_path=tmp_path / "absent.txt",
        reports_dir=tmp_path / "reports",
    )
    with pytest.raises(FileNotFoundError):
        run_evaluation(config)


def test_an_unknown_retriever_is_rejected(config):
    config.retrieval.retriever = "magic"
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_retriever(config, "some document")


def test_an_unknown_generator_is_rejected(config):
    config.generation.provider = "magic"
    with pytest.raises(ValueError, match="Unknown generation provider"):
        build_generator(config)


class FakeEmbedder:
    """Stands in for the model server, so no test needs Ollama running."""

    name = "fake"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_the_dense_retriever_can_be_built(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())
    config.retrieval.retriever = "dense"

    retriever = build_retriever(config, "The office is in Bengaluru.")

    assert retriever.name == "dense"


def test_the_hybrid_retriever_can_be_built(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())
    config.retrieval.retriever = "hybrid"

    retriever = build_retriever(config, "The office is in Bengaluru.")

    assert retriever.name == "hybrid"


def test_the_hybrid_weights_reach_the_retriever(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())
    config.retrieval.retriever = "hybrid"
    config.retrieval.lexical_weight = 0.7
    config.retrieval.dense_weight = 0.3

    retriever = build_retriever(config, "The office is in Bengaluru.")

    assert retriever.lexical_weight == pytest.approx(0.7)
    assert retriever.dense_weight == pytest.approx(0.3)


def test_a_dense_run_scores_the_dataset(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())
    config.retrieval.retriever = "dense"

    run = run_evaluation(config)

    assert run.aggregates["total_cases"] == 3.0
    assert run.aggregates["error_count"] == 0.0


def test_a_run_records_the_precision_of_the_context(config):
    run = run_evaluation(config)
    case = next(case for case in run.cases if case.test_id == 1)

    assert "precision_at_k" in case.metrics


def test_a_run_records_whether_the_retriever_refused(config):
    run = run_evaluation(config)
    case = next(case for case in run.cases if case.test_id == 3)

    assert case.metrics["retrieval_rejected"] in (0.0, 1.0)


def test_the_aggregate_reports_the_rejection_rate(config):
    run = run_evaluation(config)

    assert "unanswerable_rejection_rate" in run.aggregates
    assert "answerable_rejection_rate" in run.aggregates

"""Tests for the retriever comparison tool."""

from __future__ import annotations

from rageval.compare import METRICS, RETRIEVERS, compare, render


class FakeEmbedder:
    """Stands in for the model server, so no test needs Ollama running."""

    name = "fake"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_it_runs_every_retriever(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())

    results = compare(config)

    assert set(results) == set(RETRIEVERS)
    assert all(result["aggregates"]["total_cases"] == 3.0 for result in results.values())


def test_each_retriever_uses_its_own_relevance_floor(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())

    results = compare(config)

    assert results["lexical"]["min_score"] == 0.30
    assert results["dense"]["min_score"] == 0.35
    assert results["hybrid"]["min_score"] == 0.40


def test_an_explicit_floor_overrides_every_retriever(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())
    config.retrieval.min_score = 0.5

    results = compare(config)

    assert all(result["min_score"] == 0.5 for result in results.values())


def test_the_comparison_does_not_write_to_the_reports_directory(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())

    compare(config)

    assert not (config.reports_dir / "latest.json").exists()


def test_the_table_has_a_row_for_every_metric(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())

    table = render(compare(config))

    for _, label in METRICS:
        assert label in table
    for name in RETRIEVERS:
        assert name in table


def test_the_score_based_relevance_is_left_out_of_the_comparison():
    """Its scale differs per retriever, so comparing it would be misleading."""
    assert "retrieval_relevance" not in {key for key, _ in METRICS}

"""Tests for the comparison tool."""

from __future__ import annotations

import pytest

from rageval.compare import (
    GENERATOR_METRICS,
    RETRIEVER_METRICS,
    RETRIEVERS,
    compare_generators,
    compare_retrievers,
    render,
)


class FakeEmbedder:
    """Stands in for the model server, so no test needs Ollama running."""

    name = "fake"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeGenerator:
    """Stands in for the language model, so no test needs Ollama running."""

    name = "ollama:fake"

    def __init__(self, model: str, **kwargs) -> None:
        self.model = model
        self.cache_key: dict = {}

    def generate(self, question: str, context: str):
        from rageval.generation.base import GenerationResult

        return GenerationResult(text="Bengaluru.", model=self.name)


def test_it_runs_every_retriever(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())

    results = compare_retrievers(config)

    assert set(results) == set(RETRIEVERS)
    assert all(result["aggregates"]["total_cases"] == 3.0 for result in results.values())


def test_each_retriever_uses_its_own_relevance_floor(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())

    results = compare_retrievers(config)

    assert results["lexical"]["min_score"] == 0.30
    assert results["dense"]["min_score"] == 0.35
    assert results["hybrid"]["min_score"] == 0.40


def test_an_explicit_floor_overrides_every_retriever(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())
    config.retrieval.min_score = 0.5

    results = compare_retrievers(config)

    assert all(result["min_score"] == 0.5 for result in results.values())


def test_the_comparison_does_not_write_to_the_reports_directory(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())

    compare_retrievers(config)

    assert not (config.reports_dir / "latest.json").exists()


def test_the_table_has_a_row_for_every_metric(config, monkeypatch):
    monkeypatch.setattr("rageval.runner.build_embedder", lambda config: FakeEmbedder())

    table = render(compare_retrievers(config), RETRIEVER_METRICS, RETRIEVERS)

    for _, label in RETRIEVER_METRICS:
        assert label in table
    for name in RETRIEVERS:
        assert name in table


def test_the_score_based_relevance_is_left_out_of_the_comparison():
    """Its scale differs per retriever, so comparing it would be misleading."""
    assert "retrieval_relevance" not in {key for key, _ in RETRIEVER_METRICS}


def test_the_generator_comparison_needs_a_model(config):
    with pytest.raises(ValueError, match="language model"):
        compare_generators(config)


def test_it_compares_the_baseline_against_the_model(config, monkeypatch):
    monkeypatch.setattr(
        "rageval.runner.build_generator", lambda config: FakeGenerator("m")
    )
    config.generation.provider = "ollama"
    config.generation.model = "qwen3:1.7b"

    results = compare_generators(config)

    assert set(results) == {"extractive", "llm"}
    assert results["llm"]["generator"] == "qwen3:1.7b"
    assert results["extractive"]["generator"] == "extractive"


def test_the_generator_comparison_runs_without_the_cache(config, monkeypatch):
    """A cached answer is a lookup, and a lookup would flatter the model."""
    monkeypatch.setattr(
        "rageval.runner.build_generator", lambda config: FakeGenerator("m")
    )
    config.generation.provider = "ollama"
    config.generation.model = "qwen3:1.7b"

    results = compare_generators(config)

    assert results["llm"]["aggregates"]["cache_hits"] == 0.0


def test_the_generator_table_has_a_row_for_every_metric(config, monkeypatch):
    monkeypatch.setattr(
        "rageval.runner.build_generator", lambda config: FakeGenerator("m")
    )
    config.generation.provider = "ollama"
    config.generation.model = "qwen3:1.7b"

    table = render(
        compare_generators(config), GENERATOR_METRICS, ("extractive", "llm")
    )

    for _, label in GENERATOR_METRICS:
        assert label in table


def test_the_generator_comparison_leaves_out_the_retrieval_metrics():
    """Both arms retrieve the same way, so those numbers would be identical."""
    keys = {key for key, _ in GENERATOR_METRICS}
    assert "hit_at_k" not in keys
    assert "precision_at_k" not in keys

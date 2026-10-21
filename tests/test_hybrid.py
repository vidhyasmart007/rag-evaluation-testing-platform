"""Tests for the hybrid retriever."""

from __future__ import annotations

import pytest

from rageval.retrieval.hybrid import HybridRetriever
from rageval.retrieval.lexical import Chunk


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=0,
            text="The office is in Bengaluru.",
            sentence_ids=(0,),
            word_count=5,
        ),
        Chunk(
            chunk_id=1,
            text="Employees get 18 leave days.",
            sentence_ids=(1,),
            word_count=5,
        ),
        Chunk(chunk_id=2, text="Backups run nightly.", sentence_ids=(2,), word_count=3),
    ]


class FakeLexical:
    """Returns a fixed coverage per chunk."""

    name = "lexical"

    def __init__(self, coverages: list[float]) -> None:
        self.coverages = coverages

    def score(self, question):
        return [
            (index, 1.0, coverage) for index, coverage in enumerate(self.coverages)
        ]


class FakeDense:
    """Returns a fixed similarity per chunk."""

    name = "dense"

    def __init__(self, similarities: list[float]) -> None:
        self.similarities = similarities

    def score(self, question):
        return list(enumerate(self.similarities))


def test_it_blends_the_two_scores():
    retriever = HybridRetriever(
        make_chunks(),
        FakeLexical([1.0, 0.0, 0.0]),
        FakeDense([0.0, 0.8, 0.0]),
    )

    scores = dict(retriever.score("anything"))

    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(0.4)
    assert scores[2] == 0.0


def test_the_weights_are_normalised():
    retriever = HybridRetriever(
        make_chunks(),
        FakeLexical([1.0, 0.0, 0.0]),
        FakeDense([0.0, 0.0, 0.0]),
        lexical_weight=3.0,
        dense_weight=1.0,
    )

    scores = dict(retriever.score("anything"))

    assert scores[0] == pytest.approx(0.75)


def test_a_full_weight_on_one_side_ignores_the_other():
    retriever = HybridRetriever(
        make_chunks(),
        FakeLexical([1.0, 0.0, 0.0]),
        FakeDense([0.0, 0.0, 0.0]),
        lexical_weight=1.0,
        dense_weight=0.0,
    )

    scores = dict(retriever.score("anything"))

    assert scores == {0: pytest.approx(1.0), 1: 0.0, 2: 0.0}


def test_it_ranks_by_the_blend_rather_than_by_the_lexical_rank():
    retriever = HybridRetriever(
        make_chunks(),
        FakeLexical([0.8, 0.0, 0.0]),
        FakeDense([0.0, 1.0, 0.0]),
    )

    results = retriever.retrieve("anything", top_k=2)

    assert results[0].chunk_id == 1


def test_it_drops_chunks_below_the_relevance_floor():
    retriever = HybridRetriever(
        make_chunks(),
        FakeLexical([1.0, 0.0, 0.0]),
        FakeDense([0.0, 0.2, 0.0]),
    )

    results = retriever.retrieve("anything", top_k=3, min_score=0.3)

    assert [chunk.chunk_id for chunk in results] == [0]


def test_a_chunk_missing_from_one_side_contributes_zero():
    class PartialLexical:
        name = "lexical"

        def score(self, question):
            return [(0, 1.0, 1.0)]

    class PartialDense:
        name = "dense"

        def score(self, question):
            return [(2, 0.5)]

    retriever = HybridRetriever(make_chunks(), PartialLexical(), PartialDense())

    scores = dict(retriever.score("anything"))

    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == 0.0
    assert scores[2] == pytest.approx(0.25)


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        HybridRetriever(
            make_chunks(), FakeLexical([]), FakeDense([]), lexical_weight=-1.0
        )


def test_zero_weights_are_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        HybridRetriever(
            make_chunks(),
            FakeLexical([]),
            FakeDense([]),
            lexical_weight=0.0,
            dense_weight=0.0,
        )


def test_a_non_positive_top_k_is_rejected():
    retriever = HybridRetriever(
        make_chunks(), FakeLexical([1.0, 0.0, 0.0]), FakeDense([0.0, 0.0, 0.0])
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.retrieve("anything", top_k=0)


def test_it_returns_at_most_top_k():
    retriever = HybridRetriever(
        make_chunks(),
        FakeLexical([1.0, 0.9, 0.8]),
        FakeDense([1.0, 0.9, 0.8]),
    )

    assert len(retriever.retrieve("anything", top_k=2)) == 2


def test_the_retrieved_chunk_carries_its_sentence_ids():
    retriever = HybridRetriever(
        make_chunks(), FakeLexical([1.0, 0.0, 0.0]), FakeDense([0.5, 0.0, 0.0])
    )

    results = retriever.retrieve("anything", top_k=1)

    assert results[0].chunk_id == 0
    assert results[0].sentence_ids == (0,)

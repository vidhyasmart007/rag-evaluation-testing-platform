"""Tests for the dense retriever.

A fake embedder stands in for the model server, so the suite runs on a
machine where Ollama is not installed.
"""

from __future__ import annotations

import pytest

from rageval.retrieval.dense import DenseRetriever, clamp_similarity, cosine_similarity
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


class FakeEmbedder:
    """Embeds text as a one-hot vector, keyed by an exact match."""

    name = "fake"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts):
        items = list(texts)
        self.calls.append(items)
        return [self.vectors.get(text, [0.5, 0.5]) for text in items]


def test_cosine_similarity_of_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_ignores_the_magnitude():
    assert cosine_similarity([1.0, 0.0], [7.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_a_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_rejects_mismatched_sizes():
    with pytest.raises(ValueError, match="different sizes"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_the_similarity_is_clamped_to_the_unit_interval():
    assert clamp_similarity(-0.4) == 0.0
    assert clamp_similarity(0.4) == pytest.approx(0.4)
    assert clamp_similarity(1.4) == 1.0


def test_a_negative_similarity_is_reported_as_no_similarity():
    chunks = make_chunks()
    embedder = FakeEmbedder(
        {
            "The office is in Bengaluru.": [1.0, 0.0],
            "Employees get 18 leave days.": [0.0, 1.0],
            "Backups run nightly.": [0.0, 0.0],
        }
    )
    retriever = DenseRetriever(chunks, embedder)

    scores = dict(retriever.score("backups"))

    assert scores[2] == 0.0


def test_it_ranks_the_closest_chunk_first():
    chunks = make_chunks()
    embedder = FakeEmbedder(
        {
            "The office is in Bengaluru.": [1.0, 0.0],
            "Employees get 18 leave days.": [0.0, 1.0],
            "Backups run nightly.": [0.0, 0.9],
            "where is the office": [1.0, 0.05],
        }
    )
    retriever = DenseRetriever(chunks, embedder)

    results = retriever.retrieve("where is the office", top_k=2)

    assert results[0].chunk_id == 0
    assert results[0].score > results[1].score


def test_it_returns_at_most_top_k():
    chunks = make_chunks()
    embedder = FakeEmbedder(
        {
            "The office is in Bengaluru.": [1.0, 0.0],
            "Employees get 18 leave days.": [0.9, 0.1],
            "Backups run nightly.": [0.8, 0.2],
            "anything": [0.5, 0.5],
        }
    )
    retriever = DenseRetriever(chunks, embedder)

    assert len(retriever.retrieve("anything", top_k=2)) == 2


def test_it_drops_chunks_below_the_relevance_floor():
    chunks = make_chunks()
    embedder = FakeEmbedder(
        {
            "The office is in Bengaluru.": [1.0, 0.0],
            "Employees get 18 leave days.": [0.0, 1.0],
            "Backups run nightly.": [0.0, 0.9],
            "where is the office": [1.0, 0.05],
        }
    )
    retriever = DenseRetriever(chunks, embedder)

    results = retriever.retrieve("where is the office", top_k=3, min_score=0.9)

    assert [chunk.chunk_id for chunk in results] == [0]


def test_the_floor_can_reject_everything():
    chunks = make_chunks()
    embedder = FakeEmbedder(
        {
            "The office is in Bengaluru.": [1.0, 0.0],
            "Employees get 18 leave days.": [0.0, 1.0],
            "Backups run nightly.": [0.0, 0.9],
            "anything": [0.5, 0.5],
        }
    )
    retriever = DenseRetriever(chunks, embedder)

    assert retriever.retrieve("anything", top_k=3, min_score=1.0) == []


def test_a_non_positive_top_k_is_rejected():
    chunks = make_chunks()
    embedder = FakeEmbedder({chunk.text: [1.0, 0.0] for chunk in chunks})
    retriever = DenseRetriever(chunks, embedder)

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.retrieve("anything", top_k=0)


def test_it_embeds_every_chunk_once():
    chunks = make_chunks()
    embedder = FakeEmbedder({chunk.text: [1.0, 0.0] for chunk in chunks})
    DenseRetriever(chunks, embedder)

    assert embedder.calls == [[chunk.text for chunk in chunks]]


def test_it_embeds_the_question_on_every_query():
    chunks = make_chunks()
    embedder = FakeEmbedder({chunk.text: [1.0, 0.0] for chunk in chunks})
    retriever = DenseRetriever(chunks, embedder)

    retriever.score("a question")

    assert embedder.calls[-1] == ["a question"]


def test_the_query_prefix_is_applied_to_the_question():
    chunks = make_chunks()
    embedder = FakeEmbedder({chunk.text: [1.0, 0.0] for chunk in chunks})
    retriever = DenseRetriever(chunks, embedder, query_prefix="query: ")

    retriever.score("a question")

    assert embedder.calls[-1] == ["query: a question"]


def test_the_document_prefix_is_applied_to_the_chunks():
    chunks = make_chunks()
    embedder = FakeEmbedder(
        {f"passage: {chunk.text}": [1.0, 0.0] for chunk in chunks}
    )
    DenseRetriever(chunks, embedder, document_prefix="passage: ")

    assert embedder.calls[0] == [f"passage: {chunk.text}" for chunk in chunks]


def test_a_wrong_number_of_vectors_is_rejected():
    class ShortEmbedder:
        name = "short"

        def embed(self, texts):
            return [[1.0]]

    with pytest.raises(ValueError, match="3 chunks"):
        DenseRetriever(make_chunks(), ShortEmbedder())


def test_the_retrieved_chunk_carries_its_sentence_ids():
    chunks = make_chunks()
    embedder = FakeEmbedder({chunk.text: [1.0, 0.0] for chunk in chunks})
    retriever = DenseRetriever(chunks, embedder)

    results = retriever.retrieve("anything", top_k=1)

    assert results[0].sentence_ids == (0,)
    assert results[0].text == chunks[0].text

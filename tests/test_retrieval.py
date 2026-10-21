"""Tests for chunking and lexical retrieval."""

from __future__ import annotations

import pytest

from rageval.retrieval.lexical import (
    BM25Retriever,
    build_chunks,
    split_into_sentences,
)

SENTENCES = [
    "The company office is located in Bengaluru, Karnataka.",
    "The company uses Python as its primary programming language.",
    "Employees receive 18 annual leave days per calendar year.",
    "The evaluation suite is executed with pytest.",
]


def test_a_heading_stays_its_own_sentence():
    text = "Company Knowledge Base\n\nThe office is in Bengaluru, Karnataka."
    assert split_into_sentences(text) == [
        "Company Knowledge Base",
        "The office is in Bengaluru, Karnataka.",
    ]


def test_paragraphs_are_split_into_separate_sentences():
    text = "First point. Second point.\n\nThird point."
    assert split_into_sentences(text) == [
        "First point.",
        "Second point.",
        "Third point.",
    ]


def test_a_chunk_is_always_a_whole_number_of_sentences():
    chunks = build_chunks(SENTENCES, chunk_size=12, overlap=1)
    for chunk in chunks:
        assert chunk.text == " ".join(SENTENCES[index] for index in chunk.sentence_ids)


def test_every_sentence_is_covered_at_least_once():
    chunks = build_chunks(SENTENCES, chunk_size=12, overlap=1)
    covered = {index for chunk in chunks for index in chunk.sentence_ids}
    assert covered == set(range(len(SENTENCES)))


def test_overlap_repeats_the_last_sentence_of_the_previous_chunk():
    chunks = build_chunks(
        ["a b c d e", "f g h i j", "k l m n o"],
        chunk_size=8,
        overlap=1,
    )
    assert [chunk.sentence_ids for chunk in chunks] == [(0,), (0, 1), (1, 2)]


def test_zero_overlap_produces_disjoint_chunks():
    chunks = build_chunks(SENTENCES, chunk_size=12, overlap=0)
    seen: set[int] = set()
    for chunk in chunks:
        assert not seen & set(chunk.sentence_ids)
        seen |= set(chunk.sentence_ids)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        build_chunks(SENTENCES, chunk_size=0)


def test_overlap_must_not_be_negative():
    with pytest.raises(ValueError):
        build_chunks(SENTENCES, overlap=-1)


def test_the_matching_chunk_ranks_first():
    retriever = BM25Retriever(build_chunks(SENTENCES, chunk_size=12, overlap=0))
    results = retriever.retrieve("How many annual leave days do employees get?", top_k=4)

    assert len(results) == 4
    assert results[0].score == pytest.approx(1.0)
    assert 2 in results[0].sentence_ids
    assert results[0].score > results[-1].score


def test_a_question_the_corpus_cannot_answer_scores_zero():
    retriever = BM25Retriever(build_chunks(SENTENCES, chunk_size=100, overlap=0))
    results = retriever.retrieve("What is the capital of Mars?", top_k=4)
    assert all(chunk.score == pytest.approx(0.0) for chunk in results)


def test_the_relevance_floor_drops_unrelated_chunks():
    retriever = BM25Retriever(build_chunks(SENTENCES, chunk_size=100, overlap=0))
    found = retriever.retrieve("What is the capital of Mars?", top_k=4, min_score=0.15)
    assert found == []


def test_top_k_limits_the_number_of_chunks():
    retriever = BM25Retriever(build_chunks(SENTENCES, chunk_size=100, overlap=0))
    results = retriever.retrieve("What does the company use?", top_k=1)
    assert len(results) == 1


def test_top_k_must_be_positive():
    retriever = BM25Retriever(build_chunks(SENTENCES, chunk_size=100, overlap=0))
    with pytest.raises(ValueError):
        retriever.retrieve("anything", top_k=0)


def test_scores_stay_inside_the_unit_interval():
    retriever = BM25Retriever(build_chunks(SENTENCES, chunk_size=100, overlap=0))
    for chunk in retriever.retrieve("Which language does the company use?", top_k=4):
        assert 0.0 <= chunk.score <= 1.0

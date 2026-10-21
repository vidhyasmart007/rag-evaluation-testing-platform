"""Tests for the scoring functions."""

from __future__ import annotations

import pytest

from rageval.metrics.answer import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
    exact_match,
    numeric_match,
    token_f1,
    token_recall,
)
from rageval.metrics.retrieval import (
    evaluate_retrieval,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rageval.schema import RetrievedChunk


def chunk(
    chunk_id: int,
    sentence_ids: tuple[int, ...],
    score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text="", score=score, sentence_ids=sentence_ids
    )


def test_token_f1_is_one_for_an_identical_answer():
    assert token_f1("Bengaluru", "Bengaluru") == pytest.approx(1.0)


def test_token_f1_ignores_stop_words_and_stems():
    assert token_f1("The office is in Bengaluru", "office bengaluru") == pytest.approx(
        1.0
    )


def test_token_f1_is_zero_when_nothing_is_shared():
    assert token_f1("Paris", "Bengaluru") == 0.0


def test_token_recall_rewards_a_verbose_answer_containing_the_reference():
    answer = "The company office is located in Bengaluru, Karnataka."
    assert token_recall(answer, "Bengaluru, Karnataka.") == pytest.approx(1.0)
    assert token_f1(answer, "Bengaluru, Karnataka.") < 1.0


def test_exact_match_requires_the_same_tokens():
    assert exact_match("Bengaluru", "Bengaluru") == 1.0
    assert exact_match("Bengaluru, Karnataka", "Bengaluru") == 0.0


def test_numeric_match_is_not_applicable_without_a_number_in_the_reference():
    assert numeric_match("Bengaluru", "Bengaluru") is None


def test_numeric_match_catches_the_wrong_number():
    assert numeric_match("Employees receive 18 days", "Employees receive 18 days") == 1.0
    assert numeric_match("Employees receive 20 days", "Employees receive 18 days") == 0.0


def test_faithfulness_is_one_when_the_answer_cites_the_context():
    context = "Employees receive 18 annual leave days per calendar year."
    assert evaluate_faithfulness(context, context) == pytest.approx(1.0)


def test_faithfulness_drops_when_the_answer_invents_content():
    context = "Employees receive 18 annual leave days per calendar year."
    assert evaluate_faithfulness("The capital of France is Paris", context) < 0.5


def test_answer_relevance_measures_the_overlap_with_the_question():
    relevance = evaluate_answer_relevance(
        "Where is the office?", "The office is in Bengaluru"
    )
    assert relevance == pytest.approx(1.0)
    assert evaluate_answer_relevance("Where is the office?", "Unrelated content") < 0.5


def test_retrieval_relevance_is_the_best_chunk_score():
    chunks = [chunk(0, (1,), 0.4), chunk(1, (2,), 0.9)]
    assert evaluate_retrieval(chunks) == pytest.approx(0.9)
    assert evaluate_retrieval([]) == 0.0


def test_hit_at_k_finds_a_relevant_sentence():
    assert hit_at_k([chunk(0, (1, 2)), chunk(1, (3,))], [2]) == 1.0
    assert hit_at_k([chunk(0, (1,))], [9]) == 0.0


def test_hit_at_k_is_zero_without_labels():
    assert hit_at_k([chunk(0, (1,))], []) == 0.0


def test_recall_at_k_counts_distinct_relevant_sentences():
    chunks = [chunk(0, (1, 2)), chunk(1, (2,))]
    assert recall_at_k(chunks, [1, 2, 3]) == pytest.approx(2 / 3)


def test_reciprocal_rank_rewards_earlier_hits():
    assert reciprocal_rank([chunk(0, (9,)), chunk(1, (2,))], [2]) == pytest.approx(0.5)
    assert reciprocal_rank([chunk(0, (2,))], [2]) == pytest.approx(1.0)
    assert reciprocal_rank([chunk(0, (9,))], [2]) == 0.0


def test_precision_at_k_counts_the_useful_chunks():
    chunks = [chunk(0, (1,)), chunk(1, (9,)), chunk(2, (2,))]
    assert precision_at_k(chunks, [1]) == pytest.approx(1 / 3)


def test_precision_at_k_is_one_when_every_chunk_is_useful():
    chunks = [chunk(0, (1,)), chunk(1, (2,))]
    assert precision_at_k(chunks, [1, 2]) == pytest.approx(1.0)


def test_precision_at_k_is_zero_without_retrieved_chunks():
    assert precision_at_k([], [1]) == 0.0


def test_precision_at_k_is_zero_without_labels():
    assert precision_at_k([chunk(0, (1,))], []) == 0.0

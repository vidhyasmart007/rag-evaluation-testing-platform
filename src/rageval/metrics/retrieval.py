"""Retrieval scoring.

Retrieval can only be measured against labels. The dataset records which
sentences of the source document answer a question, and these measures turn
those labels into numbers.
"""

from __future__ import annotations

from collections.abc import Sequence

from rageval.schema import RetrievedChunk


def evaluate_retrieval(retrieved_chunks: Sequence[RetrievedChunk]) -> float:
    """Best relevance score among the retrieved chunks."""
    if not retrieved_chunks:
        return 0.0
    return max(chunk.score for chunk in retrieved_chunks)


def hit_at_k(
    retrieved_chunks: Sequence[RetrievedChunk],
    relevant_sentence_ids: Sequence[int],
) -> float:
    """1.0 when at least one retrieved chunk contains a relevant sentence."""
    if not relevant_sentence_ids:
        return 0.0
    relevant = set(relevant_sentence_ids)
    for chunk in retrieved_chunks:
        if relevant & set(chunk.sentence_ids):
            return 1.0
    return 0.0


def recall_at_k(
    retrieved_chunks: Sequence[RetrievedChunk],
    relevant_sentence_ids: Sequence[int],
) -> float:
    """Share of the relevant sentences that appear in the retrieved chunks."""
    if not relevant_sentence_ids:
        return 0.0
    relevant = set(relevant_sentence_ids)
    covered: set[int] = set()
    for chunk in retrieved_chunks:
        covered |= relevant & set(chunk.sentence_ids)
    return len(covered) / len(relevant)


def reciprocal_rank(
    retrieved_chunks: Sequence[RetrievedChunk],
    relevant_sentence_ids: Sequence[int],
) -> float:
    """Reciprocal rank of the first chunk that contains a relevant sentence."""
    if not relevant_sentence_ids:
        return 0.0
    relevant = set(relevant_sentence_ids)
    for position, chunk in enumerate(retrieved_chunks, start=1):
        if relevant & set(chunk.sentence_ids):
            return 1.0 / position
    return 0.0

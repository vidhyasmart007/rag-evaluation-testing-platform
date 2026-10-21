"""Hybrid retrieval, blending lexical coverage with dense similarity.

Lexical retrieval is precise about words and blind to meaning. Dense
retrieval is the reverse: it matches a question to a passage that says the
same thing in different words, but it cannot tell "sick leave" from
"annual leave" if the embedding space puts them close together.

The hybrid score is a weighted blend of the two components. Both are
already on the unit interval, so the blend is on the unit interval too and
can be read directly: it is the weighted average of how well the chunk
matches the question's words and how well it matches its meaning.

Ranking uses the blend rather than the lexical ranking, which is the whole
point of running two retrievers instead of one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rageval.retrieval.lexical import Chunk
from rageval.schema import RetrievedChunk


@runtime_checkable
class LexicalScorer(Protocol):
    """The part of the lexical retriever the hybrid retriever relies on."""

    def score(self, question: str) -> list[tuple[int, float, float]]:
        """Return ``(chunk_index, bm25, coverage)`` for every chunk."""
        ...


@runtime_checkable
class DenseScorer(Protocol):
    """The part of the dense retriever the hybrid retriever relies on."""

    def score(self, question: str) -> list[tuple[int, float]]:
        """Return ``(chunk_index, similarity)`` for every chunk."""
        ...


class HybridRetriever:
    """Combines the lexical and the dense relevance scores."""

    name = "hybrid"

    def __init__(
        self,
        chunks: list[Chunk],
        lexical: LexicalScorer,
        dense: DenseScorer,
        *,
        lexical_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        if lexical_weight < 0 or dense_weight < 0:
            raise ValueError("Hybrid weights must not be negative.")
        total = lexical_weight + dense_weight
        if total <= 0:
            raise ValueError(
                "At least one hybrid weight must be greater than zero."
            )

        self.chunks = chunks
        self.lexical = lexical
        self.dense = dense
        self.lexical_weight = lexical_weight / total
        self.dense_weight = dense_weight / total

    def score(self, question: str) -> list[tuple[int, float]]:
        """Return ``(chunk_index, blended_score)`` for every chunk."""
        if not self.chunks:
            return []

        coverage = {
            index: value for index, _, value in self.lexical.score(question)
        }
        similarity = {
            index: value for index, value in self.dense.score(question)
        }

        return [
            (
                index,
                self.lexical_weight * coverage.get(index, 0.0)
                + self.dense_weight * similarity.get(index, 0.0),
            )
            for index in range(len(self.chunks))
        ]

    def retrieve(
        self,
        question: str,
        top_k: int,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the top scoring chunks above the relevance floor."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        scored = self.score(question)
        if not scored:
            return []

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        selected = [item for item in ranked if item[1] >= min_score][:top_k]

        return [
            RetrievedChunk(
                chunk_id=self.chunks[index].chunk_id,
                text=self.chunks[index].text,
                score=round(score, 6),
                sentence_ids=self.chunks[index].sentence_ids,
            )
            for index, score in selected
        ]

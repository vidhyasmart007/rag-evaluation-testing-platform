"""Dense retrieval over the knowledge base.

Every chunk is embedded once and the question is embedded once, then the
chunks are ranked by the cosine similarity of the two vectors.

The reported score is the cosine similarity clamped to the unit interval.
Clamping keeps the score on the same 0-1 scale as the lexical retriever's
coverage score, which is what makes the hybrid retriever's blend
meaningful: a cosine of 0.8 and a coverage of 0.8 can be averaged without
one of them silently dominating because of its range.
"""

from __future__ import annotations

import math

from rageval.retrieval.embeddings import Embedder
from rageval.retrieval.lexical import Chunk
from rageval.schema import RetrievedChunk


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity of two vectors.

    The raw value is returned, so it can be negative. The retriever clamps
    it afterwards; keeping the raw function pure makes it easy to reason
    about and to test.
    """
    if len(left) != len(right):
        raise ValueError(
            f"Cannot compare vectors of different sizes ({len(left)} and "
            f"{len(right)}). The question and the corpus were probably "
            "embedded with different models."
        )

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b

    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def clamp_similarity(similarity: float) -> float:
    """Clamp a cosine similarity into the unit interval."""
    return max(0.0, min(1.0, similarity))


class DenseRetriever:
    """Ranks chunks by the semantic similarity of their embeddings."""

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

        self._vectors = embedder.embed(
            [f"{document_prefix}{chunk.text}" for chunk in chunks]
        )
        if len(self._vectors) != len(chunks):
            raise ValueError(
                f"The embedder returned {len(self._vectors)} vectors for "
                f"{len(chunks)} chunks."
            )

    def score(self, question: str) -> list[tuple[int, float]]:
        """Return ``(chunk_index, similarity)`` for every chunk."""
        if not self.chunks:
            return []

        query_vector = self.embedder.embed([f"{self.query_prefix}{question}"])[0]
        return [
            (index, clamp_similarity(cosine_similarity(query_vector, vector)))
            for index, vector in enumerate(self._vectors)
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
                score=round(similarity, 6),
                sentence_ids=self.chunks[index].sentence_ids,
            )
            for index, similarity in selected
        ]

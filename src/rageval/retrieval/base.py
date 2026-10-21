"""The retriever interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rageval.schema import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """Anything that can turn a question into ranked chunks.

    Implementations are expected to return at most ``top_k`` chunks, ordered
    by descending relevance, and to leave out chunks that fall below their
    own relevance floor.
    """

    name: str

    def retrieve(
        self,
        question: str,
        top_k: int,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the chunks most relevant to ``question``.

        ``min_score`` is a relevance floor. Chunks below it are dropped
        rather than returned with a low score, so that a question the
        knowledge base cannot answer retrieves nothing at all.
        """
        ...

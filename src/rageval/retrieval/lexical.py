"""Lexical retrieval over a plain text knowledge base.

Documents are split on paragraph and sentence boundaries so that a chunk
never ends in the middle of a sentence. Every chunk records the ids of the
sentences it contains, which lets the dataset label relevance at sentence
granularity and keeps those labels valid when the chunk size changes.

Ranking uses Okapi BM25. Ranking and reporting are kept separate: BM25
produces the ordering, while the reported score is the IDF weighted
fraction of the question's content words that the chunk covers, which stays
inside the 0-1 range and is directly interpretable.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from rageval.schema import RetrievedChunk
from rageval.text import idf_from_frequencies, stemmed_tokens, weighted_coverage

_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A contiguous run of sentences taken from the source document."""

    chunk_id: int
    text: str
    sentence_ids: tuple[int, ...]
    word_count: int


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Blank-line separated blocks are treated as paragraphs, and each
    paragraph is split on sentence punctuation. A heading therefore stays
    its own sentence instead of being glued to the first line of the
    paragraph that follows it.
    """
    sentences: list[str] = []
    for paragraph in _PARAGRAPH_BREAK_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_BREAK_RE.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def build_chunks(
    sentences: list[str],
    chunk_size: int = 50,
    overlap: int = 1,
) -> list[Chunk]:
    """Group sentences into chunks of roughly ``chunk_size`` words.

    ``overlap`` is a number of sentences, not words. Overlapping whole
    sentences keeps every chunk readable on its own, which matters because
    chunks are fed to a language model as context.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")

    chunks: list[Chunk] = []
    current: list[tuple[int, str]] = []
    current_words = 0

    def emit() -> None:
        nonlocal current, current_words
        if not current:
            return
        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                text=" ".join(text for _, text in current),
                sentence_ids=tuple(sentence_id for sentence_id, _ in current),
                word_count=current_words,
            )
        )
        carry = current[-overlap:] if overlap else []
        current = list(carry)
        current_words = sum(len(text.split()) for _, text in current)

    for sentence_id, sentence in enumerate(sentences):
        words = len(sentence.split())
        if current and current_words + words > chunk_size:
            emit()
        current.append((sentence_id, sentence))
        current_words += words

    emit()
    return chunks


class BM25Retriever:
    """Ranks chunks with Okapi BM25 over the content words of a question."""

    name = "lexical"

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self._doc_tokens = [stemmed_tokens(chunk.text) for chunk in chunks]
        self._doc_counts = [Counter(tokens) for tokens in self._doc_tokens]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._average_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )

        self._document_frequency: Counter[str] = Counter()
        for counts in self._doc_counts:
            self._document_frequency.update(counts.keys())

    def _idf(self, term: str) -> float:
        total = len(self.chunks)
        frequency = self._document_frequency.get(term, 0)
        return idf_from_frequencies(total, frequency)

    def score(self, question: str) -> list[tuple[int, float, float]]:
        """Return ``(chunk_id, bm25, coverage)`` for every chunk.

        BM25 decides the ordering. The coverage is the IDF weighted
        fraction of the question's content words the chunk accounts for,
        which stays inside the 0-1 range and can be read directly as a
        relevance score in the report.
        """
        query_terms = Counter(stemmed_tokens(question))
        if not query_terms or not self.chunks:
            return []

        weights = {term: self._idf(term) for term in query_terms}
        results: list[tuple[int, float, float]] = []

        for index, counts in enumerate(self._doc_counts):
            bm25_score = 0.0

            for term in query_terms:
                term_frequency = counts.get(term, 0)
                if term_frequency == 0:
                    continue
                idf = weights[term]
                length_norm = 1.0
                if self._average_length:
                    length_norm = 1 - self.b + self.b * (
                        self._doc_lengths[index] / self._average_length
                    )
                denominator = term_frequency + self.k1 * length_norm
                bm25_score += idf * (term_frequency * (self.k1 + 1)) / denominator

            coverage = weighted_coverage(
                set(query_terms), self.chunks[index].text, weights
            )
            results.append((index, bm25_score, coverage))

        return results

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

        ranked = sorted(scored, key=lambda item: (item[1], item[2]), reverse=True)
        selected = [item for item in ranked if item[2] >= min_score][:top_k]

        return [
            RetrievedChunk(
                chunk_id=self.chunks[index].chunk_id,
                text=self.chunks[index].text,
                score=round(coverage, 6),
                sentence_ids=self.chunks[index].sentence_ids,
            )
            for index, _, coverage in selected
        ]

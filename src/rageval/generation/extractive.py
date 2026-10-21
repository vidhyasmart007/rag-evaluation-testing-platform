"""The extractive baseline generator.

This is the original behaviour of the project: pick the sentence from the
retrieved context that shares the most words with the question. It is kept
as a baseline arm so that a language model has something to be measured
against.

Two things are different from the original implementation. The context is
split into whole sentences rather than fragments, and the generator abstains
when no sentence in the context actually addresses the question.

The abstention decision uses plain word coverage. Weighting by inverse
document frequency sounds better in theory, but in practice it punishes a
question whose wording does not appear in the corpus at all ("which *city*
is the office in") exactly as hard as it punishes a question that asks
about something genuinely absent ("how many *sick* leave days"). No
lexical rule can tell those two apart, and the plain signal gives the
better overall operating point.
"""

from __future__ import annotations

from rageval.generation.base import ABSTAINED_ANSWER, GenerationResult
from rageval.retrieval.lexical import split_into_sentences
from rageval.text import coverage, stemmed_tokens


def _unique_sentences(sentences: list[str]) -> list[str]:
    """Drop repeated sentences, keeping the first occurrence.

    Chunks overlap, so the same sentence can arrive more than once.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for sentence in sentences:
        fingerprint = sentence.lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(sentence)
    return unique


class ExtractiveGenerator:
    """Selects the sentence with the highest weighted overlap with the question."""

    name = "extractive"

    def __init__(self, *, min_answer_score: float = 0.45) -> None:
        self.min_answer_score = min_answer_score

    def generate(self, question: str, context: str) -> GenerationResult:
        if not context.strip():
            return GenerationResult(
                text=ABSTAINED_ANSWER,
                model=self.name,
                abstained=True,
            )

        query_tokens = set(stemmed_tokens(question))
        if not query_tokens:
            return GenerationResult(
                text=ABSTAINED_ANSWER,
                model=self.name,
                abstained=True,
            )

        sentences = _unique_sentences(split_into_sentences(context))
        if not sentences:
            return GenerationResult(
                text=ABSTAINED_ANSWER,
                model=self.name,
                abstained=True,
            )

        best_sentence = ""
        best_score = 0.0
        for sentence in sentences:
            score = coverage(query_tokens, sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if not best_sentence or best_score < self.min_answer_score:
            return GenerationResult(
                text=ABSTAINED_ANSWER,
                model=self.name,
                abstained=True,
            )

        return GenerationResult(
            text=best_sentence,
            model=self.name,
            abstained=False,
        )

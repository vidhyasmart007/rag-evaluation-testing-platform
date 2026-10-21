"""Answer scoring.

The original project compared answers with a strict string match, which
credits a pipeline only when it reproduces the reference exactly. A language
model almost never does, so these measures give partial credit instead, and
keep the strict match as one metric among several rather than the verdict.
"""

from __future__ import annotations

from collections import Counter

from rageval.text import (
    extract_numbers,
    normalized_tokens,
    stemmed_tokens,
    token_counts,
)


def exact_match(prediction: str, reference: str) -> float:
    """1.0 when the two answers match after normalisation."""
    if not prediction.strip() or not reference.strip():
        return 0.0
    return 1.0 if normalized_tokens(prediction) == normalized_tokens(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall.

    Stop words are removed and light stemming is applied, so "18 days" and
    "18 day" compare as equal.
    """
    predicted = token_counts(prediction)
    expected = token_counts(reference)
    if not predicted or not expected:
        return 0.0

    overlap = sum((predicted & expected).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, reference: str) -> float:
    """Fraction of the reference's tokens that the prediction contains.

    More forgiving than F1 when the reference is short, which happens often
    with questions whose answer is a single number or a proper noun.
    """
    expected = token_counts(reference)
    if not expected:
        return 0.0
    predicted = token_counts(prediction)
    overlap = sum((predicted & expected).values())
    return overlap / sum(expected.values())


def numeric_match(prediction: str, reference: str) -> float | None:
    """Compare the numbers in two answers.

    Returns ``None`` when the reference contains no number, because in that
    case the comparison does not apply and the metric should be excluded
    from averages rather than counted as a failure.
    """
    expected = extract_numbers(reference)
    if not expected:
        return None
    found = extract_numbers(prediction)
    return 1.0 if found == expected else 0.0


def evaluate_answer_relevance(question: str, answer: str) -> float:
    """Share of the question's content words that appear in the answer."""
    question_tokens = set(stemmed_tokens(question))
    if not question_tokens:
        return 0.0
    answer_tokens = set(stemmed_tokens(answer))
    if not answer_tokens:
        return 0.0
    return len(question_tokens & answer_tokens) / len(question_tokens)


def evaluate_faithfulness(answer: str, context: str) -> float:
    """Share of the answer's content words that appear in the context.

    A lexical approximation. It cannot tell whether the answer contradicts
    the context, only whether it stays inside its vocabulary.
    """
    answer_tokens = {token for token in stemmed_tokens(answer) if len(token) > 2}
    if not answer_tokens:
        return 0.0
    context_tokens = set(stemmed_tokens(context))
    supported = answer_tokens & context_tokens
    return len(supported) / len(answer_tokens)


def token_overlap(left: str, right: str) -> float:
    """Share of the left text's content words that appear in the right text."""
    return token_recall(left, right)


__all__ = [
    "Counter",
    "evaluate_answer_relevance",
    "evaluate_faithfulness",
    "exact_match",
    "numeric_match",
    "token_f1",
    "token_overlap",
    "token_recall",
]

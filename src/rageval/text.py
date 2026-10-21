"""Shared tokenisation helpers.

Retrieval scoring and answer scoring both need to agree on what counts as a
"content word", and on how two different surface forms of the same word are
matched. Keeping the rules in one module stops the two halves of the
framework from drifting apart.

Every comparison in the framework runs on *stems*, not raw words. The
stems are produced by a deliberately conservative suffix stripper: it never
shortens a word below three characters, and it is applied identically to
both sides of every comparison, so "measures" and "measure" match while
"leave" and "leaves" match too.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

_WORD_RE = re.compile(r"\b\w+\b")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

STOP_WORDS = frozenset(
    {
        "a", "about", "after", "all", "also", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "but", "by",
        "can", "did", "do", "does", "doing", "done", "for", "from", "had",
        "has", "have", "he", "her", "here", "hers", "him", "his", "how", "i",
        "if", "in", "into", "is", "it", "its", "just", "me", "might", "more",
        "most", "much", "must", "my", "no", "nor", "not", "now", "of", "off",
        "on", "once", "only", "or", "other", "our", "ours", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "then", "there", "these", "they",
        "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "will", "with", "would", "you", "your", "yours",
        # Generic quantifiers and filler that add noise to question matching.
        "based", "example", "few", "get", "give", "given", "got", "known",
        "know", "lot", "lots", "made", "make", "many", "please", "several",
        "tell", "thanks", "thing", "things", "use", "used", "using", "want",
        "way", "ways",
    }
)

# Longest suffixes first so that "ations" is stripped before "ation", and
# "iness" before "s".
_SUFFIXES = (
    "ations",
    "ation",
    "nesses",
    "ness",
    "ments",
    "ment",
    "ings",
    "ing",
    "ies",
    "ied",
    "es",
    "ed",
    "s",
    "al",
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stop words included."""
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Word tokens with stop words removed, but not stemmed."""
    return [token for token in tokenize(text) if token not in STOP_WORDS]


def _drop_final_e(word: str) -> str:
    """Drop a trailing silent ``e`` so that "measure" and "measures" agree."""
    if len(word) >= 5 and word.endswith("e"):
        return word[:-1]
    return word


def stem(word: str) -> str:
    """Strip common English suffixes from a single word.

    The rules are intentionally shallow. A deeper stemmer would merge words
    that should stay apart, and the framework compares words against a
    corpus small enough that precision matters more than recall.

    The trailing silent ``e`` is removed both before and after the suffix
    is stripped. Doing it only once would make the result depend on which
    path the word took, and "employee" and "employees" would end up with
    different stems.
    """
    base = _drop_final_e(word)
    for suffix in _SUFFIXES:
        if base.endswith(suffix) and len(base) - len(suffix) >= 3:
            base = base[: -len(suffix)]
            if suffix in ("ies", "ied"):
                base += "y"
            break
    return _drop_final_e(base)


def stemmed_tokens(text: str) -> list[str]:
    """Content tokens with light stemming applied.

    This is the token form used everywhere two pieces of text are compared.
    """
    return [stem(token) for token in content_tokens(text)]


def normalized_tokens(text: str) -> list[str]:
    """Alias of :func:`stemmed_tokens`, kept for readability at call sites."""
    return stemmed_tokens(text)


def token_counts(text: str) -> Counter[str]:
    """Multiset of stemmed tokens for a piece of text."""
    return Counter(stemmed_tokens(text))


def extract_numbers(text: str) -> set[str]:
    """Numeric literals appearing in the text, as strings.

    Kept as strings so that "18" and "18.0" stay distinguishable and the
    caller decides how to compare them.
    """
    return set(_NUMBER_RE.findall(text))


def coverage(query_tokens: set[str], text: str) -> float:
    """Fraction of the query's stemmed tokens that appear in the text."""
    if not query_tokens:
        return 0.0
    present = set(stemmed_tokens(text))
    return len(query_tokens & present) / len(query_tokens)


def idf_from_frequencies(total: int, frequency: int) -> float:
    """Smoothed inverse document frequency of a term.

    Uses the Okapi BM25 form, so that a term which appears in every
    document still carries a small positive weight rather than zero.
    """
    return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))


def idf_weights(
    documents: Sequence[str],
    terms: Iterable[str],
) -> dict[str, float]:
    """Inverse document frequency of each term across the given documents."""
    total = len(documents)
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(set(stemmed_tokens(document)))

    return {
        term: idf_from_frequencies(total, frequencies.get(term, 0))
        for term in terms
    }


def weighted_coverage(
    query_tokens: set[str],
    text: str,
    weights: dict[str, float],
) -> float:
    """IDF weighted fraction of the query's tokens that appear in the text.

    Weighting by inverse document frequency means that failing to match a
    rare, distinctive term hurts far more than failing to match a common
    one. That is the property that makes this useful for deciding whether a
    piece of context actually addresses a question.
    """
    if not query_tokens:
        return 0.0
    total = sum(weights.get(term, 0.0) for term in query_tokens)
    if total <= 0:
        return 0.0
    present = set(stemmed_tokens(text))
    matched = sum(weights.get(term, 0.0) for term in query_tokens & present)
    return matched / total

"""Tests for the shared tokenisation helpers.

The tokenizer is the foundation every metric sits on, so its behaviour is
pinned down here rather than being assumed by the other test modules.
"""

from __future__ import annotations

import pytest

from rageval.text import (
    content_tokens,
    coverage,
    extract_numbers,
    idf_weights,
    stem,
    stemmed_tokens,
    token_counts,
    weighted_coverage,
)


def test_stop_words_are_removed_but_content_words_survive():
    assert content_tokens("The office is in Bengaluru") == ["office", "bengaluru"]


def test_generic_quantifiers_are_treated_as_stop_words():
    assert "many" not in content_tokens("How many annual leave days?")


def test_stemming_matches_singular_and_plural_forms():
    assert stem("measure") == stem("measures")
    assert stem("employee") == stem("employees")
    assert stem("day") == stem("days")
    assert stem("development") == stem("develop")
    assert stem("retrieval") == stem("retrieve")
    assert stem("language") == stem("languages")


def test_stemming_leaves_short_words_alone():
    assert stem("is") == "is"
    assert stem("use") == stem("uses")


def test_stemming_is_idempotent():
    for word in ("measure", "employees", "development", "office", "retrieval"):
        assert stem(stem(word)) == stem(word)


def test_coverage_is_the_fraction_of_query_terms_present():
    assert coverage({"offic"}, "The office is in Bengaluru") == pytest.approx(1.0)
    assert coverage({"offic", "cit"}, "The office is in Bengaluru") == pytest.approx(0.5)
    assert coverage(set(), "anything at all") == 0.0


def test_weighted_coverage_punishes_a_missing_rare_term_harder():
    documents = [
        "the office is in bengaluru",
        "the office is in mumbai",
        "the capital of mars is unknown",
    ]
    query = set(stemmed_tokens("office mars"))
    weights = idf_weights(documents, query)

    weighted = weighted_coverage(query, documents[0], weights)
    plain = coverage(query, documents[0])

    # The plain count gives both terms a half each. Weighting pushes the
    # score down because the rare term is the one that is missing.
    assert plain == pytest.approx(0.5)
    assert weighted < plain
    assert weighted_coverage(set(stemmed_tokens("office")), documents[0], weights) == pytest.approx(1.0)


def test_token_counts_are_multisets():
    counts = token_counts("leave day leave")
    assert counts["leav"] == 2
    assert counts["day"] == 1


def test_extract_numbers_keeps_literals_as_written():
    assert extract_numbers("Employees receive 18 annual leave days") == {"18"}
    assert extract_numbers("The budget is 1.5 million") == {"1.5"}
    assert extract_numbers("no digits here") == set()

"""Tests for the answer generation layer."""

from __future__ import annotations

from rageval.generation.base import ABSTAINED_ANSWER
from rageval.generation.extractive import ExtractiveGenerator

CONTEXT = (
    "The company office is located in Bengaluru, Karnataka. "
    "Employees receive 18 annual leave days per calendar year."
)


def test_answers_with_the_matching_sentence():
    generator = ExtractiveGenerator()
    result = generator.generate("How many annual leave days do employees get?", CONTEXT)
    assert not result.abstained
    assert "18 annual leave days" in result.text


def test_abstains_when_the_context_is_empty():
    result = ExtractiveGenerator().generate("How many annual leave days?", "")
    assert result.abstained
    assert result.text == ABSTAINED_ANSWER


def test_abstains_when_the_context_does_not_address_the_question():
    generator = ExtractiveGenerator()
    question = "What is the name of the chief executive officer?"
    result = generator.generate(question, CONTEXT)
    assert result.abstained
    assert result.text == ABSTAINED_ANSWER


def test_abstains_when_the_question_has_no_content_words():
    generator = ExtractiveGenerator()
    result = generator.generate("What is it?", CONTEXT)
    assert result.abstained


def test_a_repeated_sentence_is_only_answered_once():
    """Chunks overlap, so the same sentence can arrive more than once."""
    generator = ExtractiveGenerator()
    repeated = "Employees receive 18 annual leave days per calendar year. " * 3
    result = generator.generate("How many annual leave days?", repeated)
    assert result.text == "Employees receive 18 annual leave days per calendar year."


def test_the_answer_is_one_whole_sentence():
    generator = ExtractiveGenerator()
    result = generator.generate("Where is the office?", CONTEXT)
    assert result.text.endswith(".")
    assert "\n" not in result.text


def test_the_threshold_controls_abstention():
    strict = ExtractiveGenerator(min_answer_score=0.99)
    lenient = ExtractiveGenerator(min_answer_score=0.1)

    question = "Which city is the office in?"
    assert strict.generate(question, CONTEXT).abstained
    assert not lenient.generate(question, CONTEXT).abstained

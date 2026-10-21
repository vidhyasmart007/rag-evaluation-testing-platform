import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from evaluator import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_retrieval,
)


def test_relevant_answer_has_positive_score():
    score = evaluate_answer_relevance(
        "What is Python?",
        "Python is a programming language."
    )

    assert score > 0


def test_faithful_answer_is_supported_by_context():
    score = evaluate_faithfulness(
        "Python is a programming language.",
        "Python is a programming language."
    )

    assert score == 1.0


def test_unsupported_answer_has_low_faithfulness():
    score = evaluate_faithfulness(
        "Zebras live on Mars.",
        "Python is a programming language."
    )

    assert score == 0.0


def test_empty_retrieval_returns_zero():
    score = evaluate_retrieval(
        "What is Python?",
        []
    )

    assert score == 0.0
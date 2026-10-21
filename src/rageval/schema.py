"""Core data structures shared by the dataset, the runner and the reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TestCase:
    """A single evaluation case.

    ``answerable`` marks whether the knowledge base can answer the question
    at all. Unanswerable cases exist so the framework can score abstention
    behaviour instead of only scoring answers.

    ``relevant_sentence_ids`` points at sentence ids in the source
    document. Sentence ids rather than chunk ids, because chunk boundaries
    move whenever the chunk size changes and labels would silently rot.
    """

    id: int
    question: str
    reference_answer: str = ""
    answerable: bool = True
    relevant_sentence_ids: tuple[int, ...] = ()
    difficulty: str = "medium"
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any, *, index: int) -> TestCase:
        if not isinstance(raw, dict):
            raise ValueError(f"Test case at index {index} must be an object.")

        missing = {"id", "question"} - set(raw)
        if missing:
            raise ValueError(
                f"Test case at index {index} is missing: "
                f"{', '.join(sorted(missing))}"
            )

        case_id = raw["id"]
        if isinstance(case_id, bool) or not isinstance(case_id, int):
            raise ValueError(f"Test case at index {index} has a non-integer id.")

        question = raw["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Test case {case_id} must have a non-empty question.")

        reference_answer = raw.get("reference_answer", raw.get("expected_answer", ""))
        if not isinstance(reference_answer, str):
            raise ValueError(f"Test case {case_id} has a non-string reference_answer.")

        answerable = raw.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"Test case {case_id} has a non-boolean answerable flag.")

        sentence_ids = raw.get("relevant_sentence_ids", [])
        if not isinstance(sentence_ids, list) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in sentence_ids
        ):
            raise ValueError(
                f"Test case {case_id} must list relevant_sentence_ids as integers."
            )

        difficulty = raw.get("difficulty", "medium")
        if not isinstance(difficulty, str):
            raise ValueError(f"Test case {case_id} has a non-string difficulty.")

        tags = raw.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"Test case {case_id} must list tags as strings.")

        return cls(
            id=case_id,
            question=question.strip(),
            reference_answer=reference_answer.strip(),
            answerable=answerable,
            relevant_sentence_ids=tuple(sorted(set(sentence_ids))),
            difficulty=difficulty,
            tags=tuple(tags),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "answerable": self.answerable,
            "relevant_sentence_ids": list(self.relevant_sentence_ids),
            "difficulty": self.difficulty,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a retriever, with its relevance score."""

    chunk_id: int
    text: str
    score: float
    sentence_ids: tuple[int, ...] = ()


@dataclass
class CaseResult:
    """The outcome of evaluating a single test case."""

    test_id: int
    question: str
    reference_answer: str
    answerable: bool
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    answer: str = ""
    abstained: bool = False
    abstention_reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: int = 0
    generator: str = ""
    cached: bool = False
    judge: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "answerable": self.answerable,
            "retrieved": [
                {
                    "chunk_id": chunk.chunk_id,
                    "score": chunk.score,
                    "text": chunk.text,
                    "sentence_ids": list(chunk.sentence_ids),
                }
                for chunk in self.retrieved
            ],
            "answer": self.answer,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
            "generator": self.generator,
            "cached": self.cached,
            "judge": self.judge,
            "error": self.error,
        }


@dataclass
class RunResult:
    """The outcome of a full evaluation run."""

    run_id: str
    created_at: str
    config: dict[str, Any]
    dataset_version: str
    dataset_hash: str
    git_commit: str | None
    cases: list[CaseResult] = field(default_factory=list)
    aggregates: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": self.config,
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "git_commit": self.git_commit,
            "aggregates": self.aggregates,
            "cases": [case.to_dict() for case in self.cases],
        }

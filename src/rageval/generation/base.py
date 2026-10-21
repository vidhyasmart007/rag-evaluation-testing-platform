"""The generator interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

ABSTAINED_ANSWER = "I don't know based on the available context."


class GenerationError(RuntimeError):
    """Raised when an answer could not be produced.

    The runner catches this per case, so one unreachable model degrades a
    single case instead of the whole run.
    """


@dataclass(frozen=True)
class GenerationResult:
    """A generated answer plus the accounting needed to report on it.

    ``reason`` says why a generator abstained. "The context was empty" and
    "the model decided the context did not answer the question" are very
    different failures, and a framework that only records the boolean cannot
    tell them apart.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False
    abstained: bool = False
    reason: str = ""


@runtime_checkable
class Generator(Protocol):
    """Anything that can answer a question from a retrieved context."""

    name: str

    def generate(self, question: str, context: str) -> GenerationResult:
        """Return an answer for ``question`` grounded in ``context``."""
        ...

    @property
    def cache_key(self) -> dict[str, Any]:
        """The settings that change the answer, for use in a cache key.

        A generator whose output depends on settings the caller can change,
        such as a temperature or a threshold, has to say so. Otherwise the
        disk cache happily serves an answer produced by different settings
        and the run silently reports the old number.
        """
        ...

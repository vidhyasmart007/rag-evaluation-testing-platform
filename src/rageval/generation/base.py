"""The generator interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

ABSTAINED_ANSWER = "I don't know based on the available context."


@dataclass(frozen=True)
class GenerationResult:
    """A generated answer plus the accounting needed to report on it."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False
    abstained: bool = False


@runtime_checkable
class Generator(Protocol):
    """Anything that can answer a question from a retrieved context."""

    name: str

    def generate(self, question: str, context: str) -> GenerationResult:
        """Return an answer for ``question`` grounded in ``context``."""
        ...

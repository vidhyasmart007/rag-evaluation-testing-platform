"""Answer generation strategies."""

from __future__ import annotations

from rageval.generation.base import ABSTAINED_ANSWER, GenerationResult, Generator
from rageval.generation.extractive import ExtractiveGenerator

__all__ = [
    "ABSTAINED_ANSWER",
    "ExtractiveGenerator",
    "GenerationResult",
    "Generator",
]

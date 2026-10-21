"""RAG evaluation and testing platform.

The package is organised around one idea: a single evaluation run should be
reproducible and should not depend on any particular front end. Everything
here runs from the command line, and the API and dashboard layers are thin
wrappers over the same functions.
"""

from __future__ import annotations

from rageval.config import EvalConfig
from rageval.dataset import Dataset, load_dataset
from rageval.runner import run_evaluation
from rageval.schema import CaseResult, RetrievedChunk, RunResult, TestCase

__version__ = "0.2.0"

__all__ = [
    "CaseResult",
    "Dataset",
    "EvalConfig",
    "RetrievedChunk",
    "RunResult",
    "TestCase",
    "__version__",
    "load_dataset",
    "run_evaluation",
]

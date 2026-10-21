"""Shared fixtures for the test suite."""

from __future__ import annotations

import json

import pytest

from rageval.config import EvalConfig

KNOWLEDGE_BASE = """Company Knowledge Base

The company office is located in Bengaluru, Karnataka.

Employees receive 18 annual leave days per calendar year.

The evaluation suite is executed with pytest.
"""

DATASET = {
    "version": "2.0.0",
    "description": "A tiny set used to exercise the runner end to end.",
    "cases": [
        {
            "id": 1,
            "question": "Where is the company office?",
            "reference_answer": "The company office is located in Bengaluru, Karnataka.",
            "answerable": True,
            "relevant_sentence_ids": [1],
        },
        {
            "id": 2,
            "question": "How many annual leave days do employees receive?",
            "reference_answer": (
                "Employees receive 18 annual leave days per calendar year."
            ),
            "answerable": True,
            "relevant_sentence_ids": [2],
        },
        {
            "id": 3,
            "question": "What is the capital of Mars?",
            "reference_answer": "",
            "answerable": False,
            "relevant_sentence_ids": [],
        },
    ],
}


@pytest.fixture
def knowledge_base(tmp_path):
    path = tmp_path / "knowledge_base.txt"
    path.write_text(KNOWLEDGE_BASE, encoding="utf-8")
    return path


@pytest.fixture
def dataset_path(tmp_path):
    path = tmp_path / "eval_dataset.json"
    path.write_text(json.dumps(DATASET), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path, knowledge_base, dataset_path):
    """A configuration that runs entirely inside the temporary directory."""
    return EvalConfig(
        dataset_path=dataset_path,
        knowledge_base_path=knowledge_base,
        reports_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        run_name="test_run",
    )

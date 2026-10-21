"""Tests for dataset loading and validation."""

from __future__ import annotations

import json

import pytest

from rageval.dataset import load_dataset


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_the_versioned_object_format(tmp_path):
    path = write_json(
        tmp_path / "dataset.json",
        {
            "version": "2.0.0",
            "description": "A test set.",
            "cases": [
                {
                    "id": 1,
                    "question": "Where is the office?",
                    "reference_answer": "Bengaluru",
                    "answerable": True,
                    "relevant_sentence_ids": [1],
                }
            ],
        },
    )
    dataset = load_dataset(path)

    assert dataset.version == "2.0.0"
    assert len(dataset) == 1
    assert dataset.cases[0].relevant_sentence_ids == (1,)
    assert dataset.content_hash


def test_accepts_the_legacy_list_format_with_a_warning(tmp_path):
    path = write_json(
        tmp_path / "dataset.json",
        [
            {
                "id": 1,
                "question": "Where is the office?",
                "expected_answer": "Bengaluru",
                "actual_answer": "Bengaluru",
            }
        ],
    )
    dataset = load_dataset(path)

    assert dataset.version == "legacy"
    assert dataset.cases[0].reference_answer == "Bengaluru"
    assert any("legacy" in warning for warning in dataset.warnings)


def test_the_legacy_actual_answer_is_ignored(tmp_path):
    """Answers must come from the pipeline, never from the dataset."""
    path = write_json(
        tmp_path / "dataset.json",
        [
            {
                "id": 1,
                "question": "Where is the office?",
                "expected_answer": "Bengaluru",
                "actual_answer": "a hand written answer",
            }
        ],
    )
    dataset = load_dataset(path)
    assert dataset.cases[0].reference_answer == "Bengaluru"


def test_duplicate_ids_are_rejected(tmp_path):
    path = write_json(
        tmp_path / "dataset.json",
        [
            {"id": 1, "question": "One?", "reference_answer": "One"},
            {"id": 1, "question": "Two?", "reference_answer": "Two"},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)


def test_an_answerable_case_without_a_reference_is_rejected(tmp_path):
    path = write_json(
        tmp_path / "dataset.json",
        [{"id": 1, "question": "Where is the office?", "reference_answer": ""}],
    )
    with pytest.raises(ValueError, match="reference_answer"):
        load_dataset(path)


def test_a_non_integer_id_is_rejected(tmp_path):
    path = write_json(
        tmp_path / "dataset.json",
        [{"id": "one", "question": "Where?", "reference_answer": "Here"}],
    )
    with pytest.raises(ValueError, match="non-integer id"):
        load_dataset(path)


def test_a_missing_question_is_rejected(tmp_path):
    path = write_json(
        tmp_path / "dataset.json",
        [{"id": 1, "reference_answer": "Here"}],
    )
    with pytest.raises(ValueError, match="missing"):
        load_dataset(path)


def test_non_integer_sentence_ids_are_rejected(tmp_path):
    path = write_json(
        tmp_path / "dataset.json",
        [
            {
                "id": 1,
                "question": "Where?",
                "reference_answer": "Here",
                "relevant_sentence_ids": ["one"],
            }
        ],
    )
    with pytest.raises(ValueError, match="relevant_sentence_ids"):
        load_dataset(path)


def test_invalid_json_is_reported_clearly(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_dataset(path)


def test_a_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.json")


def test_the_content_hash_is_stable_and_content_sensitive(tmp_path):
    first = write_json(
        tmp_path / "a.json",
        [{"id": 1, "question": "Where?", "reference_answer": "Here"}],
    )
    second = write_json(
        tmp_path / "b.json",
        [{"id": 1, "question": "Where?", "reference_answer": "Here"}],
    )
    third = write_json(
        tmp_path / "c.json",
        [{"id": 1, "question": "Where?", "reference_answer": "There"}],
    )

    assert load_dataset(first).content_hash == load_dataset(second).content_hash
    assert load_dataset(first).content_hash != load_dataset(third).content_hash


def test_answerable_cases_without_sentence_labels_produce_a_warning(tmp_path):
    path = write_json(
        tmp_path / "dataset.json",
        [{"id": 1, "question": "Where?", "reference_answer": "Here"}],
    )
    dataset = load_dataset(path)
    assert any("relevant_sentence_ids" in warning for warning in dataset.warnings)


def test_unanswerable_cases_are_counted_separately(tmp_path):
    path = write_json(
        tmp_path / "dataset.json",
        [
            {"id": 1, "question": "Where?", "reference_answer": "Here"},
            {"id": 2, "question": "Who is the CEO?", "reference_answer": "", "answerable": False},
        ],
    )
    dataset = load_dataset(path)
    assert len(dataset.answerable_cases) == 1
    assert len(dataset.unanswerable_cases) == 1

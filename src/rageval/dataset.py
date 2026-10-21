"""Dataset loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rageval.schema import TestCase


@dataclass(frozen=True)
class Dataset:
    """A validated collection of evaluation cases."""

    version: str
    description: str
    cases: tuple[TestCase, ...]
    content_hash: str
    warnings: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.cases)

    def get(self, case_id: int) -> TestCase | None:
        for case in self.cases:
            if case.id == case_id:
                return case
        return None

    @property
    def answerable_cases(self) -> tuple[TestCase, ...]:
        return tuple(case for case in self.cases if case.answerable)

    @property
    def unanswerable_cases(self) -> tuple[TestCase, ...]:
        return tuple(case for case in self.cases if not case.answerable)


def load_dataset(path: Path) -> Dataset:
    """Load and validate the evaluation dataset.

    Both the current object format and the older bare-list format are
    accepted. The legacy format carried a hand written ``actual_answer``
    field, which is ignored: answers are supposed to be produced by the
    pipeline under test, never supplied by the dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Dataset is not valid JSON: {error}") from error

    warnings: list[str] = []

    if isinstance(raw, list):
        version = "legacy"
        description = "Legacy dataset (bare list format)."
        raw_cases: Any = raw
        warnings.append(
            "Dataset uses the legacy list format; consider migrating to the "
            "versioned object format with 'version' and 'cases'."
        )
    elif isinstance(raw, dict):
        version = str(raw.get("version", "unversioned"))
        description = str(raw.get("description", ""))
        raw_cases = raw.get("cases", [])
        if not isinstance(raw_cases, list):
            raise ValueError("Dataset field 'cases' must be a list.")
    else:
        raise ValueError("Dataset must be a JSON list or a JSON object.")

    cases = tuple(
        TestCase.from_dict(entry, index=index) for index, entry in enumerate(raw_cases)
    )

    seen: set[int] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"Duplicate test case id: {case.id}")
        seen.add(case.id)

    for case in cases:
        if case.answerable and not case.reference_answer:
            raise ValueError(
                f"Test case {case.id} is answerable but has no reference_answer."
            )
        if case.answerable and not case.relevant_sentence_ids:
            warnings.append(
                f"Test case {case.id} has no relevant_sentence_ids, so retrieval "
                "metrics will be skipped for it."
            )
        if not case.answerable and case.relevant_sentence_ids:
            warnings.append(
                f"Test case {case.id} is unanswerable but lists relevant sentences."
            )

    content_hash = hashlib.sha256(
        json.dumps(
            [case.to_dict() for case in cases],
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    return Dataset(
        version=version,
        description=description,
        cases=cases,
        content_hash=content_hash,
        warnings=tuple(warnings),
    )

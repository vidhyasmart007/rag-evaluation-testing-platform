"""The hand labels: the ground truth a judge is measured against.

A judge is measured by how well it agrees with a person who read the
question, the sources and the answer and decided what was right. That
decision is the label, and it is made by reading, not by running a metric.

The rule the labels are made against is deliberately narrow: an answer is
correct when it says something true in answer to the question, and
incorrect when it says something false, says nothing when an answer was
available, or answers a different question. A shorter answer is not a
wrong answer. An answer that adds a true detail is not a wrong answer.
This is the definition the token rule fails to apply, and the whole reason
a judge is worth building.

The context is part of each item, so a labelled set is self contained:
grading it needs no retriever, no knowledge base and no model server
beyond the judge itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORRECT = "correct"
INCORRECT = "incorrect"
LABELS = (CORRECT, INCORRECT)


@dataclass(frozen=True)
class LabelledItem:
    """One answer, with a person's judgement of it.

    ``used_for_tuning`` marks the items that were looked at while the rubric
    was being written. A rubric written with a validation set open in front
    of you will fit that set better than it generalises, and the only
    honest way to report the difference is to say which items were open.
    The items that were not touched are the held-out ones, and the
    agreement on those is the number that means something.
    """

    test_id: int
    question: str
    reference_answer: str
    context: str
    answer: str
    label: str
    note: str = ""
    source: str = ""
    used_for_tuning: bool = False

    @property
    def correct(self) -> bool:
        """Whether the person judged this answer correct."""
        return self.label == CORRECT

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "context": self.context,
            "answer": self.answer,
            "label": self.label,
            "note": self.note,
            "source": self.source,
            "used_for_tuning": self.used_for_tuning,
        }


def _require_str(raw: dict[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{where} needs a string '{key}'.")
    return value


def load_labels(path: Path | str) -> list[LabelledItem]:
    """Load a labelled set from a JSON file.

    The file holds the items under ``items`` and may carry a ``version``
    and a ``description`` alongside them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hand labels not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object.")
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError(f"{path} must hold a list of items under 'items'.")

    items: list[LabelledItem] = []
    seen: set[tuple[int, str]] = set()

    for index, raw in enumerate(raw_items):
        where = f"{path} item {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be an object.")

        test_id = raw.get("test_id")
        if isinstance(test_id, bool) or not isinstance(test_id, int):
            raise ValueError(f"{where} needs an integer 'test_id'.")

        label = _require_str(raw, "label", where)
        if label not in LABELS:
            raise ValueError(
                f"{where} has label '{label}'. Use one of: {', '.join(LABELS)}."
            )

        answer = _require_str(raw, "answer", where)
        key = (test_id, answer)
        if key in seen:
            raise ValueError(
                f"{where} repeats the answer for case {test_id}. Each distinct "
                "answer should be labelled once."
            )
        seen.add(key)

        items.append(
            LabelledItem(
                test_id=test_id,
                question=_require_str(raw, "question", where),
                reference_answer=_require_str(raw, "reference_answer", where),
                context=_require_str(raw, "context", where),
                answer=answer,
                label=label,
                note=_require_str(raw, "note", where) if "note" in raw else "",
                source=_require_str(raw, "source", where) if "source" in raw else "",
                used_for_tuning=bool(raw.get("used_for_tuning", False)),
            )
        )

    if not items:
        raise ValueError(f"{path} holds no labelled items.")
    return items


def write_labels(
    path: Path | str,
    items: list[LabelledItem],
    *,
    version: str = "1.0",
    description: str = "",
) -> Path:
    """Write a labelled set back out, for review and for the repository."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "description": description,
        "items": [item.to_dict() for item in items],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path

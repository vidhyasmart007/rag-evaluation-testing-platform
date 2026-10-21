"""Collect the distinct answers a set of runs produced.

The hand labelled set is every distinct answer the framework produced
during the phase before the judge was built, not a sample of it. That is
deliberate: a hand picked set of twenty can be arranged to flatter
whatever is being measured, and this one cannot be, because it contains
every answer the framework got wrong as well as every answer it got right.

The items carry the context that was actually used, so the labelled set is
self contained: grading it needs no retriever and no knowledge base.

This module produces the **candidates**. The label itself, and the
``used_for_tuning`` flag that marks the items the rubric was written against,
are a human's judgements and live in ``data/hand_labels.json``. Collecting is
mechanical; labelling is not, and it is not automated here.

Run it as a module to print the candidate set:

    python -m rageval.judge.collect reports/phase3_*.json reports/baseline.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CollectedItem:
    """One distinct answer, with the context it was produced from."""

    test_id: int
    question: str
    reference_answer: str
    context: str
    answer: str
    sources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "context": self.context,
            "answer": self.answer,
            "source": ", ".join(self.sources),
        }


def collect_from_runs(paths: list[Path]) -> list[CollectedItem]:
    """Collect every distinct answer from a set of run reports.

    The same answer produced by three runs is one item, not three. The
    runs are recorded so that an item can be traced back to the run it
    came from.
    """
    items: dict[tuple[int, str], CollectedItem] = {}

    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        run_id = str(report.get("run_id", Path(path).stem))
        for case in report.get("cases", []):
            answer = str(case.get("answer", ""))
            key = (int(case["test_id"]), answer)
            if key in items:
                existing = items[key]
                if run_id not in existing.sources:
                    items[key] = CollectedItem(
                        test_id=existing.test_id,
                        question=existing.question,
                        reference_answer=existing.reference_answer,
                        context=existing.context,
                        answer=existing.answer,
                        sources=existing.sources + (run_id,),
                    )
                continue

            # The context is the retrieved chunks joined the way the runner
            # joins them, so the judge sees exactly what the generator saw.
            context = "\n".join(
                str(chunk.get("text", "")) for chunk in case.get("retrieved", [])
            )
            items[key] = CollectedItem(
                test_id=int(case["test_id"]),
                question=str(case.get("question", "")),
                reference_answer=str(case.get("reference_answer", "")),
                context=context,
                answer=answer,
                sources=(run_id,),
            )

    return sorted(items.values(), key=lambda item: (item.test_id, item.answer))


def main(argv: list[str] | None = None) -> int:
    """Print the candidate set as JSON, ready to be labelled."""
    paths = [Path(argument) for argument in (argv or sys.argv[1:])]
    if not paths:
        print(
            "Usage: python -m rageval.judge.collect REPORT [REPORT ...]",
            file=sys.stderr,
        )
        return 2

    items = collect_from_runs(paths)
    payload = {
        "version": "1.0",
        "description": "Distinct answers collected from run reports.",
        "items": [item.to_dict() for item in items],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

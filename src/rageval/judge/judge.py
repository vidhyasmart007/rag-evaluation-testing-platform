"""The judge: a rubric applied to a chat client, several times.

This is the piece that turns a chat client into a measurement. It runs the
rubric in both orders, three times each by default, and folds the samples
into a single verdict while recording how much the judge agreed with
itself.

The failure of a single sample is tolerated and counted. The failure of
every sample is not tolerated: an item the judge could not read at all is
an error, not a score of one, because an unreadable reply silently
becomes "incorrect" and drags every metric that depends on the judge down
without saying so.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rageval.judge.base import (
    ORDER_ANSWER_FIRST,
    ORDER_REFERENCE_FIRST,
    ChatClient,
    JudgeError,
    JudgeItem,
    Judgement,
    RawJudgement,
    aggregate,
)
from rageval.judge.rubric import build_messages, parse_judgement


class RubricJudge:
    """Grades answers against the rubric, in both orders, several times."""

    def __init__(
        self,
        client: ChatClient,
        *,
        n_samples: int = 3,
        swap_order: bool = True,
        threshold: int = 4,
    ) -> None:
        if n_samples < 1:
            raise ValueError("A judge needs at least one sample per order.")
        self.client = client
        self.n_samples = n_samples
        self.swap_order = swap_order
        self.threshold = threshold
        self.name = client.name

    @property
    def orders(self) -> tuple[str, ...]:
        """The orders the rubric is run in."""
        if self.swap_order:
            return (ORDER_REFERENCE_FIRST, ORDER_ANSWER_FIRST)
        return (ORDER_REFERENCE_FIRST,)

    def judge(self, item: JudgeItem) -> Judgement:
        """Grade one item."""
        raws: list[RawJudgement] = []
        failures = 0
        last_error: Exception | None = None

        for order in self.orders:
            messages = build_messages(
                item.question,
                item.answer,
                item.context,
                item.reference_answer,
                order,
            )
            for _ in range(self.n_samples):
                try:
                    text = self.client.complete(messages)
                except Exception as error:  # noqa: BLE001 - counted, not hidden
                    failures += 1
                    last_error = error
                    continue

                parsed = parse_judgement(text, order=order)
                if parsed is None:
                    failures += 1
                    continue
                raws.append(parsed)

        if not raws:
            raise JudgeError(
                f"The judge produced no readable verdict for this item "
                f"({failures} failed sample(s))."
                + (f" Last error: {last_error}" if last_error else "")
            )

        judgement = aggregate(raws, threshold=self.threshold)
        return replace(judgement, n_failed=failures)

    def to_dict(self) -> dict[str, Any]:
        """The judge settings that go into a report."""
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "swap_order": self.swap_order,
            "threshold": self.threshold,
            "client": dict(getattr(self.client, "cache_key", {})),
        }

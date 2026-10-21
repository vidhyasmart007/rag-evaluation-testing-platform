"""The judge interface, and how several judgements become one verdict.

A judge is asked the same question several times on purpose. There are two
reasons for that, and both of them are about honesty rather than about
averaging.

Three samples per order expose the judge's own instability. A judge that
changes its mind about the same answer between two calls is not a
measurement, and the framework should say so rather than hide it behind a
single number.

The order in which the reference answer and the graded answer appear is
swapped. A judge that changes its verdict when the two blocks swap is
sensitive to position rather than to content, and its verdict is then
measuring the prompt layout. Both orders are run, and the disagreement rate
is reported.

The aggregated verdict is the lower median of the sample scores and a strict
majority on the binary verdict. The median is used rather than the mean
because a single wild sample should not move the result, and the vote
rather than the first sample because one call is not a measurement either.
The two are the same rule applied to the same numbers, so the score the
report shows and the verdict the report gives can never contradict each
other.

The binary verdict is derived from the correctness score rather than asked
for separately. Two fields that can disagree with each other is one field
too many, and a judge asked to produce both will eventually contradict
itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: The dimensions the rubric scores. Each is an integer from 1 to 5.
DIMENSIONS = ("correctness", "relevance", "groundedness", "completeness")

#: The bounds of the rubric scale.
MIN_SCORE = 1
MAX_SCORE = 5

#: The two orders the reference answer and the graded answer can appear in.
ORDER_REFERENCE_FIRST = "reference_first"
ORDER_ANSWER_FIRST = "answer_first"
ORDERS = (ORDER_REFERENCE_FIRST, ORDER_ANSWER_FIRST)


class JudgeError(RuntimeError):
    """Raised when a judgement could not be produced at all.

    The runner catches this per case, so one unreachable judge degrades a
    single case rather than the whole run.
    """


@dataclass(frozen=True)
class JudgeItem:
    """One answer to be graded.

    The context is part of the item rather than something the judge looks
    up, so a validation set is self contained: grading it needs no
    retriever, no knowledge base and no model server beyond the judge
    itself.
    """

    question: str
    answer: str
    context: str = ""
    reference_answer: str = ""
    test_id: int | None = None


@dataclass(frozen=True)
class RawJudgement:
    """One sample: a single call to the judge, parsed but not aggregated."""

    scores: dict[str, int]
    reason: str = ""
    order: str = ORDER_REFERENCE_FIRST
    raw_text: str = ""

    @property
    def correctness(self) -> int:
        return int(self.scores.get("correctness", MIN_SCORE))


@dataclass(frozen=True)
class Judgement:
    """The judge's verdict on one item, aggregated over its samples.

    ``sample_agreement`` is the share of samples that agreed with the
    majority verdict. It is a property of the judge, not of the answer, and
    a low value on an otherwise good answer is the signal that the judge
    is unstable on that kind of item.

    ``order_consistent`` is ``None`` when the order was not swapped, so
    that "not tested" never reads as "passed".
    """

    correct: bool
    correctness: int
    relevance: int
    groundedness: int
    completeness: int
    reason: str = ""
    n_samples: int = 0
    n_failed: int = 0
    sample_agreement: float = 1.0
    order_consistent: bool | None = None
    order_verdicts: tuple[tuple[str, bool], ...] = field(default=())

    @property
    def scores(self) -> dict[str, int]:
        return {
            "correctness": self.correctness,
            "relevance": self.relevance,
            "groundedness": self.groundedness,
            "completeness": self.completeness,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            **self.scores,
            "reason": self.reason,
            "n_samples": self.n_samples,
            "n_failed": self.n_failed,
            "sample_agreement": round(self.sample_agreement, 4),
            "order_consistent": self.order_consistent,
            "order_verdicts": [
                {"order": order, "correct": verdict}
                for order, verdict in self.order_verdicts
            ],
        }


@runtime_checkable
class ChatClient(Protocol):
    """Anything that can turn a list of chat messages into a response.

    The judge is built on top of this rather than on top of a particular
    provider, so the same rubric, the same parsing and the same
    aggregation run against a hosted model and against a local one.
    """

    name: str

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant's reply to ``messages``."""
        ...


def lower_median(values: list[int]) -> int:
    """The lower median of a list of integers.

    For an even number of samples this is the smaller of the two middle
    values, not their average. The average would produce a score that
    belongs to no sample: three samples of 3 and three of 5 have an
    average of 4, and a report that says ``correctness: 4`` next to a
    verdict of "incorrect" is a report that contradicts itself.
    """
    if not values:
        raise JudgeError("No judge samples were produced.")

    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def majority(values: list[bool], ties_to: bool = False) -> bool:
    """The majority of a list of booleans.

    A tie is broken by ``ties_to``, which defaults to ``False``: the
    framework does not call an answer correct when the judge is exactly
    undecided about it. A tie is reported separately through the sample
    agreement, so nothing is lost by being conservative here.

    The tie matters more than it looks. The verdict and the aggregated
    correctness score have to agree, because a reader who sees
    ``correctness: 3`` and a threshold of 4 has to be able to work out the
    verdict from the score. A strict majority over the thresholded samples
    is the same rule as the lower median of the scores, so with the tie
    broken against "correct" the two can never contradict each other.
    """
    if not values:
        return False
    yes = sum(1 for value in values if value)
    no = len(values) - yes
    if yes == no:
        return ties_to
    return yes > no


def aggregate(raws: list[RawJudgement], threshold: int = 4) -> Judgement:
    """Fold the samples of one item into a single verdict.

    The scores are combined with the lower median, which ignores a single
    wild sample. The binary verdict is a strict majority over the same
    samples, derived from the correctness score by the same threshold
    everywhere.

    The lower median and a strict majority are the same rule, so the
    verdict and the reported correctness score always agree: a reader can
    work out the verdict from the score and the threshold without
    guessing. The median is the lower one rather than the average of the
    two middle samples, because an average would report a score that no
    sample produced.
    """
    if not raws:
        raise JudgeError("No judge samples were produced.")

    scores: dict[str, int] = {}
    for dimension in DIMENSIONS:
        values = [raw.scores.get(dimension, MIN_SCORE) for raw in raws]
        scores[dimension] = lower_median(values)

    verdicts = [raw.correctness >= threshold for raw in raws]
    verdict = majority(verdicts)
    agreement = sum(1 for value in verdicts if value == verdict) / len(verdicts)

    by_order: dict[str, list[int]] = {}
    for raw in raws:
        by_order.setdefault(raw.order, []).append(raw.correctness)
    order_verdicts = tuple(
        (order, lower_median(values) >= threshold)
        for order, values in by_order.items()
    )

    order_consistent: bool | None = None
    if len(order_verdicts) > 1:
        order_consistent = len({verdict for _, verdict in order_verdicts}) == 1

    # The reason comes from the sample that agrees with the verdict, so the
    # explanation shown to a reader is the explanation for the answer that
    # was reported.
    reason = ""
    for raw, value in zip(raws, verdicts):
        if value == verdict and raw.reason:
            reason = raw.reason
            break

    return Judgement(
        correct=verdict,
        correctness=scores["correctness"],
        relevance=scores["relevance"],
        groundedness=scores["groundedness"],
        completeness=scores["completeness"],
        reason=reason,
        n_samples=len(raws),
        sample_agreement=agreement,
        order_consistent=order_consistent,
        order_verdicts=order_verdicts,
    )

"""Measuring a judge against the hand labels.

A judge that scores well on its own is not evidence of anything. The only
evidence that matters is agreement with a person, and the only way to read
that agreement honestly is to put the judge and the token rule on the same
items and compare them.

The token rule is the baseline, not a straw man. It is the rule the
framework already uses to decide whether an answer is correct, and it is
right far more often than it is wrong. What it cannot do is tell that
"Berlin" is the right answer to "Which city is the office in?" when the
reference answer happens to read "Berlin, Germany". That is a limitation
of counting tokens, and it is exactly the kind of mistake a reader would
not make.

So the comparison is: both raters label the same set, and the set of
labels is the one thing neither rater produced.

Two subsets are reported. The whole set, and the items that were held out
while the rubric was written. A rubric written with a validation set open
in front of you fits that set better than it generalises, and the held-out
figure is the one that says whether the rubric generalises.
"""

from __future__ import annotations

from typing import Any

from rageval.judge.agreement import agreement, weighted_kappa
from rageval.judge.base import DIMENSIONS, JudgeItem, Judgement
from rageval.judge.judge import RubricJudge
from rageval.judge.labels import CORRECT, INCORRECT, LabelledItem
from rageval.metrics.answer import exact_match, numeric_match, token_recall

#: The two categories the binary verdict is reported over.
CATEGORIES = [INCORRECT, CORRECT]


def token_rule_verdict(
    answer: str,
    reference: str,
    threshold: float = 0.9,
) -> bool:
    """The rule the framework already uses to call an answer correct.

    Exact match, numeric match, or token recall above the threshold. This
    is what the judge has to beat.
    """
    if not reference:
        return False
    if exact_match(answer, reference) >= 1.0:
        return True
    numeric = numeric_match(answer, reference)
    if numeric == 1.0:
        return True
    return token_recall(answer, reference) >= threshold


def _item_from_label(item: LabelledItem) -> JudgeItem:
    return JudgeItem(
        question=item.question,
        answer=item.answer,
        context=item.context,
        reference_answer=item.reference_answer,
        test_id=item.test_id,
    )


def _subset_report(
    human: list[str],
    judge_labels: list[str],
    token_labels: list[str],
    correctness_scores: list[int],
    indexes: list[int],
) -> dict[str, Any]:
    """Agreement figures for one subset of the graded items."""
    if not indexes:
        return {}

    human_subset = [human[index] for index in indexes]
    judge_subset = [judge_labels[index] for index in indexes]
    token_subset = [token_labels[index] for index in indexes]

    judge_agreement = agreement(human_subset, judge_subset, categories=CATEGORIES)
    token_agreement = agreement(human_subset, token_subset, categories=CATEGORIES)
    judge_vs_token = agreement(judge_subset, token_subset, categories=CATEGORIES)

    # The human label is binary and the judge's correctness score is
    # ordinal, so the human label is mapped onto the ends of the scale and
    # the distance between them is what the weighted kappa charges for.
    human_scores = [5 if label == CORRECT else 1 for label in human_subset]
    score_kappa = weighted_kappa(
        human_scores,
        [correctness_scores[index] for index in indexes],
    )

    return {
        "n_items": len(indexes),
        "n_correct": sum(1 for label in human_subset if label == CORRECT),
        "judge": judge_agreement,
        "token_rule": token_agreement,
        "judge_vs_token_rule": judge_vs_token,
        "score_kappa": round(score_kappa, 4),
    }


def validate(
    judge: RubricJudge,
    items: list[LabelledItem],
    *,
    recall_threshold: float = 0.9,
    progress: Any = None,
) -> dict[str, Any]:
    """Grade every labelled item and measure the judge against the labels.

    ``progress`` is called with ``(done, total)`` after each item, so a
    long validation can report where it is rather than sitting silent.
    """
    human: list[str] = []
    judge_labels: list[str] = []
    token_labels: list[str] = []
    correctness_scores: list[int] = []
    rows: list[dict[str, Any]] = []

    sample_agreements: list[float] = []
    order_consistency: list[bool] = []
    failures = 0

    for position, item in enumerate(items, start=1):
        token_label = (
            CORRECT
            if token_rule_verdict(item.answer, item.reference_answer, recall_threshold)
            else INCORRECT
        )

        try:
            judgement: Judgement = judge.judge(_item_from_label(item))
        except Exception as error:  # noqa: BLE001 - one bad item is not the run
            failures += 1
            human.append(item.label)
            judge_labels.append(INCORRECT)
            token_labels.append(token_label)
            correctness_scores.append(1)
            rows.append(
                {
                    "test_id": item.test_id,
                    "question": item.question,
                    "answer": item.answer,
                    "human_label": item.label,
                    "judge_label": "error",
                    "token_rule_label": token_label,
                    "used_for_tuning": item.used_for_tuning,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if progress is not None:
                progress(position, len(items))
            continue

        judge_label = CORRECT if judgement.correct else INCORRECT

        human.append(item.label)
        judge_labels.append(judge_label)
        token_labels.append(token_label)
        correctness_scores.append(judgement.correctness)
        sample_agreements.append(judgement.sample_agreement)
        if judgement.order_consistent is not None:
            order_consistency.append(judgement.order_consistent)

        rows.append(
            {
                "test_id": item.test_id,
                "question": item.question,
                "answer": item.answer,
                "human_label": item.label,
                "judge_label": judge_label,
                "token_rule_label": token_label,
                "judge_scores": judgement.scores,
                "judge_reason": judgement.reason,
                "sample_agreement": round(judgement.sample_agreement, 4),
                "order_consistent": judgement.order_consistent,
                "token_recall": round(
                    token_recall(item.answer, item.reference_answer), 4
                ),
                "used_for_tuning": item.used_for_tuning,
                "note": item.note,
                "source": item.source,
            }
        )

        if progress is not None:
            progress(position, len(items))

    every = list(range(len(items)))
    held_out = [
        index
        for index, item in enumerate(items)
        if not item.used_for_tuning
    ]

    mean_scores = {
        dimension: round(
            sum(row.get("judge_scores", {}).get(dimension, 0) for row in rows)
            / max(len(rows), 1),
            4,
        )
        for dimension in DIMENSIONS
    }

    disagreements = [
        {
            "test_id": row["test_id"],
            "answer": row["answer"],
            "human_label": row["human_label"],
            "judge_label": row["judge_label"],
            "token_rule_label": row["token_rule_label"],
            "token_recall": row.get("token_recall"),
            "used_for_tuning": row.get("used_for_tuning", False),
            "note": row.get("note", ""),
        }
        for row in rows
        if row.get("judge_label") in (CORRECT, INCORRECT)
        and row["judge_label"] != row["human_label"]
        or row["token_rule_label"] != row["human_label"]
    ]

    return {
        "n_items": len(items),
        "n_failed": failures,
        "judge": {
            "name": judge.name,
            "n_samples": judge.n_samples,
            "swap_order": judge.swap_order,
            "threshold": judge.threshold,
            "mean_scores": mean_scores,
            "sample_agreement": round(
                sum(sample_agreements) / max(len(sample_agreements), 1), 4
            ),
            "order_consistency": (
                round(sum(order_consistency) / max(len(order_consistency), 1), 4)
                if order_consistency
                else None
            ),
        },
        "token_rule": {
            "recall_threshold": recall_threshold,
        },
        "all": _subset_report(
            human, judge_labels, token_labels, correctness_scores, every
        ),
        "held_out": _subset_report(
            human, judge_labels, token_labels, correctness_scores, held_out
        ),
        "disagreements": disagreements,
        "items": rows,
    }

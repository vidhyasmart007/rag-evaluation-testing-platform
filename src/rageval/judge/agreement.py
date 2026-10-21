"""Agreement between two raters.

The judge is only worth trusting if it agrees with a human on the cases
where the two differ, so the number that matters is not "how often was it
right" but "how often did it agree with the human, given that chance
agreement would have produced some agreement anyway". That is what
Cohen's kappa measures.

Cohen's kappa corrects for chance agreement, which matters here because
most answers in a validation set are correct and most raters say so. Two
raters who agree on ninety percent of a set where ninety percent of the
answers are correct would look excellent under raw agreement and would
show a kappa near zero. The correction is what stops a judge that always
says "correct" from looking like a good judge.

The unweighted kappa is the headline figure, because the binary verdict is
what the framework actually acts on. For the one to five scores, which are
ordinal, a quadratic weighted kappa is reported as well: two judges who
disagree by one point on a five point scale have not disagreed the way two
judges who disagree by four points have, and the unweighted statistic
cannot tell those apart.

The bands below are the conventional Landis and Koch ones. They are a
convention rather than a law, and they are here so that a reader does not
have to remember them.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Hashable, Sequence

#: The Landis and Koch bands for a kappa.
KAPPA_BANDS = (
    (0.00, "poor"),
    (0.01, "slight"),
    (0.21, "fair"),
    (0.41, "moderate"),
    (0.61, "substantial"),
    (0.81, "almost perfect"),
)


def interpret_kappa(kappa: float) -> str:
    """Name the band a kappa falls in."""
    if kappa < 0:
        return "poor"
    label = KAPPA_BANDS[0][1]
    for threshold, name in KAPPA_BANDS:
        if kappa >= threshold:
            label = name
    return label


def _confusion(
    first: Sequence[Hashable],
    second: Sequence[Hashable],
    categories: Sequence[Hashable],
) -> dict[Hashable, dict[Hashable, int]]:
    counts: dict[Hashable, dict[Hashable, int]] = {
        row: {column: 0 for column in categories} for row in categories
    }
    for left, right in zip(first, second):
        counts[left][right] += 1
    return counts


def cohen_kappa(
    first: Sequence[Hashable],
    second: Sequence[Hashable],
) -> float:
    """Cohen's kappa between two sets of labels.

    A kappa of one means the two raters agreed on every item. A kappa of
    zero means they agreed exactly as often as two raters guessing from
    the same marginals would have.
    """
    if len(first) != len(second):
        raise ValueError(
            f"Both raters must label the same number of items "
            f"({len(first)} and {len(second)} given)."
        )
    if not first:
        raise ValueError("Agreement needs at least one item.")

    total = len(first)
    categories = sorted(set(first) | set(second), key=str)
    first_counts = Counter(first)
    second_counts = Counter(second)

    observed = sum(1 for a, b in zip(first, second) if a == b) / total
    expected = sum(
        (first_counts.get(category, 0) / total)
        * (second_counts.get(category, 0) / total)
        for category in categories
    )

    if 1.0 - expected == 0.0:
        # Both raters used a single category, so chance agreement is
        # already one and the statistic is undefined. Perfect agreement is
        # still perfect agreement, and disagreement is not.
        return 1.0 if observed == 1.0 else 0.0

    return (observed - expected) / (1.0 - expected)


def weighted_kappa(
    first: Sequence[Any],
    second: Sequence[Any],
    weights: str = "quadratic",
) -> float:
    """Cohen's kappa with a penalty for the size of the disagreement.

    Only meaningful for ordinal labels such as the one to five scores.
    The labels are placed in the order they are given in, so the scores
    have to be passed in their natural order.
    """
    if weights not in ("quadratic", "linear"):
        raise ValueError(f"Unknown weighting '{weights}'. Use quadratic or linear.")

    if len(first) != len(second):
        raise ValueError(
            f"Both raters must label the same number of items "
            f"({len(first)} and {len(second)} given)."
        )
    if not first:
        raise ValueError("Agreement needs at least one item.")

    total = len(first)
    categories = sorted(set(first) | set(second), key=float)
    index = {category: position for position, category in enumerate(categories)}
    span = max(len(categories) - 1, 1)

    def weight(left: Hashable, right: Hashable) -> float:
        distance = abs(index[left] - index[right])
        if weights == "linear":
            return distance / span
        return (distance / span) ** 2

    observed_sum = sum(
        weight(a, b) for a, b in zip(first, second)
    )
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected_sum = sum(
        weight(left, right)
        * (first_counts.get(left, 0) / total)
        * (second_counts.get(right, 0) / total)
        for left in categories
        for right in categories
    )

    if expected_sum == 0.0:
        return 1.0 if observed_sum == 0.0 else 0.0
    return 1.0 - (observed_sum / total) / expected_sum


def agreement(
    first: Sequence[Hashable],
    second: Sequence[Hashable],
    categories: Sequence[Hashable] | None = None,
) -> dict[str, Any]:
    """Everything worth knowing about how two raters agreed.

    The raw agreement is reported next to the kappa because the two answer
    different questions, and a reader needs both: the kappa says how much
    of the agreement is more than chance, and the raw number says how much
    of the set the two raters actually examined.
    """
    if len(first) != len(second):
        raise ValueError(
            f"Both raters must label the same number of items "
            f"({len(first)} and {len(second)} given)."
        )
    if not first:
        raise ValueError("Agreement needs at least one item.")

    if categories is None:
        categories = sorted(set(first) | set(second), key=str)

    total = len(first)
    observed = sum(1 for a, b in zip(first, second) if a == b) / total
    kappa = cohen_kappa(first, second)

    return {
        "n": total,
        "observed_agreement": round(observed, 4),
        "cohen_kappa": round(kappa, 4),
        "interpretation": interpret_kappa(kappa),
        "categories": [str(category) for category in categories],
        "confusion": {
            str(row): {str(column): count for column, count in cells.items()}
            for row, cells in _confusion(first, second, categories).items()
        },
    }

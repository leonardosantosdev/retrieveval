"""Retrieval metrics.

These are pure functions over a *judged ranking*: a sequence of relevance
grades in rank order, where ``grades[0]`` is the top-ranked result. They know
nothing about chunks, documents or retrievers -- turning search results into
grades is the job of :mod:`retrieveval.evaluation.judgments`.

V1 uses **binary relevance**: every grade is ``0.0`` or ``1.0``. The ideal
ranking used to normalise nDCG is derived from ``num_relevant`` on that
assumption.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["recall_at_k", "reciprocal_rank", "ndcg_at_k", "dcg"]


def _validate(grades: Sequence[float], num_relevant: int, k: int | None) -> None:
    if num_relevant <= 0:
        raise ValueError(f"num_relevant must be positive, got {num_relevant}")
    if k is not None and k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    for i, g in enumerate(grades):
        if g not in (0, 1):
            raise ValueError(
                f"grades must be binary (0 or 1) in V1; got {g!r} at position {i}"
            )


def recall_at_k(grades: Sequence[float], num_relevant: int, k: int) -> float:
    """Fraction of the relevant items that appear in the top ``k`` results.

    Ranges from 0.0 (nothing relevant retrieved) to 1.0 (every relevant item
    retrieved within the cutoff).
    """
    _validate(grades, num_relevant, k)
    return sum(grades[:k]) / num_relevant


def reciprocal_rank(grades: Sequence[float]) -> float:
    """``1 / rank`` of the first relevant result, or 0.0 if there is none.

    Averaged over a query set this is MRR. It answers "how far down the list
    does the user have to look before hitting something relevant?".
    """
    for index, grade in enumerate(grades):
        if grade:
            return 1.0 / (index + 1)
    return 0.0


def dcg(grades: Sequence[float]) -> float:
    """Discounted Cumulative Gain, with the standard ``1 / log2(rank + 1)`` discount."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(grades: Sequence[float], num_relevant: int, k: int) -> float:
    """DCG@k normalised by the DCG of the best ranking achievable at that cutoff.

    The ideal ranking places ``min(num_relevant, k)`` relevant items in the top
    positions, so the result is always in ``[0.0, 1.0]``.
    """
    _validate(grades, num_relevant, k)
    ideal = [1.0] * min(num_relevant, k)
    idcg = dcg(ideal)
    if idcg == 0.0:  # unreachable while num_relevant > 0 and k > 0, but explicit
        return 0.0
    return dcg(grades[:k]) / idcg

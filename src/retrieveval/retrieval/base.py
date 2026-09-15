"""The retriever abstraction every strategy is benchmarked through.

The benchmark runner only ever calls :meth:`Retriever.index`,
:meth:`Retriever.search` and :meth:`Retriever.describe`. That is what lets
BM25, dense, hybrid and reranked retrieval be compared without the evaluation
layer knowing anything about embeddings, postings lists or cross-encoders.

``describe()`` exists so each strategy can report its own parameters into the
results file. Reproducibility then does not require the reporting layer to
grow a special case per retriever.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..models import Chunk, SearchResult

__all__ = ["Retriever", "rank_top_k", "build_results"]


@runtime_checkable
class Retriever(Protocol):
    """A retrieval strategy that can be indexed once and queried many times."""

    name: str
    """Human-readable label used in reports (e.g. ``"BM25"``)."""

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build whatever internal structures the strategy needs."""
        ...

    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return at most ``k`` results, best first, ranked from 1."""
        ...

    def describe(self) -> dict[str, Any]:
        """Parameters that define this configuration, for the results record."""
        ...


def rank_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` highest scores, best first.

    Ties are broken by ascending index so that repeated runs over an unchanged
    corpus produce identical rankings -- without this, tied BM25 or fusion
    scores would make benchmark output jitter between runs.
    """
    total = len(scores)
    k = min(k, total)
    if k <= 0:
        return np.empty(0, dtype=int)

    if k < total:
        candidates = np.argpartition(-scores, k - 1)[:k]
        # argpartition picks an arbitrary k among equal scores, so the cut can
        # land in the middle of a tied group and the tie-break below would only
        # order whichever members introselect happened to choose. Pulling in
        # every index sharing the boundary score puts the whole tied group in
        # front of the sort, which is what makes the ordering reproducible.
        boundary = scores[candidates].min()
        candidates = np.union1d(candidates, np.flatnonzero(scores == boundary))
    else:
        candidates = np.arange(total)

    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order][:k]


def build_results(
    chunks: Sequence[Chunk],
    scores: np.ndarray,
    k: int,
    *,
    min_score: float | None = None,
) -> list[SearchResult]:
    """Assemble the top ``k`` scored chunks into ranked results.

    ``min_score`` drops results at or below a threshold. Lexical retrieval uses
    it to exclude chunks that share no term with the query: such a chunk is not
    a weak match but a non-match, and letting it occupy a rank would credit the
    retriever for finding a document it did not actually match.
    """
    indices = rank_top_k(scores, k)
    results: list[SearchResult] = []
    for rank, index in enumerate(indices, start=1):
        score = float(scores[index])
        if min_score is not None and score <= min_score:
            break
        results.append(SearchResult(chunk=chunks[index], score=score, rank=len(results) + 1))
    return results

"""Hybrid retrieval by Reciprocal Rank Fusion.

RRF combines rankings, not scores. That matters here: a BM25 score and a
cosine similarity live on different scales with different distributions, and
any attempt to compare them directly requires a normalisation step whose
constants would themselves become untested tuning knobs. RRF sidesteps the
question by using only each retriever's rank ordering:

    score(chunk) = sum over retrievers of 1 / (rrf_k + rank)

Both retrievers are queried to ``depth`` rather than to the final ``k``. A
document ranked 30th by BM25 and 25th by dense is exactly the kind of
consensus RRF is meant to promote, and it would be invisible if each list were
truncated at 10 first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..models import Chunk, SearchResult
from .base import Retriever

__all__ = ["HybridRetriever", "reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], rrf_k: int = 60
) -> list[SearchResult]:
    """Fuse ranked lists into one, best first.

    Ties are broken by the best rank any input list gave the chunk, and then by
    chunk id, so the output is deterministic regardless of dict ordering.
    """
    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks: dict[str, Chunk] = {}

    for ranking in rankings:
        for result in ranking:
            key = result.chunk.id
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + result.rank)
            best_rank[key] = min(best_rank.get(key, result.rank), result.rank)
            chunks.setdefault(key, result.chunk)

    order = sorted(fused, key=lambda key: (-fused[key], best_rank[key], key))
    return [
        SearchResult(chunk=chunks[key], score=fused[key], rank=rank)
        for rank, key in enumerate(order, start=1)
    ]


class HybridRetriever:
    """BM25 and dense retrieval fused with RRF.

    Sub-retrievers are supplied already constructed. When they are also already
    indexed -- which is how the benchmark builds them, so that the corpus is
    embedded once and shared -- ``index()`` need not be called at all.
    """

    name = "Hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        rrf_k: int = 60,
        depth: int = 100,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("hybrid retrieval needs at least two retrievers to fuse")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.depth = depth

    def index(self, chunks: Sequence[Chunk]) -> None:
        for retriever in self.retrievers:
            retriever.index(chunks)

    def search(self, query: str, k: int) -> list[SearchResult]:
        pool = max(k, self.depth)
        rankings = [retriever.search(query, pool) for retriever in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.rrf_k)[:k]

    def describe(self) -> dict[str, Any]:
        return {
            "retriever": "hybrid",
            "fusion": "reciprocal_rank_fusion",
            "rrf_k": self.rrf_k,
            "depth": self.depth,
            "components": [retriever.describe() for retriever in self.retrievers],
        }

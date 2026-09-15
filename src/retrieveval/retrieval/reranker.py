"""Cross-encoder reranking of another retriever's candidates.

A cross-encoder reads the query and a candidate together, so it can judge
relevance far more precisely than a bi-encoder that embedded each side
independently -- and it costs one model forward pass per candidate, which is
why it can only ever be applied to a shortlist.

Reranking is modelled as a retriever that wraps another retriever. That keeps
it comparable in the benchmark: "Hybrid" and "Hybrid + reranker" are two
strategies measured through the identical interface, so the quality gain and
the latency it costs are read off the same table.

The candidate pool is the parameter that matters most. A pool of 50 gives the
cross-encoder more to work with than a pool of 10 and costs roughly five times
as much per query; recall of the *base* retriever at the pool size is a hard
ceiling on what reranking can achieve, since it can only reorder what it is
given.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..models import Chunk, SearchResult
from .base import Retriever, rank_top_k

__all__ = ["RerankingRetriever"]


def _load_cross_encoder(model_name: str):
    from .dense import ModelUnavailableError

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ModelUnavailableError(
            "reranking requires the `sentence-transformers` package"
        ) from exc
    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        raise ModelUnavailableError(
            f"could not load the reranking model {model_name!r}: {exc}\n"
            "The first run downloads the model and needs network access; afterwards it is "
            "served from the local Hugging Face cache. Check the model name, or disable "
            "reranking with `reranker.enabled: false`."
        ) from exc


class RerankingRetriever:
    """Wraps a base retriever and reorders its top candidates with a cross-encoder."""

    def __init__(
        self,
        base: Retriever,
        model_name: str,
        *,
        candidates: int = 50,
        batch_size: int = 32,
        model: Any = None,
    ) -> None:
        self.base = base
        self.model_name = model_name
        self.candidates = candidates
        self.batch_size = batch_size
        self.name = f"{base.name} + reranker"
        # An already-loaded cross-encoder can be passed in so that several
        # reranked strategies share one copy in memory.
        self._model: Any = model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    def load(self) -> None:
        """Force model loading, so its cost is attributed to indexing rather
        than to whichever query happened to run first."""
        _ = self.model

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.base.index(chunks)
        self.load()

    def search(self, query: str, k: int) -> list[SearchResult]:
        pool = self.base.search(query, max(k, self.candidates))
        if not pool:
            return []

        scores = np.asarray(
            self.model.predict(
                [(query, result.chunk.text) for result in pool],
                batch_size=self.batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        ).reshape(-1)

        return [
            SearchResult(chunk=pool[index].chunk, score=float(scores[index]), rank=rank)
            for rank, index in enumerate(rank_top_k(scores, k), start=1)
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "retriever": "reranked",
            "model": self.model_name,
            "candidates": self.candidates,
            "batch_size": self.batch_size,
            "base": self.base.describe(),
        }

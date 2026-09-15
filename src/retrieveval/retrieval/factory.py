"""Building the set of strategies a benchmark will compare.

Composition lives here, and so does index timing, because the two are the same
question: hybrid retrieval reuses the very BM25 and dense indexes that the
standalone strategies use, and only the code that assembles them knows what
was shared.

``index_time_ms`` is therefore reported as *the cost of making that strategy
queryable from cold* -- the sum of the components it depends on. The figures
across strategies overlap by design and are not meant to be added up.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..config import Config
from ..evaluation.runner import Strategy
from ..models import Chunk
from .base import Retriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .reranker import RerankingRetriever

__all__ = ["build_strategies"]

StepCallback = Callable[[str], None]

DISPLAY_NAMES = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}


@dataclass
class _Component:
    retriever: Retriever
    build_ms: float


def _timed(label: str, build: Callable[[], Retriever], on_step: StepCallback | None) -> _Component:
    if on_step is not None:
        on_step(label)
    started = time.perf_counter()
    retriever = build()
    return _Component(retriever, (time.perf_counter() - started) * 1000)


def build_strategies(
    chunks: Sequence[Chunk],
    config: Config,
    *,
    on_step: StepCallback | None = None,
) -> list[Strategy]:
    """Construct and index every strategy the configuration asks for."""
    needs_bm25 = "bm25" in config.retrievers or "hybrid" in config.retrievers
    needs_dense = "dense" in config.retrievers or "hybrid" in config.retrievers

    components: dict[str, _Component] = {}

    if needs_bm25:
        def build_bm25() -> Retriever:
            retriever = BM25Retriever()
            retriever.index(chunks)
            return retriever

        components["bm25"] = _timed("Building BM25 index", build_bm25, on_step)

    if needs_dense:
        def build_dense() -> Retriever:
            retriever = DenseRetriever(
                config.dense.model,
                batch_size=config.dense.batch_size,
                cache_dir=config.dense.cache_dir,
            )
            retriever.index(chunks)
            return retriever

        components["dense"] = _timed(
            f"Embedding corpus with {config.dense.model}", build_dense, on_step
        )

    base_strategies: dict[str, Strategy] = {}
    for name in config.retrievers:
        if name == "hybrid":
            # The components are already indexed; fusion adds no index cost of
            # its own, only the two it depends on.
            retriever: Retriever = HybridRetriever(
                [components["bm25"].retriever, components["dense"].retriever],
                rrf_k=config.hybrid.rrf_k,
                depth=config.hybrid.depth,
            )
            build_ms = components["bm25"].build_ms + components["dense"].build_ms
        else:
            component = components[name]
            retriever = component.retriever
            build_ms = component.build_ms

        base_strategies[name] = Strategy(
            name=DISPLAY_NAMES[name],
            retriever=retriever,
            index_time_ms=build_ms,
            configuration=retriever.describe(),
        )

    strategies = list(base_strategies.values())

    if config.reranker.enabled:
        applies_to = config.reranker.applies_to or config.retrievers
        shared_model = None
        model_load_ms = 0.0
        for name in config.retrievers:
            if name not in applies_to:
                continue
            base = base_strategies[name]
            reranking = RerankingRetriever(
                base.retriever,
                config.reranker.model,
                candidates=config.reranker.candidates,
                model=shared_model,
            )
            if shared_model is None:
                if on_step is not None:
                    on_step(f"Loading reranker {config.reranker.model}")
                started = time.perf_counter()
                reranking.load()
                model_load_ms = (time.perf_counter() - started) * 1000
                shared_model = reranking.model
            strategies.append(
                Strategy(
                    name=reranking.name,
                    retriever=reranking,
                    index_time_ms=base.index_time_ms + model_load_ms,
                    configuration=reranking.describe(),
                )
            )

    return strategies

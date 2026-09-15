"""The benchmark loop.

This module knows how to time and score a retriever. It does not know what
kind of retriever it is holding -- everything it needs comes through the
:class:`~retrieveval.retrieval.base.Retriever` interface and the strategy's own
``describe()``. Adding a retrieval strategy therefore requires no change here.

Indexing cost is measured by the strategy builder rather than here, because
strategies share components (hybrid reuses the same BM25 and dense indexes)
and only the builder knows which work was shared.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import EvaluationConfig
from ..models import EvalQuery
from ..retrieval.base import Retriever
from .judgments import count_relevant, grade_ranking
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from .results import LatencyStats, QueryResult, StrategyResult

__all__ = ["Strategy", "evaluate_strategy", "run_benchmark", "metric_names"]

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class Strategy:
    """A named, already-indexed retrieval configuration ready to be evaluated."""

    name: str
    retriever: Retriever
    index_time_ms: float
    configuration: dict[str, Any]


def metric_names(evaluation: EvaluationConfig) -> list[str]:
    """Metric keys in report order, e.g. ``["recall@1", ..., "mrr", "ndcg@1", ...]``."""
    return (
        [f"recall@{k}" for k in evaluation.k]
        + ["mrr"]
        + [f"ndcg@{k}" for k in evaluation.k]
    )


def _score_query(
    grades: list[float], num_relevant: int, evaluation: EvaluationConfig
) -> dict[str, float]:
    scores = {f"recall@{k}": recall_at_k(grades, num_relevant, k) for k in evaluation.k}
    scores["mrr"] = reciprocal_rank(grades)
    scores.update({f"ndcg@{k}": ndcg_at_k(grades, num_relevant, k) for k in evaluation.k})
    return scores


def _first_relevant_rank(grades: Sequence[float]) -> int | None:
    for index, grade in enumerate(grades, start=1):
        if grade:
            return index
    return None


def evaluate_strategy(
    strategy: Strategy,
    queries: Sequence[EvalQuery],
    evaluation: EvaluationConfig,
    *,
    depth: int | None = None,
    warm_up: bool = True,
    on_query: ProgressCallback | None = None,
) -> StrategyResult:
    """Run every query against one strategy, timing and scoring each.

    ``depth`` is how many results to request per query; it defaults to the
    largest configured cutoff, since nothing beyond that can affect a metric.

    ``warm_up`` runs the first query once without recording it. Lazily
    initialised backends (a torch model's first forward pass in particular)
    would otherwise charge one-off setup cost to whichever query happened to
    run first, which is not the latency a served query would see.
    """
    if not queries:
        raise ValueError("cannot evaluate an empty query set")

    requested = depth if depth is not None else evaluation.max_k
    if warm_up:
        strategy.retriever.search(queries[0].query, requested)

    per_query: list[QueryResult] = []
    latencies: list[float] = []
    total = len(queries)

    for index, query in enumerate(queries, start=1):
        started = time.perf_counter()
        results = strategy.retriever.search(query.query, requested)
        elapsed_ms = (time.perf_counter() - started) * 1000

        grades = grade_ranking(results, query)
        scores = _score_query(grades, count_relevant(query), evaluation)

        latencies.append(elapsed_ms)
        per_query.append(
            QueryResult(
                query=query.query,
                metrics=scores,
                first_relevant_rank=_first_relevant_rank(grades),
                latency_ms=elapsed_ms,
            )
        )
        if on_query is not None:
            on_query(strategy.name, index, total)

    keys = metric_names(evaluation)
    aggregate = {
        key: sum(result.metrics[key] for result in per_query) / total for key in keys
    }
    return StrategyResult(
        name=strategy.name,
        metrics=aggregate,
        latency=LatencyStats.from_samples(latencies),
        index_time_ms=strategy.index_time_ms,
        configuration=strategy.configuration,
        queries=per_query,
    )


def run_benchmark(
    strategies: Sequence[Strategy],
    queries: Sequence[EvalQuery],
    evaluation: EvaluationConfig,
    *,
    depth: int | None = None,
    on_query: ProgressCallback | None = None,
) -> list[StrategyResult]:
    """Evaluate every strategy over the same query set, in configuration order."""
    return [
        evaluate_strategy(strategy, queries, evaluation, depth=depth, on_query=on_query)
        for strategy in strategies
    ]

"""Result types produced by a benchmark run.

Kept separate from the runner so the reporting layer can consume results
without importing the machinery that produced them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = ["LatencyStats", "QueryResult", "StrategyResult", "BenchmarkRun"]


@dataclass(frozen=True)
class LatencyStats:
    """Query-time latency, in milliseconds.

    Percentiles use linear interpolation between samples. With a small
    evaluation set p95 is an interpolation between the top two or three
    queries, so treat it as indicative rather than precise -- the sample count
    is recorded alongside it for exactly that reason.
    """

    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    samples: int

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> LatencyStats:
        if not samples:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        values = np.asarray(samples, dtype=np.float64)
        return cls(
            mean_ms=float(values.mean()),
            p50_ms=float(np.percentile(values, 50)),
            p95_ms=float(np.percentile(values, 95)),
            min_ms=float(values.min()),
            max_ms=float(values.max()),
            samples=len(samples),
        )


@dataclass(frozen=True)
class QueryResult:
    """Per-query outcome, kept so a low aggregate score can be traced to the
    queries responsible for it."""

    query: str
    metrics: dict[str, float]
    first_relevant_rank: int | None
    """Rank of the first relevant result, or None if none was retrieved."""
    latency_ms: float


@dataclass(frozen=True)
class StrategyResult:
    """Everything measured about one retrieval configuration."""

    name: str
    metrics: dict[str, float]
    latency: LatencyStats
    index_time_ms: float
    configuration: dict[str, Any]
    queries: list[QueryResult] = field(default_factory=list)

    def metric(self, key: str) -> float:
        return self.metrics[key]


@dataclass(frozen=True)
class BenchmarkRun:
    """A complete run: what was measured, and the configuration that produced it."""

    timestamp: str
    corpus: dict[str, Any]
    dataset: dict[str, Any]
    config: dict[str, Any]
    strategies: list[StrategyResult]
    environment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

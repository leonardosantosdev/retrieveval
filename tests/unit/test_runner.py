"""Benchmark runner tests, driven by a scripted retriever.

The runner must work for any retriever, so these tests deliberately use one
that is not BM25 or dense: if the runner ever grows knowledge of a concrete
strategy, these tests stop passing.
"""

import pytest

from retrieveval.config import EvaluationConfig
from retrieveval.evaluation.runner import Strategy, evaluate_strategy, metric_names, run_benchmark
from retrieveval.models import Chunk, EvalQuery, SearchResult


class ScriptedRetriever:
    """Returns a fixed ranking of document ids per query."""

    name = "Scripted"

    def __init__(self, script: dict[str, list[str]]):
        self.script = script
        self.calls: list[tuple[str, int]] = []

    def index(self, chunks):
        pass

    def search(self, query, k):
        self.calls.append((query, k))
        return [
            SearchResult(
                chunk=Chunk(f"{d}::chunk_0", d, d, 0), score=1.0 / rank, rank=rank
            )
            for rank, d in enumerate(self.script.get(query, [])[:k], start=1)
        ]

    def describe(self):
        return {"retriever": "scripted"}


def strategy(script, name="Scripted"):
    return Strategy(name=name, retriever=ScriptedRetriever(script), index_time_ms=5.0,
                    configuration={"retriever": "scripted"})


EVALUATION = EvaluationConfig(k=(1, 3))


class TestMetricNames:
    def test_report_order(self):
        assert metric_names(EvaluationConfig(k=(1, 5))) == [
            "recall@1", "recall@5", "mrr", "ndcg@1", "ndcg@5",
        ]


class TestEvaluateStrategy:
    def test_averages_metrics_across_queries(self):
        # One query is answered perfectly, the other not at all.
        queries = [
            EvalQuery("hit", frozenset({"a.md"})),
            EvalQuery("miss", frozenset({"b.md"})),
        ]
        result = evaluate_strategy(
            strategy({"hit": ["a.md"], "miss": ["z.md"]}), queries, EVALUATION
        )
        assert result.metrics["recall@1"] == pytest.approx(0.5)
        assert result.metrics["mrr"] == pytest.approx(0.5)

    def test_records_each_query_so_failures_can_be_traced(self):
        queries = [
            EvalQuery("hit", frozenset({"a.md"})),
            EvalQuery("miss", frozenset({"b.md"})),
        ]
        result = evaluate_strategy(
            strategy({"hit": ["a.md"], "miss": ["z.md"]}), queries, EVALUATION
        )
        assert [q.query for q in result.queries] == ["hit", "miss"]
        assert [q.first_relevant_rank for q in result.queries] == [1, None]

    def test_requests_the_largest_configured_cutoff(self):
        strat = strategy({"q": ["a.md"]})
        evaluate_strategy(strat, [EvalQuery("q", frozenset({"a.md"}))], EvaluationConfig(k=(1, 20)))
        assert all(k == 20 for _, k in strat.retriever.calls)

    def test_depth_can_be_overridden(self):
        strat = strategy({"q": ["a.md"]})
        evaluate_strategy(
            strat, [EvalQuery("q", frozenset({"a.md"}))], EVALUATION, depth=50, warm_up=False
        )
        assert strat.retriever.calls == [("q", 50)]

    def test_the_warm_up_query_is_not_measured(self):
        strat = strategy({"q": ["a.md"]})
        result = evaluate_strategy(strat, [EvalQuery("q", frozenset({"a.md"}))], EVALUATION)
        assert len(strat.retriever.calls) == 2  # warm-up + the measured query
        assert result.latency.samples == 1

    def test_warm_up_can_be_disabled(self):
        strat = strategy({"q": ["a.md"]})
        evaluate_strategy(
            strat, [EvalQuery("q", frozenset({"a.md"}))], EVALUATION, warm_up=False
        )
        assert len(strat.retriever.calls) == 1

    def test_carries_index_time_and_configuration_through(self):
        result = evaluate_strategy(
            strategy({"q": ["a.md"]}), [EvalQuery("q", frozenset({"a.md"}))], EVALUATION
        )
        assert result.index_time_ms == 5.0
        assert result.configuration == {"retriever": "scripted"}

    def test_latency_is_recorded_per_query(self):
        queries = [EvalQuery(f"q{i}", frozenset({"a.md"})) for i in range(4)]
        result = evaluate_strategy(
            strategy({f"q{i}": ["a.md"] for i in range(4)}), queries, EVALUATION
        )
        assert result.latency.samples == 4
        assert result.latency.min_ms <= result.latency.p50_ms <= result.latency.max_ms

    def test_an_empty_query_set_is_an_error(self):
        with pytest.raises(ValueError, match="empty query set"):
            evaluate_strategy(strategy({}), [], EVALUATION)

    def test_a_retriever_returning_nothing_scores_zero(self):
        result = evaluate_strategy(
            strategy({"q": []}), [EvalQuery("q", frozenset({"a.md"}))], EVALUATION
        )
        assert result.metrics["mrr"] == 0.0
        assert result.metrics["recall@3"] == 0.0


class TestRunBenchmark:
    def test_evaluates_every_strategy_in_order(self):
        queries = [EvalQuery("q", frozenset({"a.md"}))]
        results = run_benchmark(
            [strategy({"q": ["a.md"]}, "First"), strategy({"q": ["z.md"]}, "Second")],
            queries,
            EVALUATION,
        )
        assert [r.name for r in results] == ["First", "Second"]
        assert results[0].metrics["recall@1"] == 1.0
        assert results[1].metrics["recall@1"] == 0.0

"""Reporting tests: formatting, the summary's claims, and the JSON artifact."""

import json

import pytest
from rich.console import Console

from retrieveval.config import Config
from retrieveval.evaluation.results import LatencyStats, StrategyResult
from retrieveval.reporting.report import (
    build_run,
    format_duration,
    print_report,
    quality_table,
    summary_lines,
    render_markdown,
    write_artifacts,
    write_json,
    write_markdown,
)

METRIC_KEYS = ["recall@5", "mrr", "ndcg@10"]


def result(name, ndcg, p95, index_ms=1000.0):
    return StrategyResult(
        name=name,
        metrics={"recall@5": ndcg, "mrr": ndcg, "ndcg@10": ndcg},
        latency=LatencyStats(p95, p95, p95, p95, p95, 10),
        index_time_ms=index_ms,
        configuration={"retriever": name.lower()},
    )


class TestFormatDuration:
    @pytest.mark.parametrize(
        "milliseconds,expected",
        [
            (0.064, "0.06ms"),
            (2.85, "2.9ms"),
            (41.2, "41ms"),
            (1450.0, "1.45s"),
            (18_100.0, "18.1s"),
        ],
    )
    def test_precision_adapts_to_magnitude(self, milliseconds, expected):
        assert format_duration(milliseconds) == expected

    def test_sub_millisecond_latency_is_not_rounded_away(self):
        # A BM25 query on a small corpus really is this fast; "0ms" would hide
        # the three orders of magnitude between it and a reranked query.
        assert format_duration(0.09) != "0ms"


class TestSummaryLines:
    results = [
        result("BM25", 0.69, 0.1),
        result("Dense", 0.96, 9.0),
        result("Hybrid", 0.92, 10.0),
        result("Hybrid + reranker", 0.98, 28.0),
    ]

    def test_names_the_quality_leader_with_its_cost(self):
        line = summary_lines(self.results, "ndcg@10")[0]
        assert "Hybrid + reranker" in line
        assert "0.980" in line
        assert "28ms" in line

    def test_names_the_latency_leader_with_its_quality(self):
        line = summary_lines(self.results, "ndcg@10")[1]
        assert "BM25" in line
        assert "0.690" in line

    def test_names_the_best_unreranked_strategy_separately(self):
        lines = summary_lines(self.results, "ndcg@10")
        assert any("without reranking" in line and "Dense" in line for line in lines)

    def test_spells_the_metric_out_in_prose(self):
        assert "nDCG@10" in summary_lines(self.results, "ndcg@10")[0]

    def test_no_unreranked_line_when_nothing_was_reranked(self):
        lines = summary_lines([result("BM25", 0.7, 1.0), result("Dense", 0.9, 5.0)], "ndcg@10")
        assert not any("without reranking" in line for line in lines)

    def test_no_unreranked_line_when_the_leader_is_already_unreranked(self):
        results = [result("Dense", 0.99, 9.0), result("Hybrid + reranker", 0.90, 28.0)]
        assert not any("without reranking" in line for line in summary_lines(results, "ndcg@10"))

    def test_makes_no_overall_recommendation(self):
        text = " ".join(summary_lines(self.results, "ndcg@10")).lower()
        for word in ("recommend", "best overall", "you should", "winner"):
            assert word not in text

    def test_handles_a_single_strategy(self):
        assert len(summary_lines([result("BM25", 0.7, 1.0)], "ndcg@10")) == 2

    def test_handles_no_results(self):
        assert summary_lines([], "ndcg@10") == []


class TestQualityTable:
    def test_lists_every_strategy_and_metric(self):
        table = quality_table([result("BM25", 0.7, 1.0)], METRIC_KEYS)
        assert [column.header for column in table.columns] == ["Strategy", "R@5", "MRR", "N@10"]
        assert table.row_count == 1

    def test_fits_in_an_eighty_column_terminal(self):
        # The default report must be readable without a wide terminal.
        console = Console(width=80, record=True, force_terminal=False)
        results = [
            result("BM25", 0.69, 0.1),
            result("Hybrid + reranker", 0.98, 28.0),
        ]
        keys = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@1", "ndcg@5", "ndcg@10"]
        for name in keys:
            for r in results:
                r.metrics.setdefault(name, 0.5)
        print_report(console, results, keys, "ndcg@10")
        text = console.export_text()
        assert "…" not in text, "a column was truncated at 80 columns"
        assert max(len(line) for line in text.splitlines()) <= 80


class TestWriteJson:
    def test_writes_a_timestamped_artifact(self, tmp_path):
        run = build_run(
            [result("BM25", 0.7, 1.0)],
            Config(),
            corpus={"path": "docs", "documents": 8, "chunks": 21},
            dataset={"path": "eval.json", "queries": 18},
        )
        path = write_json(run, tmp_path)
        assert path.parent == tmp_path
        assert path.name.endswith("Z.json")

    def test_the_artifact_records_the_configuration_that_produced_it(self, tmp_path):
        run = build_run(
            [result("BM25", 0.7, 1.0)],
            Config(),
            corpus={"path": "docs", "documents": 8, "chunks": 21},
            dataset={"path": "eval.json", "queries": 18},
        )
        record = json.loads(write_json(run, tmp_path).read_text(encoding="utf-8"))
        assert record["config"]["chunking"] == {"size": 500, "overlap": 50}
        assert record["config"]["dense"]["model"]
        assert record["corpus"]["chunks"] == 21
        assert record["dataset"]["queries"] == 18
        assert record["environment"]["python"]
        assert record["strategies"][0]["metrics"]["ndcg@10"] == 0.7
        assert record["strategies"][0]["latency"]["p95_ms"] == 1.0

    def test_creates_the_output_directory(self, tmp_path):
        run = build_run([result("BM25", 0.7, 1.0)], Config(), corpus={}, dataset={})
        path = write_json(run, tmp_path / "nested" / "results")
        assert path.exists()


class TestRenderMarkdown:
    results = [
        result("BM25", 0.69, 0.1),
        result("Hybrid + reranker", 0.98, 28.0),
    ]

    def build(self):
        return build_run(
            self.results,
            Config(),
            corpus={"path": "docs", "documents": 8, "chunks": 21},
            dataset={"path": "eval.json", "queries": 18},
        )

    def test_contains_a_quality_table_with_the_best_value_bolded(self):
        text = render_markdown(self.build(), METRIC_KEYS, "ndcg@10")
        assert "| Strategy | R@5 | MRR | N@10 |" in text
        assert "**0.980**" in text
        assert "0.690" in text and "**0.690**" not in text

    def test_contains_a_latency_table(self):
        text = render_markdown(self.build(), METRIC_KEYS, "ndcg@10")
        assert "| Strategy | Index | Avg | p50 | p95 | Max |" in text
        assert "28ms" in text

    def test_contains_the_same_summary_lines_as_the_console(self):
        text = render_markdown(self.build(), METRIC_KEYS, "ndcg@10")
        for line in summary_lines(self.results, "ndcg@10"):
            assert line in text

    def test_makes_no_overall_recommendation(self):
        text = render_markdown(self.build(), METRIC_KEYS, "ndcg@10").lower()
        for word in ("recommend", "best overall", "you should", "winner"):
            assert word not in text

    def test_records_corpus_and_dataset_context(self):
        text = render_markdown(self.build(), METRIC_KEYS, "ndcg@10")
        assert "docs" in text and "8 documents" in text and "21 chunks" in text
        assert "eval.json" in text and "18 labeled queries" in text


class TestWriteMarkdown:
    def test_shares_the_json_artifact_timestamp(self, tmp_path):
        run = build_run(
            [result("BM25", 0.7, 1.0)],
            Config(),
            corpus={"path": "docs", "documents": 8, "chunks": 21},
            dataset={"path": "eval.json", "queries": 18},
        )
        json_path = write_json(run, tmp_path)
        markdown_path = write_markdown(run, tmp_path, METRIC_KEYS, "ndcg@10")
        assert markdown_path.stem == json_path.stem
        assert markdown_path.suffix == ".md"

    def test_creates_the_output_directory(self, tmp_path):
        run = build_run([result("BM25", 0.7, 1.0)], Config(), corpus={}, dataset={})
        path = write_markdown(run, tmp_path / "nested" / "results", METRIC_KEYS, "ndcg@10")
        assert path.exists()


class TestArtifactNaming:
    def build(self):
        return build_run(
            [result("BM25", 0.7, 1.0)],
            Config(),
            corpus={"path": "docs", "documents": 1, "chunks": 1},
            dataset={"path": "eval.json", "queries": 1},
        )

    def test_two_runs_in_the_same_second_do_not_overwrite_each_other(self, tmp_path):
        # A benchmark over a small corpus finishes in well under a second, so
        # second-precision names collided and the later run silently replaced
        # the earlier one's results.
        write_artifacts(self.build(), tmp_path, METRIC_KEYS, "ndcg@10")
        write_artifacts(self.build(), tmp_path, METRIC_KEYS, "ndcg@10")

        assert len(list(tmp_path.glob("*.json"))) == 2
        assert len(list(tmp_path.glob("*.md"))) == 2

    def test_the_json_and_markdown_of_one_run_still_pair_up(self, tmp_path):
        json_path, markdown_path = write_artifacts(
            self.build(), tmp_path, METRIC_KEYS, "ndcg@10"
        )
        assert json_path.stem == markdown_path.stem

    def test_the_name_remains_a_sortable_utc_timestamp(self, tmp_path):
        import re

        name = write_artifacts(self.build(), tmp_path, METRIC_KEYS, "ndcg@10")[0].stem
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{9}Z", name)

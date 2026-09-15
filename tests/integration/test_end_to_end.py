"""The whole pipeline over a real corpus directory.

The BM25 path runs in the default suite because it needs no model download.
The dense, hybrid and reranked paths are marked ``slow``: they load local
models, which the first run has to fetch.
"""

import json

import pytest

from retrieveval.config import ChunkingConfig, Config, EvaluationConfig, RerankerConfig
from retrieveval.evaluation.dataset import load_dataset, validate_against_corpus
from retrieveval.evaluation.runner import metric_names, run_benchmark
from retrieveval.ingestion.chunking import chunk_documents
from retrieveval.ingestion.loaders import load_corpus
from retrieveval.reporting.report import build_run, write_json
from retrieveval.retrieval.factory import build_strategies


def pipeline(corpus_dir, dataset_file, config):
    documents = load_corpus(corpus_dir)
    chunks = chunk_documents(documents, config.chunking)
    queries = load_dataset(dataset_file)
    validate_against_corpus(queries, (d.id for d in documents), (c.id for c in chunks))
    strategies = build_strategies(chunks, config)
    return documents, chunks, queries, run_benchmark(strategies, queries, config.evaluation)


BM25_ONLY = Config(
    retrievers=("bm25",),
    chunking=ChunkingConfig(size=300, overlap=30),
    evaluation=EvaluationConfig(k=(1, 3)),
)


class TestBM25Pipeline:
    def test_runs_end_to_end(self, corpus_dir, dataset_file):
        documents, chunks, queries, results = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        assert len(documents) == 4
        assert len(chunks) >= len(documents)
        assert len(queries) == 4
        assert [r.name for r in results] == ["BM25"]

    def test_every_configured_metric_is_produced(self, corpus_dir, dataset_file):
        _, _, _, results = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        assert set(results[0].metrics) == set(metric_names(BM25_ONLY.evaluation))

    def test_metrics_are_in_range(self, corpus_dir, dataset_file):
        _, _, _, results = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        for value in results[0].metrics.values():
            assert 0.0 <= value <= 1.0

    def test_finds_the_exact_error_code(self, corpus_dir, dataset_file):
        # A bare identifier is the case lexical retrieval is expected to nail.
        _, _, _, results = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        by_query = {q.query: q for q in results[0].queries}
        assert by_query["TOKEN_EXPIRED"].first_relevant_rank == 1

    def test_latency_is_measured_per_query(self, corpus_dir, dataset_file):
        _, _, _, results = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        assert results[0].latency.samples == 4
        assert results[0].latency.mean_ms > 0

    def test_indexing_time_is_reported_apart_from_query_latency(self, corpus_dir, dataset_file):
        _, _, _, results = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        assert results[0].index_time_ms > 0
        assert results[0].index_time_ms != results[0].latency.mean_ms

    def test_results_are_reproducible_across_runs(self, corpus_dir, dataset_file):
        _, _, _, first = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        _, _, _, second = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        assert first[0].metrics == second[0].metrics

    def test_writes_a_json_artifact_that_can_be_read_back(
        self, corpus_dir, dataset_file, tmp_path
    ):
        documents, chunks, queries, results = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        run = build_run(
            results,
            BM25_ONLY,
            corpus={"path": str(corpus_dir), "documents": len(documents), "chunks": len(chunks)},
            dataset={"path": str(dataset_file), "queries": len(queries)},
        )
        record = json.loads(write_json(run, tmp_path / "results").read_text(encoding="utf-8"))
        assert record["config"]["chunking"] == {"size": 300, "overlap": 30}
        assert record["corpus"]["documents"] == 4
        assert record["strategies"][0]["name"] == "BM25"
        assert len(record["strategies"][0]["queries"]) == 4

    def test_chunk_size_changes_the_chunking_not_the_contract(self, corpus_dir, dataset_file):
        small = Config(
            retrievers=("bm25",),
            chunking=ChunkingConfig(size=120, overlap=20),
            evaluation=EvaluationConfig(k=(1, 3)),
        )
        _, coarse_chunks, _, _ = pipeline(corpus_dir, dataset_file, BM25_ONLY)
        _, fine_chunks, _, results = pipeline(corpus_dir, dataset_file, small)
        assert len(fine_chunks) > len(coarse_chunks)
        assert set(results[0].metrics) == set(metric_names(small.evaluation))


class TestMixedFormatCorpus:
    def test_a_pdf_is_benchmarked_alongside_text_documents(
        self, corpus_dir, tmp_path, make_pdf
    ):
        (corpus_dir / "refunds.pdf").write_bytes(
            make_pdf("Annual plans are refunded in full within 14 days of purchase")
        )
        dataset = tmp_path / "eval.json"
        dataset.write_text(
            json.dumps(
                [{"query": "refund an annual plan", "relevant_documents": ["refunds.pdf"]}]
            ),
            encoding="utf-8",
        )
        documents, _, _, results = pipeline(corpus_dir, dataset, BM25_ONLY)
        assert "refunds.pdf" in {d.id for d in documents}
        assert results[0].queries[0].first_relevant_rank == 1

    def test_every_supported_format_is_retrievable_in_one_corpus(
        self, corpus_dir, tmp_path, make_pdf, make_docx
    ):
        # One corpus holding all five extensions, each the only document that
        # can answer its query. Nothing downstream of ingestion should be able
        # to tell what file a chunk came from.
        (corpus_dir / "refunds.pdf").write_bytes(
            make_pdf("Annual plans are refunded within 14 days of purchase")
        )
        make_docx(
            corpus_dir / "onboarding.docx",
            paragraphs=["New hires receive a laptop on their first Monday."],
        )
        (corpus_dir / "changelog.html").write_text(
            "<html><body><h1>Changelog</h1>"
            "<p>Version 4.2 introduced scheduled exports.</p></body></html>",
            encoding="utf-8",
        )
        (corpus_dir / "glossary.htm").write_text(
            "<p>A workspace is the top-level container for all content.</p>",
            encoding="utf-8",
        )

        dataset = tmp_path / "eval.json"
        dataset.write_text(
            json.dumps([
                {"query": "refund an annual plan", "relevant_documents": ["refunds.pdf"]},
                {"query": "laptop for new hires", "relevant_documents": ["onboarding.docx"]},
                {"query": "scheduled exports version", "relevant_documents": ["changelog.html"]},
                {"query": "top-level container for content",
                 "relevant_documents": ["glossary.htm"]},
                {"query": "TOKEN_EXPIRED", "relevant_documents": ["authentication.md"]},
            ]),
            encoding="utf-8",
        )

        documents, _, _, results = pipeline(corpus_dir, dataset, BM25_ONLY)
        loaded = {d.id for d in documents}
        assert {"refunds.pdf", "onboarding.docx", "changelog.html",
                "glossary.htm", "authentication.md"} <= loaded

        # Each query's one relevant document is found first, regardless of format.
        assert [q.first_relevant_rank for q in results[0].queries] == [1, 1, 1, 1, 1]


@pytest.mark.slow
class TestAllStrategies:
    config = Config(
        retrievers=("bm25", "dense", "hybrid"),
        chunking=ChunkingConfig(size=300, overlap=30),
        evaluation=EvaluationConfig(k=(1, 3)),
        reranker=RerankerConfig(
            enabled=True,
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            candidates=10,
            applies_to=("hybrid",),
        ),
    )

    def test_benchmarks_all_four_strategies(self, corpus_dir, dataset_file):
        _, _, _, results = pipeline(corpus_dir, dataset_file, self.config)
        assert [r.name for r in results] == ["BM25", "Dense", "Hybrid", "Hybrid + reranker"]

    def test_every_strategy_produces_the_same_metric_set(self, corpus_dir, dataset_file):
        _, _, _, results = pipeline(corpus_dir, dataset_file, self.config)
        expected = set(metric_names(self.config.evaluation))
        assert all(set(r.metrics) == expected for r in results)

    def test_reranking_costs_latency(self, corpus_dir, dataset_file):
        _, _, _, results = pipeline(corpus_dir, dataset_file, self.config)
        by_name = {r.name: r for r in results}
        assert by_name["Hybrid + reranker"].latency.mean_ms > by_name["Hybrid"].latency.mean_ms

    def test_dense_retrieval_answers_a_paraphrase(self, corpus_dir, dataset_file):
        # "when is staging wiped" shares no content word with "reset every
        # Sunday night" beyond 'staging'.
        _, _, _, results = pipeline(corpus_dir, dataset_file, self.config)
        dense = next(r for r in results if r.name == "Dense")
        by_query = {q.query: q for q in dense.queries}
        assert by_query["when is staging wiped"].first_relevant_rank == 1

    def test_the_embedding_cache_makes_a_second_index_faster(
        self, corpus_dir, dataset_file, tmp_path
    ):
        config = Config(
            retrievers=("dense",),
            chunking=self.config.chunking,
            evaluation=self.config.evaluation,
            dense=type(self.config.dense)(
                model=self.config.dense.model,
                batch_size=self.config.dense.batch_size,
                cache_dir=str(tmp_path / "cache"),
            ),
        )
        documents = load_corpus(corpus_dir)
        chunks = chunk_documents(documents, config.chunking)

        first = build_strategies(chunks, config)[0]
        second = build_strategies(chunks, config)[0]
        assert first.configuration["embeddings_from_cache"] is False
        assert second.configuration["embeddings_from_cache"] is True
        assert (tmp_path / "cache").exists()

"""CLI tests.

The emphasis is on failure paths. The promise the CLI makes is that a user
error produces one clear line and a non-zero exit code -- never a traceback --
so each of the documented error cases is checked for both.
"""

import json
import re

import pytest
from typer.testing import CliRunner

from retrieveval.cli import app

runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, list(args))


def output(result) -> str:
    """All console output with whitespace collapsed.

    Rich hard-wraps to the terminal width, so a message can be split across
    lines mid-sentence. Collapsing whitespace lets a test assert on the message
    the user reads rather than on where it happened to wrap.
    """
    text = result.stdout
    try:
        text += result.stderr
    except ValueError:  # pragma: no cover - depends on click version
        pass
    return re.sub(r"\s+", " ", text)


@pytest.fixture
def bm25_config(tmp_path):
    path = tmp_path / "retrieveval.yaml"
    path.write_text("retrievers: [bm25]\nevaluation:\n  k: [1, 3]\n", encoding="utf-8")
    return path


class TestEvaluate:
    def test_runs_and_reports(self, corpus_dir, dataset_file, bm25_config, tmp_path):
        result = invoke(
            "evaluate",
            "--corpus", str(corpus_dir),
            "--dataset", str(dataset_file),
            "--config", str(bm25_config),
            "--output", str(tmp_path / "results"),
        )
        assert result.exit_code == 0, output(result)
        assert "BM25" in output(result)
        assert "Retrieval quality" in output(result)
        assert "Latency" in output(result)

    def test_writes_the_json_artifact(self, corpus_dir, dataset_file, bm25_config, tmp_path):
        results_dir = tmp_path / "results"
        invoke(
            "evaluate",
            "--corpus", str(corpus_dir),
            "--dataset", str(dataset_file),
            "--config", str(bm25_config),
            "--output", str(results_dir),
        )
        written = list(results_dir.glob("*.json"))
        assert len(written) == 1
        record = json.loads(written[0].read_text(encoding="utf-8"))
        assert record["strategies"][0]["name"] == "BM25"

    def test_writes_a_paired_markdown_report(self, corpus_dir, dataset_file, bm25_config, tmp_path):
        results_dir = tmp_path / "results"
        invoke(
            "evaluate",
            "--corpus", str(corpus_dir),
            "--dataset", str(dataset_file),
            "--config", str(bm25_config),
            "--output", str(results_dir),
        )
        [json_path] = results_dir.glob("*.json")
        [markdown_path] = results_dir.glob("*.md")
        assert markdown_path.stem == json_path.stem
        text = markdown_path.read_text(encoding="utf-8")
        assert "| Strategy |" in text
        assert "BM25" in text

    def test_no_save_skips_the_artifact(self, corpus_dir, dataset_file, bm25_config, tmp_path):
        results_dir = tmp_path / "results"
        result = invoke(
            "evaluate",
            "--corpus", str(corpus_dir),
            "--dataset", str(dataset_file),
            "--config", str(bm25_config),
            "--output", str(results_dir),
            "--no-save",
        )
        assert result.exit_code == 0
        assert not results_dir.exists()

    def test_reports_corpus_and_dataset_size(self, corpus_dir, dataset_file, bm25_config):
        result = invoke(
            "evaluate",
            "--corpus", str(corpus_dir),
            "--dataset", str(dataset_file),
            "--config", str(bm25_config),
            "--no-save",
        )
        assert "4 documents" in output(result)
        assert "4 labeled queries" in output(result)

    def test_makes_no_overall_recommendation(self, corpus_dir, dataset_file, bm25_config):
        result = invoke(
            "evaluate",
            "--corpus", str(corpus_dir),
            "--dataset", str(dataset_file),
            "--config", str(bm25_config),
            "--no-save",
        )
        assert "your call" in output(result)
        assert "recommend" not in output(result).lower()


class TestValidationErrors:
    """Every case the project promises a clear message for."""

    def assert_clean_failure(self, result, fragment):
        assert result.exit_code == 1, output(result)
        assert "Traceback" not in output(result)
        assert fragment in output(result)

    def test_missing_corpus(self, dataset_file, tmp_path):
        result = invoke(
            "evaluate", "--corpus", str(tmp_path / "nope"), "--dataset", str(dataset_file)
        )
        self.assert_clean_failure(result, "corpus directory not found")

    def test_unsupported_files(self, corpus_dir, dataset_file, bm25_config):
        (corpus_dir / "budget.xlsx").write_bytes(b"binary")
        result = invoke(
            "evaluate", "--corpus", str(corpus_dir), "--dataset", str(dataset_file),
            "--config", str(bm25_config),
        )
        self.assert_clean_failure(result, "unsupported file type")

    def test_unsupported_files_can_be_skipped(self, corpus_dir, dataset_file, bm25_config):
        (corpus_dir / "budget.xlsx").write_bytes(b"binary")
        result = invoke(
            "evaluate", "--corpus", str(corpus_dir), "--dataset", str(dataset_file),
            "--config", str(bm25_config), "--skip-unsupported", "--no-save",
        )
        assert result.exit_code == 0, output(result)

    def test_malformed_dataset(self, corpus_dir, tmp_path, bm25_config):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        result = invoke(
            "evaluate", "--corpus", str(corpus_dir), "--dataset", str(bad),
            "--config", str(bm25_config),
        )
        self.assert_clean_failure(result, "not valid JSON")

    def test_missing_relevant_document(self, corpus_dir, tmp_path, bm25_config):
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps([{"query": "q", "relevant_documents": ["does_not_exist.md"]}]),
            encoding="utf-8",
        )
        result = invoke(
            "evaluate", "--corpus", str(corpus_dir), "--dataset", str(bad),
            "--config", str(bm25_config),
        )
        self.assert_clean_failure(result, "not in the corpus")

    def test_invalid_configuration(self, corpus_dir, dataset_file, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("chunking:\n  size: 100\n  overlap: 200\n", encoding="utf-8")
        result = invoke(
            "evaluate", "--corpus", str(corpus_dir), "--dataset", str(dataset_file),
            "--config", str(bad),
        )
        self.assert_clean_failure(result, "overlap must be smaller")

    def test_unknown_retriever(self, corpus_dir, dataset_file, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("retrievers: [splade]\n", encoding="utf-8")
        result = invoke(
            "evaluate", "--corpus", str(corpus_dir), "--dataset", str(dataset_file),
            "--config", str(bad),
        )
        self.assert_clean_failure(result, "supported retrievers are")

    def test_unavailable_model(self, corpus_dir, dataset_file, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "retrievers: [dense]\ndense:\n  model: not-a-real-org/not-a-real-model\n",
            encoding="utf-8",
        )
        result = invoke(
            "evaluate", "--corpus", str(corpus_dir), "--dataset", str(dataset_file),
            "--config", str(bad),
        )
        self.assert_clean_failure(result, "could not load the embedding model")


class TestIndexCommand:
    def test_says_there_is_nothing_to_do_without_a_dense_retriever(
        self, corpus_dir, bm25_config
    ):
        result = invoke("index", "--corpus", str(corpus_dir), "--config", str(bm25_config))
        assert result.exit_code == 0
        assert "nothing" in output(result)

    @pytest.mark.slow
    def test_builds_the_embedding_cache(self, corpus_dir, tmp_path):
        config = tmp_path / "dense.yaml"
        config.write_text(
            f"retrievers: [dense]\ndense:\n  cache_dir: {tmp_path / 'cache'}\n", encoding="utf-8"
        )
        result = invoke("index", "--corpus", str(corpus_dir), "--config", str(config))
        assert result.exit_code == 0, output(result)
        assert "Embeddings ready" in output(result)
        assert list((tmp_path / "cache").glob("*.npy"))


class TestHelp:
    def test_bare_invocation_shows_help(self):
        assert "evaluate" in output(invoke())

    def test_evaluate_help_documents_every_option(self):
        text = output(invoke("evaluate", "--help"))
        for option in ("--corpus", "--dataset", "--config", "--output", "--skip-unsupported"):
            assert option in text

"""Configuration parsing tests, focused on the errors a user will actually hit."""

import pytest

from retrieveval.config import Config, ConfigError


def parse(**sections):
    return Config.from_mapping(sections)


class TestDefaults:
    def test_an_empty_configuration_is_valid(self):
        config = Config.from_mapping({})
        assert config.chunking.size == 500
        assert config.chunking.overlap == 50
        assert config.retrievers == ("bm25", "dense", "hybrid")
        assert config.evaluation.k == (1, 5, 10)
        assert config.reranker.enabled is False

    def test_none_is_treated_as_empty(self):
        assert Config.from_mapping(None).chunking.size == 500


class TestChunking:
    def test_parses_values(self):
        chunking = parse(chunking={"size": 800, "overlap": 100}).chunking
        assert (chunking.size, chunking.overlap) == (800, 100)

    def test_overlap_may_be_zero(self):
        assert parse(chunking={"overlap": 0}).chunking.overlap == 0

    def test_overlap_at_or_above_size_is_rejected(self):
        with pytest.raises(ConfigError, match="never advance through the document"):
            parse(chunking={"size": 100, "overlap": 100})

    def test_negative_overlap_is_rejected(self):
        with pytest.raises(ConfigError, match="chunking.overlap must not be negative"):
            parse(chunking={"overlap": -1})

    def test_non_integer_size_is_rejected(self):
        with pytest.raises(ConfigError, match="chunking.size must be an integer"):
            parse(chunking={"size": "500"})

    def test_booleans_are_not_accepted_as_integers(self):
        with pytest.raises(ConfigError, match="must be an integer"):
            parse(chunking={"size": True})


class TestRetrievers:
    def test_accepts_a_subset(self):
        assert parse(retrievers=["bm25"]).retrievers == ("bm25",)

    def test_normalises_case_and_duplicates(self):
        assert parse(retrievers=["BM25", "bm25", " Dense "]).retrievers == ("bm25", "dense")

    def test_unknown_retriever_lists_the_supported_ones(self):
        with pytest.raises(ConfigError, match="supported retrievers are: bm25, dense, hybrid"):
            parse(retrievers=["splade"])

    def test_empty_list_is_rejected(self):
        with pytest.raises(ConfigError, match="non-empty list"):
            parse(retrievers=[])


class TestEvaluation:
    def test_sorts_and_deduplicates_cutoffs(self):
        assert parse(evaluation={"k": [10, 1, 5, 5]}).evaluation.k == (1, 5, 10)

    def test_accepts_a_bare_integer(self):
        assert parse(evaluation={"k": 5}).evaluation.k == (5,)

    def test_max_k(self):
        assert parse(evaluation={"k": [1, 20]}).evaluation.max_k == 20

    def test_rejects_non_positive_cutoffs(self):
        with pytest.raises(ConfigError, match="positive integers"):
            parse(evaluation={"k": [0]})


class TestReranker:
    def test_defaults_apply_to_every_configured_retriever(self):
        config = parse(retrievers=["bm25", "hybrid"], reranker={"enabled": True})
        assert config.reranker.applies_to == ("bm25", "hybrid")

    def test_can_be_narrowed(self):
        config = parse(
            retrievers=["bm25", "dense", "hybrid"],
            reranker={"enabled": True, "applies_to": ["hybrid"]},
        )
        assert config.reranker.applies_to == ("hybrid",)

    def test_applying_to_an_unconfigured_retriever_is_rejected(self):
        with pytest.raises(ConfigError, match="not configured"):
            parse(retrievers=["bm25"], reranker={"enabled": True, "applies_to": ["hybrid"]})

    def test_enabled_must_be_a_boolean(self):
        with pytest.raises(ConfigError, match="must be true or false"):
            parse(reranker={"enabled": "yes"})


class TestTypos:
    """Unknown keys are rejected so a typo cannot silently change what is measured."""

    def test_unknown_top_level_key(self):
        with pytest.raises(ConfigError, match="unknown key in the configuration file: chunk"):
            Config.from_mapping({"chunk": {"size": 100}})

    def test_unknown_nested_key_lists_the_allowed_ones(self):
        with pytest.raises(ConfigError, match="allowed: overlap, size"):
            parse(chunking={"chunk_size": 100})

    def test_reports_several_typos_at_once(self):
        with pytest.raises(ConfigError, match="unknown keys in chunking: overlaps, sizes"):
            parse(chunking={"sizes": 1, "overlaps": 2})


class TestLoading:
    def test_reads_a_yaml_file(self, tmp_path):
        path = tmp_path / "retrieveval.yaml"
        path.write_text("chunking:\n  size: 300\nretrievers: [bm25]\n", encoding="utf-8")
        config = Config.load(path)
        assert config.chunking.size == 300
        assert config.retrievers == ("bm25",)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="configuration file not found"):
            Config.load(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("chunking: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            Config.load(path)

    def test_errors_are_prefixed_with_the_file_name(self, tmp_path):
        path = tmp_path / "retrieveval.yaml"
        path.write_text("chunking:\n  size: -1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match=r"retrieveval\.yaml: chunking\.size"):
            Config.load(path)


class TestReproducibilityRecord:
    def test_to_dict_captures_every_parameter(self):
        record = parse(retrievers=["hybrid"], reranker={"enabled": True}).to_dict()
        assert record["retrievers"] == ["hybrid"]
        assert record["chunking"] == {"size": 500, "overlap": 50}
        assert record["hybrid"]["rrf_k"] == 60
        assert record["dense"]["model"]
        assert record["reranker"]["candidates"] == 50


class TestNonStringKeys:
    """YAML keys are not always strings, and the CLI promises no tracebacks."""

    def test_an_integer_key_is_reported_as_a_config_error(self):
        with pytest.raises(ConfigError, match="unknown key"):
            Config.from_mapping({2024: {"size": 100}})

    def test_a_boolean_key_is_reported_as_a_config_error(self):
        # YAML 1.1 reads bare `on:` and `yes:` as booleans, not strings.
        with pytest.raises(ConfigError, match="unknown key"):
            Config.from_mapping({True: 1})

    def test_a_non_string_nested_key_is_reported(self):
        with pytest.raises(ConfigError, match="unknown key"):
            Config.from_mapping({"chunking": {1: 2}})

    def test_a_directory_given_as_the_config_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not a file"):
            Config.load(tmp_path)

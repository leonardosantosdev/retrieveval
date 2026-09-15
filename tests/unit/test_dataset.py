"""Evaluation dataset loading and cross-checking against the corpus."""

import json

import pytest

from retrieveval.evaluation.dataset import DatasetError, load_dataset, validate_against_corpus
from retrieveval.models import EvalQuery


def write(tmp_path, payload, name="eval.json"):
    path = tmp_path / name
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    return path


class TestLoading:
    def test_reads_the_minimum_schema(self, tmp_path):
        path = write(tmp_path, [{"query": "How do I cancel?", "relevant_documents": ["c.md"]}])
        queries = load_dataset(path)
        assert queries == [EvalQuery("How do I cancel?", frozenset({"c.md"}))]

    def test_reads_optional_chunk_labels(self, tmp_path):
        path = write(
            tmp_path,
            [
                {
                    "query": "q",
                    "relevant_documents": ["c.md"],
                    "relevant_chunks": ["c.md::chunk_0"],
                }
            ],
        )
        assert load_dataset(path)[0].relevant_chunks == frozenset({"c.md::chunk_0"})

    def test_allows_annotation_keys(self, tmp_path):
        path = write(
            tmp_path,
            [{"query": "q", "relevant_documents": ["c.md"], "id": "q1", "notes": "lexical"}],
        )
        assert len(load_dataset(path)) == 1

    def test_trims_whitespace(self, tmp_path):
        path = write(tmp_path, [{"query": "  q  ", "relevant_documents": ["  c.md "]}])
        query = load_dataset(path)[0]
        assert query.query == "q"
        assert query.relevant_documents == frozenset({"c.md"})


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetError, match="evaluation dataset not found"):
            load_dataset(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path):
        with pytest.raises(DatasetError, match="not valid JSON"):
            load_dataset(write(tmp_path, "{not json"))

    def test_top_level_must_be_an_array(self, tmp_path):
        with pytest.raises(DatasetError, match="expected a JSON array"):
            load_dataset(write(tmp_path, {"queries": []}))

    def test_empty_dataset(self, tmp_path):
        with pytest.raises(DatasetError, match="contains no queries"):
            load_dataset(write(tmp_path, []))

    def test_errors_name_the_offending_entry(self, tmp_path):
        path = write(
            tmp_path,
            [
                {"query": "ok", "relevant_documents": ["a.md"]},
                {"query": "", "relevant_documents": ["a.md"]},
            ],
        )
        with pytest.raises(DatasetError, match="entry 1"):
            load_dataset(path)

    def test_missing_relevant_documents(self, tmp_path):
        with pytest.raises(DatasetError, match="missing required key 'relevant_documents'"):
            load_dataset(write(tmp_path, [{"query": "q"}]))

    def test_empty_relevant_documents_explains_why_it_is_rejected(self, tmp_path):
        with pytest.raises(DatasetError, match="cannot be scored"):
            load_dataset(write(tmp_path, [{"query": "q", "relevant_documents": []}]))

    def test_a_mistyped_key_is_rejected_rather_than_ignored(self, tmp_path):
        path = write(tmp_path, [{"query": "q", "relevant_docs": ["a.md"]}])
        with pytest.raises(DatasetError, match="unknown key"):
            load_dataset(path)

    def test_relevant_documents_must_be_a_list(self, tmp_path):
        with pytest.raises(DatasetError, match="must be a list of strings"):
            load_dataset(write(tmp_path, [{"query": "q", "relevant_documents": "a.md"}]))


class TestValidateAgainstCorpus:
    def test_accepts_labels_that_match(self):
        validate_against_corpus([EvalQuery("q", frozenset({"a.md"}))], ["a.md", "b.md"])

    def test_reports_a_label_with_no_matching_document(self):
        with pytest.raises(DatasetError, match="not in the corpus"):
            validate_against_corpus([EvalQuery("q", frozenset({"missing.md"}))], ["a.md"])

    def test_suggests_the_full_path_for_a_bare_filename(self):
        with pytest.raises(DatasetError, match=r"did you mean 'faq/cancellation\.md'"):
            validate_against_corpus(
                [EvalQuery("q", frozenset({"cancellation.md"}))],
                ["faq/cancellation.md", "billing.md"],
            )

    def test_suggests_a_near_miss(self):
        with pytest.raises(DatasetError, match=r"did you mean 'billing\.md'"):
            validate_against_corpus([EvalQuery("q", frozenset({"biling.md"}))], ["billing.md"])

    def test_explains_that_ids_are_relative_paths(self):
        with pytest.raises(DatasetError, match="corpus-relative paths"):
            validate_against_corpus([EvalQuery("q", frozenset({"zzz"}))], ["a.md"])

    def test_checks_chunk_labels_when_chunk_ids_are_supplied(self):
        query = EvalQuery("q", frozenset({"a.md"}), frozenset({"a.md::chunk_9"}))
        with pytest.raises(DatasetError, match="chunk id"):
            validate_against_corpus([query], ["a.md"], ["a.md::chunk_0"])

    def test_the_chunk_error_explains_the_dependency_on_chunking_settings(self):
        query = EvalQuery("q", frozenset({"a.md"}), frozenset({"a.md::chunk_9"}))
        with pytest.raises(DatasetError, match="chunking.size"):
            validate_against_corpus([query], ["a.md"], ["a.md::chunk_0"])

    def test_chunk_labels_are_not_checked_when_chunk_ids_are_omitted(self):
        query = EvalQuery("q", frozenset({"a.md"}), frozenset({"a.md::chunk_9"}))
        validate_against_corpus([query], ["a.md"])


class TestPathHandling:
    def test_a_directory_given_as_the_dataset(self, tmp_path):
        with pytest.raises(DatasetError, match="not a file"):
            load_dataset(tmp_path)


class TestIterableQueries:
    def test_a_generator_of_queries_is_still_fully_validated(self):
        # The queries are walked twice, once per label kind. A generator that
        # was consumed by the first pass used to skip chunk validation entirely.
        query = EvalQuery("q", frozenset({"a.md"}), frozenset({"a.md::chunk_9"}))
        with pytest.raises(DatasetError, match="chunk id"):
            validate_against_corpus(iter([query]), ["a.md"], ["a.md::chunk_0"])

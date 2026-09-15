"""Tests for the chunk-ranking -> judged-ranking conversion."""

import math

import pytest

from retrieveval.evaluation.judgments import count_relevant, grade_ranking
from retrieveval.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from retrieveval.models import Chunk, EvalQuery, SearchResult


def ranking(*document_ids: str) -> list[SearchResult]:
    """Build a ranking of one chunk per entry, in the given document order."""
    results = []
    seen: dict[str, int] = {}
    for rank, document_id in enumerate(document_ids, start=1):
        position = seen.get(document_id, 0)
        seen[document_id] = position + 1
        chunk = Chunk(
            id=Chunk.make_id(document_id, position),
            document_id=document_id,
            text=f"text of {document_id} chunk {position}",
            position=position,
        )
        results.append(SearchResult(chunk=chunk, score=1.0 / rank, rank=rank))
    return results


def query(*relevant: str) -> EvalQuery:
    return EvalQuery(query="q", relevant_documents=frozenset(relevant))


class TestDocumentLevelGrading:
    def test_grades_relevant_documents(self):
        assert grade_ranking(ranking("a", "c", "b"), query("a", "b")) == [1.0, 0.0, 1.0]

    def test_repeated_chunks_of_a_credited_document_grade_zero(self):
        # 'a' is relevant, but finding it three times is still one document found.
        assert grade_ranking(ranking("a", "a", "a"), query("a")) == [1.0, 0.0, 0.0]

    def test_repeats_keep_their_rank_positions(self):
        # a, a, c, b, a  ->  the hit on 'b' stays at rank 4, not rank 2.
        grades = grade_ranking(ranking("a", "a", "c", "b", "a"), query("a", "b"))
        assert grades == [1.0, 0.0, 0.0, 1.0, 0.0]

    def test_nothing_relevant_retrieved(self):
        assert grade_ranking(ranking("x", "y"), query("a")) == [0.0, 0.0]

    def test_empty_ranking(self):
        assert grade_ranking([], query("a")) == []


class TestChunkLevelLabels:
    def test_chunk_labels_take_precedence_over_document_labels(self):
        results = ranking("a", "a")  # a::chunk_0, a::chunk_1
        labeled = EvalQuery(
            query="q",
            relevant_documents=frozenset({"a"}),
            relevant_chunks=frozenset({"a::chunk_1"}),
        )
        # Only the second chunk is relevant, even though its document is listed.
        assert grade_ranking(results, labeled) == [0.0, 1.0]
        assert count_relevant(labeled) == 1


class TestCountRelevant:
    def test_counts_documents_by_default(self):
        assert count_relevant(query("a", "b")) == 2

    def test_counts_chunks_when_labeled(self):
        labeled = EvalQuery(
            query="q",
            relevant_documents=frozenset({"a", "b"}),
            relevant_chunks=frozenset({"a::chunk_0"}),
        )
        assert count_relevant(labeled) == 1


class TestEndToEndOnADuplicateHeavyRanking:
    """The worked example from the `judgments` module docstring."""

    results = ranking("a", "a", "c", "b", "a")
    labels = query("a", "b")

    def test_grades(self):
        assert grade_ranking(self.results, self.labels) == [1.0, 0.0, 0.0, 1.0, 0.0]

    def test_recall_finds_both_documents(self):
        grades = grade_ranking(self.results, self.labels)
        assert recall_at_k(grades, count_relevant(self.labels), k=5) == 1.0

    def test_recall_at_three_finds_only_one(self):
        grades = grade_ranking(self.results, self.labels)
        assert recall_at_k(grades, count_relevant(self.labels), k=3) == pytest.approx(0.5)

    def test_mrr_uses_the_real_position_of_the_first_hit(self):
        grades = grade_ranking(self.results, self.labels)
        assert reciprocal_rank(grades) == 1.0

    def test_ndcg_penalises_the_duplicates_that_pushed_b_down(self):
        # DCG = 1/log2(2) + 1/log2(5) ; IDCG = 1/log2(2) + 1/log2(3)
        expected = (1 + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
        grades = grade_ranking(self.results, self.labels)
        assert ndcg_at_k(grades, count_relevant(self.labels), k=5) == pytest.approx(expected)
        assert expected < 1.0  # the duplicates cost something

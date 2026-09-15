"""Tests for the shared ranking helpers."""

import numpy as np

from retrieveval.models import Chunk
from retrieveval.retrieval.base import build_results, rank_top_k


class TestRankTopK:
    def test_orders_by_descending_score(self):
        assert rank_top_k(np.array([0.1, 0.9, 0.5]), 3).tolist() == [1, 2, 0]

    def test_respects_k(self):
        assert rank_top_k(np.array([0.1, 0.9, 0.5]), 2).tolist() == [1, 2]

    def test_k_larger_than_the_input_is_clamped(self):
        assert len(rank_top_k(np.array([0.1, 0.9]), 10)) == 2

    def test_empty_input(self):
        assert rank_top_k(np.array([]), 5).tolist() == []

    def test_non_positive_k(self):
        assert rank_top_k(np.array([0.1, 0.9]), 0).tolist() == []

    def test_ties_break_by_ascending_index(self):
        assert rank_top_k(np.array([1.0, 1.0, 1.0]), 3).tolist() == [0, 1, 2]

    def test_ties_break_by_index_even_in_a_large_tied_group(self):
        # argpartition selects an arbitrary k among equal scores, so a cut
        # landing inside a tied group used to return whichever members
        # introselect happened to pick -- indices in the 900s for a corpus of
        # 1000 identical scores. Reproducible benchmarks depend on this.
        scores = np.ones(1000)
        assert rank_top_k(scores, 5).tolist() == [0, 1, 2, 3, 4]

    def test_a_tied_boundary_still_prefers_lower_indices(self):
        # One clear winner, then 999 tied for second place.
        scores = np.ones(1000)
        scores[500] = 2.0
        assert rank_top_k(scores, 4).tolist() == [500, 0, 1, 2]

    def test_is_reproducible_across_calls(self):
        scores = np.ones(500)
        assert rank_top_k(scores, 10).tolist() == rank_top_k(scores, 10).tolist()


class TestBuildResults:
    def chunks(self, count):
        return [
            Chunk(id=f"d{i}.md::chunk_0", document_id=f"d{i}.md", text="t", position=0)
            for i in range(count)
        ]

    def test_ranks_start_at_one(self):
        results = build_results(self.chunks(3), np.array([0.1, 0.9, 0.5]), 3)
        assert [r.rank for r in results] == [1, 2, 3]
        assert [r.chunk.document_id for r in results] == ["d1.md", "d2.md", "d0.md"]

    def test_min_score_excludes_non_matches(self):
        results = build_results(self.chunks(3), np.array([0.0, 0.9, 0.0]), 3, min_score=0.0)
        assert [r.chunk.document_id for r in results] == ["d1.md"]

    def test_ranks_stay_contiguous_after_filtering(self):
        results = build_results(self.chunks(3), np.array([0.5, 0.9, 0.0]), 3, min_score=0.0)
        assert [r.rank for r in results] == [1, 2]

"""Metric tests on small rankings whose expected values are derived by hand.

Where a value is not a round number it is written as the explicit arithmetic
that produces it (e.g. ``1 + 1 / log2(3)``) rather than a copied decimal, so
the derivation stays readable and reviewable.
"""

import math

import pytest

from retrieveval.evaluation.metrics import dcg, ndcg_at_k, recall_at_k, reciprocal_rank


class TestRecallAtK:
    def test_counts_relevant_hits_within_the_cutoff(self):
        # 2 of the 3 relevant items appear in the top 5.
        assert recall_at_k([0, 1, 0, 1, 0], num_relevant=3, k=5) == pytest.approx(2 / 3)

    def test_cutoff_excludes_later_hits(self):
        # Only the hit at rank 2 survives a cutoff of 2.
        assert recall_at_k([0, 1, 0, 1, 0], num_relevant=3, k=2) == pytest.approx(1 / 3)

    def test_nothing_relevant_in_the_cutoff(self):
        assert recall_at_k([0, 1, 0, 1, 0], num_relevant=3, k=1) == 0.0

    def test_all_relevant_retrieved(self):
        assert recall_at_k([1, 1], num_relevant=2, k=2) == 1.0

    def test_k_beyond_the_ranking_length_is_harmless(self):
        assert recall_at_k([1, 0], num_relevant=2, k=100) == pytest.approx(0.5)

    def test_empty_ranking(self):
        assert recall_at_k([], num_relevant=2, k=5) == 0.0


class TestReciprocalRank:
    def test_first_result_relevant(self):
        assert reciprocal_rank([1, 0, 0]) == 1.0

    def test_third_result_relevant(self):
        assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)

    def test_only_the_first_hit_matters(self):
        assert reciprocal_rank([0, 1, 1, 1]) == pytest.approx(0.5)

    def test_no_relevant_result(self):
        assert reciprocal_rank([0, 0, 0]) == 0.0

    def test_empty_ranking(self):
        assert reciprocal_rank([]) == 0.0


class TestDCG:
    def test_discounts_by_log2_of_rank_plus_one(self):
        # rank 1 -> 1/log2(2) = 1 ; rank 3 -> 1/log2(4) = 0.5
        assert dcg([1, 0, 1]) == pytest.approx(1.5)

    def test_rank_two_discount(self):
        assert dcg([0, 1]) == pytest.approx(1 / math.log2(3))

    def test_empty_ranking(self):
        assert dcg([]) == 0.0


class TestNDCGAtK:
    def test_ideal_ranking_scores_one(self):
        assert ndcg_at_k([1, 1, 0], num_relevant=2, k=3) == 1.0

    def test_perfect_ranking_longer_than_k(self):
        assert ndcg_at_k([1, 1, 1], num_relevant=3, k=2) == 1.0

    def test_relevant_item_pushed_down_the_ranking(self):
        # DCG = 1/log2(2) + 1/log2(4) ; IDCG = 1/log2(2) + 1/log2(3)
        expected = (1 + 1 / 2) / (1 + 1 / math.log2(3))
        assert ndcg_at_k([1, 0, 1], num_relevant=2, k=3) == pytest.approx(expected)

    def test_single_relevant_item_at_rank_two(self):
        # IDCG for one relevant item is 1.0, so nDCG is just the rank-2 discount.
        assert ndcg_at_k([0, 1], num_relevant=1, k=2) == pytest.approx(1 / math.log2(3))

    def test_cutoff_hides_the_relevant_results(self):
        assert ndcg_at_k([0, 1, 1], num_relevant=2, k=1) == 0.0

    def test_nothing_relevant_retrieved(self):
        assert ndcg_at_k([0, 0, 0], num_relevant=2, k=3) == 0.0

    def test_never_exceeds_one(self):
        # The de-duplication rule in `judgments` guarantees at most
        # `num_relevant` ones, which is what keeps this bound true.
        for grades, n in ([[1, 1, 1], 3], [[1, 1], 2], [[1], 1], [[1, 0, 1, 1], 3]):
            assert ndcg_at_k(grades, num_relevant=n, k=4) <= 1.0


class TestValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: recall_at_k([1], num_relevant=0, k=1),
            lambda: ndcg_at_k([1], num_relevant=0, k=1),
        ],
    )
    def test_rejects_queries_with_no_relevant_items(self, call):
        with pytest.raises(ValueError, match="num_relevant must be positive"):
            call()

    @pytest.mark.parametrize(
        "call",
        [
            lambda: recall_at_k([1], num_relevant=1, k=0),
            lambda: ndcg_at_k([1], num_relevant=1, k=-1),
        ],
    )
    def test_rejects_non_positive_k(self, call):
        with pytest.raises(ValueError, match="k must be positive"):
            call()

    def test_rejects_non_binary_grades(self):
        with pytest.raises(ValueError, match="binary"):
            recall_at_k([0.5], num_relevant=1, k=1)

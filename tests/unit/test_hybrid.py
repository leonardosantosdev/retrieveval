"""Reciprocal Rank Fusion tests."""

import pytest

from retrieveval.models import Chunk, SearchResult
from retrieveval.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def ranking(*chunk_ids: str) -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(id=cid, document_id=cid.split("::")[0], text=cid, position=0),
            score=1.0 / rank,
            rank=rank,
        )
        for rank, cid in enumerate(chunk_ids, start=1)
    ]


def ids(results):
    return [r.chunk.id for r in results]


class TestFusion:
    def test_appearing_in_both_rankings_beats_leading_only_one(self):
        # 'a' is 2nd for x and 1st for y; 'z' leads x but is absent from y.
        # a = 1/62 + 1/61 = 0.03227 ; z = 1/61 = 0.01639
        fused = reciprocal_rank_fusion([ranking("z", "a"), ranking("a", "w")])
        assert ids(fused)[0] == "a"

    def test_the_discount_is_convex_so_a_top_rank_outweighs_consistency(self):
        # 'b' is 2nd in both lists, 'a' and 'c' are 1st in one and 3rd in the
        # other. Because 1/x is convex, 1/61 + 1/63 > 2/62 -- so 'b' loses to
        # both. This is a real property of RRF, not a tie-breaking artefact:
        # the fusion rewards a strong opinion from one retriever more than
        # lukewarm agreement from two.
        fused = reciprocal_rank_fusion([ranking("a", "b", "c"), ranking("c", "b", "a")])
        assert ids(fused)[-1] == "b"
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)
        assert fused[-1].score == pytest.approx(2 / 62)

    def test_a_consistently_high_rank_beats_one_first_place(self):
        # 'a' at ranks 1 and 2 (1/61 + 1/62) beats 'c' at ranks 3 and 1.
        fused = reciprocal_rank_fusion([ranking("a", "b", "c"), ranking("c", "a", "d")])
        assert ids(fused)[0] == "a"

    def test_scores_follow_the_rrf_formula(self):
        fused = reciprocal_rank_fusion([ranking("a"), ranking("a")], rrf_k=60)
        assert fused[0].score == pytest.approx(2 / 61)

    def test_a_chunk_in_only_one_ranking_still_appears(self):
        fused = reciprocal_rank_fusion([ranking("a"), ranking("b")])
        assert set(ids(fused)) == {"a", "b"}

    def test_ranks_are_renumbered_from_one(self):
        fused = reciprocal_rank_fusion([ranking("a", "b"), ranking("b", "a")])
        assert [r.rank for r in fused] == [1, 2]

    def test_a_smaller_rrf_k_sharpens_the_advantage_of_top_ranks(self):
        lists = [ranking("a", "b", "c"), ranking("c", "b", "a")]
        sharp = reciprocal_rank_fusion(lists, rrf_k=1)
        flat = reciprocal_rank_fusion(lists, rrf_k=1000)
        spread = lambda f: f[0].score - f[-1].score  # noqa: E731
        assert spread(sharp) > spread(flat)

    def test_ties_are_broken_deterministically(self):
        lists = [ranking("b", "a"), ranking("a", "b")]
        assert ids(reciprocal_rank_fusion(lists)) == ids(reciprocal_rank_fusion(lists))

    def test_empty_rankings_fuse_to_nothing(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_one_empty_ranking_does_not_discard_the_other(self):
        assert ids(reciprocal_rank_fusion([ranking("a"), []])) == ["a"]


class FakeRetriever:
    """A retriever that replays a fixed ranking, to test fusion in isolation."""

    def __init__(self, name, chunk_ids):
        self.name = name
        self.chunk_ids = chunk_ids
        self.requested_k = None

    def index(self, chunks):
        pass

    def search(self, query, k):
        self.requested_k = k
        return ranking(*self.chunk_ids)[:k]

    def describe(self):
        return {"retriever": self.name}


class TestHybridRetriever:
    def test_fuses_its_components(self):
        # The chunk both components retrieved outranks each component's own
        # top hit -- the whole reason to fuse a lexical and a dense ranking.
        hybrid = HybridRetriever(
            [FakeRetriever("x", ["z", "a"]), FakeRetriever("y", ["a", "w"])]
        )
        assert ids(hybrid.search("q", k=1)) == ["a"]

    def test_queries_components_deeper_than_the_requested_k(self):
        # Fusing only the final top-k would hide exactly the mid-list
        # agreements RRF exists to promote.
        components = [FakeRetriever("x", ["a"]), FakeRetriever("y", ["a"])]
        HybridRetriever(components, depth=100).search("q", k=5)
        assert [c.requested_k for c in components] == [100, 100]

    def test_a_k_larger_than_depth_is_honoured(self):
        components = [FakeRetriever("x", ["a"]), FakeRetriever("y", ["a"])]
        HybridRetriever(components, depth=10).search("q", k=50)
        assert [c.requested_k for c in components] == [50, 50]

    def test_truncates_to_k(self):
        hybrid = HybridRetriever(
            [FakeRetriever("x", ["a", "b", "c"]), FakeRetriever("y", ["a", "b", "c"])]
        )
        assert len(hybrid.search("q", k=2)) == 2

    def test_needs_at_least_two_retrievers(self):
        with pytest.raises(ValueError, match="at least two retrievers"):
            HybridRetriever([FakeRetriever("x", ["a"])])

    def test_describe_nests_its_components(self):
        hybrid = HybridRetriever(
            [FakeRetriever("x", []), FakeRetriever("y", [])], rrf_k=42, depth=7
        )
        described = hybrid.describe()
        assert described["rrf_k"] == 42
        assert described["depth"] == 7
        assert described["components"] == [{"retriever": "x"}, {"retriever": "y"}]

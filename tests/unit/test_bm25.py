"""BM25 tests.

The scoring test derives its expected value from the Okapi formula written out
with the corpus's actual numbers, so a change in behaviour is visible as a
change in arithmetic rather than in an opaque constant.
"""

import math

import pytest

from retrieveval.models import Chunk
from retrieveval.retrieval.bm25 import BM25Retriever, tokenize


def chunks(*texts: str) -> list[Chunk]:
    return [
        Chunk(id=f"doc{i}.md::chunk_0", document_id=f"doc{i}.md", text=text, position=0)
        for i, text in enumerate(texts)
    ]


class TestTokenize:
    def test_lowercases_and_splits_on_non_word_characters(self):
        assert tokenize("Cancel my Subscription!") == ["cancel", "my", "subscription"]

    def test_keeps_identifiers_and_error_codes_usable(self):
        assert tokenize("ERROR_CODE_500 raised by auth-service") == [
            "error_code_500",
            "raised",
            "by",
            "auth",
            "service",
        ]

    def test_keeps_accented_words_intact(self):
        assert tokenize("cancelamento da assinatura") == ["cancelamento", "da", "assinatura"]


class TestScoring:
    def test_matches_the_okapi_formula(self):
        # Corpus: "cat" / "cat dog" / "dog"  ->  N=3, avgdl = 4/3
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index(chunks("cat", "cat dog", "dog"))

        # "cat" appears in 2 of 3 documents.
        idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        # doc0 has length 1 and term frequency 1.
        norm = 1 - 0.75 + 0.75 * (1 / (4 / 3))
        expected = idf * (1 * (1.5 + 1) / (1 + 1.5 * norm))

        assert retriever.score("cat")[0] == pytest.approx(expected)

    def test_shorter_documents_score_higher_for_the_same_term_frequency(self):
        retriever = BM25Retriever()
        retriever.index(chunks("cat", "cat dog"))
        scores = retriever.score("cat")
        assert scores[0] > scores[1]

    def test_a_term_present_everywhere_carries_little_weight(self):
        retriever = BM25Retriever()
        retriever.index(chunks("cat dog", "cat bird", "cat fish"))
        common = retriever.score("cat")
        distinctive = retriever.score("bird")
        assert distinctive.max() > common.max()

    def test_repeating_a_query_term_does_not_change_the_score(self):
        retriever = BM25Retriever()
        retriever.index(chunks("cat", "cat dog", "dog"))
        assert retriever.score("cat") == pytest.approx(retriever.score("cat cat cat"))

    def test_unmatched_query_scores_zero_everywhere(self):
        retriever = BM25Retriever()
        retriever.index(chunks("cat", "dog"))
        assert retriever.score("elephant").tolist() == [0.0, 0.0]


class TestSearch:
    def test_returns_ranked_results(self):
        retriever = BM25Retriever()
        retriever.index(chunks("cat", "cat dog", "dog"))
        results = retriever.search("cat", k=5)
        assert [r.document_id for r in results] == ["doc0.md", "doc1.md"]
        assert [r.rank for r in results] == [1, 2]
        assert results[0].score > results[1].score

    def test_excludes_chunks_that_share_no_term_with_the_query(self):
        # doc2 is not a weak match for "cat" -- it is not a match at all, and
        # letting it fill a rank would inflate recall.
        retriever = BM25Retriever()
        retriever.index(chunks("cat", "cat dog", "dog"))
        assert len(retriever.search("cat", k=3)) == 2

    def test_returns_nothing_when_no_chunk_matches(self):
        retriever = BM25Retriever()
        retriever.index(chunks("cat", "dog"))
        assert retriever.search("elephant", k=5) == []

    def test_respects_k(self):
        retriever = BM25Retriever()
        retriever.index(chunks("cat", "cat", "cat"))
        assert len(retriever.search("cat", k=2)) == 2

    def test_ties_are_broken_by_index_so_runs_are_reproducible(self):
        retriever = BM25Retriever()
        retriever.index(chunks("cat", "cat", "cat"))
        results = retriever.search("cat", k=3)
        assert [r.document_id for r in results] == ["doc0.md", "doc1.md", "doc2.md"]

    def test_searching_before_indexing_is_an_error(self):
        with pytest.raises(RuntimeError, match="before index"):
            BM25Retriever().search("cat", k=1)

    def test_indexing_an_empty_corpus_is_an_error(self):
        with pytest.raises(ValueError, match="empty corpus"):
            BM25Retriever().index([])


class TestDescribe:
    def test_reports_its_parameters_for_the_results_record(self):
        assert BM25Retriever(k1=1.2, b=0.5).describe() == {
            "retriever": "bm25",
            "k1": 1.2,
            "b": 0.5,
            "tokenizer": r"\w+ lowercase",
        }

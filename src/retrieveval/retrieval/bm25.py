"""Lexical retrieval with BM25 (Okapi).

Implemented directly rather than pulled in as a dependency: the scoring
function is small, its behaviour is worth being able to read while
interpreting a benchmark, and an explicit implementation can be unit-tested
against hand-computed values.

Tokenisation is deliberately plain -- lowercase, ``\\w+`` -- with no stemming
and no stopword list. That keeps identifiers, error codes and accented words
intact, which is precisely where a lexical baseline is expected to beat a
dense one. The trade-off is recorded in the README's limitations.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..models import Chunk, SearchResult
from .base import build_results

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

__all__ = ["BM25Retriever", "tokenize"]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Shared by indexing and querying, so the two
    sides can never disagree about what a term is."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over chunk texts."""

    name = "BM25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_lengths = np.zeros(0)
        self._avg_doc_length = 0.0
        # term -> (document indices, term frequencies) as parallel arrays
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        total = len(self._chunks)
        if total == 0:
            raise ValueError("cannot index an empty corpus")

        raw_postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(total, dtype=np.float64)
        for index, chunk in enumerate(self._chunks):
            tokens = tokenize(chunk.text)
            lengths[index] = len(tokens)
            for term, frequency in Counter(tokens).items():
                raw_postings.setdefault(term, []).append((index, frequency))

        self._doc_lengths = lengths
        self._avg_doc_length = float(lengths.mean()) or 1.0
        self._postings = {
            term: (
                np.fromiter((i for i, _ in entries), dtype=np.int64, count=len(entries)),
                np.fromiter((f for _, f in entries), dtype=np.float64, count=len(entries)),
            )
            for term, entries in raw_postings.items()
        }
        self._idf = {
            term: math.log(1 + (total - len(entries) + 0.5) / (len(entries) + 0.5))
            for term, entries in raw_postings.items()
        }

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every indexed chunk against ``query``."""
        if not self._chunks:
            raise RuntimeError("BM25Retriever.search() called before index()")

        scores = np.zeros(len(self._chunks), dtype=np.float64)
        # A term repeated in the query contributes once, as in standard BM25.
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_indices, frequencies = posting
            norm = 1 - self.b + self.b * (self._doc_lengths[doc_indices] / self._avg_doc_length)
            contribution = frequencies * (self.k1 + 1) / (frequencies + self.k1 * norm)
            scores[doc_indices] += self._idf[term] * contribution
        return scores

    def search(self, query: str, k: int) -> list[SearchResult]:
        # Chunks scoring zero share no term with the query; see `build_results`.
        return build_results(self._chunks, self.score(query), k, min_score=0.0)

    def describe(self) -> dict[str, Any]:
        return {"retriever": "bm25", "k1": self.k1, "b": self.b, "tokenizer": r"\w+ lowercase"}

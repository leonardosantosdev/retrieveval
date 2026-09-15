"""Dense retrieval with local sentence embeddings.

Chunk embeddings are cached on disk keyed by a hash of the model name and the
exact chunk texts, so re-running a benchmark after changing only, say, the RRF
constant does not re-embed the corpus. Changing chunk size or the model
changes the key and the cache is rebuilt -- there is no way for a stale
embedding to be silently reused.

Query embeddings are deliberately *not* cached: embedding the query is part of
what a served query costs, and hiding it would make the reported latency a
number no production system could reproduce.

``sentence_transformers`` and ``torch`` are imported lazily so that a BM25-only
run does not pay a multi-second import.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..models import Chunk, SearchResult
from .base import build_results

__all__ = ["DenseRetriever", "ModelUnavailableError"]


class ModelUnavailableError(RuntimeError):
    """Raised when a local model cannot be loaded."""


def _load_cache(path: Path | None, expected_rows: int) -> np.ndarray | None:
    """Read cached embeddings, or ``None`` if there is nothing usable there.

    A cache file is disposable by definition, so any failure to read one is
    answered by rebuilding rather than by raising. Interrupting a run midway
    through writing leaves a truncated file, and without this guard every
    subsequent run would abort until the user deleted the cache by hand.
    """
    if path is None or not path.exists():
        return None
    try:
        embeddings = np.load(path)
    except (OSError, ValueError, EOFError):
        return None
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        return None
    return embeddings


def _store_cache(path: Path | None, embeddings: np.ndarray) -> None:
    """Write embeddings so that an interrupted run cannot leave a partial file.

    The data goes to a temporary file in the same directory and is then moved
    into place; ``os.replace`` is atomic, so readers see either the previous
    cache or the complete new one, never a half-written array.
    """
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            np.save(stream, embeddings)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ModelUnavailableError(
            "dense retrieval requires the `sentence-transformers` package"
        ) from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise ModelUnavailableError(
            f"could not load the embedding model {model_name!r}: {exc}\n"
            "The first run downloads the model and needs network access; afterwards it is "
            "served from the local Hugging Face cache. Check the model name, or set "
            "HF_HOME to a cache that already contains it."
        ) from exc


class DenseRetriever:
    """Semantic retrieval by cosine similarity over normalised embeddings."""

    name = "Dense"

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model: Any = None
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._cache_hit = False

    # -- model ------------------------------------------------------------

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = _load_sentence_transformer(self.model_name)
        return self._model

    # -- caching ----------------------------------------------------------

    def _cache_key(self, chunks: Sequence[Chunk]) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_name.encode("utf-8"))
        for chunk in chunks:
            digest.update(b"\x00")
            digest.update(chunk.id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(chunk.text.encode("utf-8"))
        return digest.hexdigest()[:16]

    def _cache_path(self, chunks: Sequence[Chunk]) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"embeddings-{self._cache_key(chunks)}.npy"

    # -- retriever interface ----------------------------------------------

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            raise ValueError("cannot index an empty corpus")

        # Load the model even on a cache hit: a served system has to load it
        # before answering anything, so it belongs to the cost of becoming
        # queryable rather than to the first query's latency.
        _ = self.model

        path = self._cache_path(self._chunks)
        cached = _load_cache(path, len(self._chunks))
        if cached is not None:
            self._embeddings = cached
            self._cache_hit = True
            return

        self._cache_hit = False
        self._embeddings = self._encode([chunk.text for chunk in self._chunks])
        _store_cache(path, self._embeddings)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def score(self, query: str) -> np.ndarray:
        if self._embeddings is None:
            raise RuntimeError("DenseRetriever.search() called before index()")
        query_vector = self._encode([query])[0]
        # Both sides are L2-normalised, so the dot product is cosine similarity.
        return self._embeddings @ query_vector

    def search(self, query: str, k: int) -> list[SearchResult]:
        # No score floor: unlike BM25, a low cosine similarity is a weak match
        # rather than the absence of one, and the ranking is what is measured.
        return build_results(self._chunks, self.score(query), k)

    def describe(self) -> dict[str, Any]:
        device = None
        if self._model is not None:
            device = str(getattr(self._model, "device", None))
        return {
            "retriever": "dense",
            "model": self.model_name,
            "batch_size": self.batch_size,
            "similarity": "cosine",
            "device": device,
            "embeddings_from_cache": self._cache_hit,
        }

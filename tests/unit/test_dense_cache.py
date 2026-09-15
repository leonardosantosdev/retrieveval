"""Embedding cache behaviour.

Exercised directly rather than through DenseRetriever so the tests need no
model download: the cache is a file-handling concern, independent of embedding.
"""

import numpy as np
import pytest

from retrieveval.retrieval.dense import _load_cache, _store_cache

EMBEDDINGS = np.arange(12, dtype=np.float32).reshape(4, 3)


class TestRoundTrip:
    def test_stores_and_reads_back(self, tmp_path):
        path = tmp_path / "embeddings.npy"
        _store_cache(path, EMBEDDINGS)
        assert np.array_equal(_load_cache(path, 4), EMBEDDINGS)

    def test_creates_the_cache_directory(self, tmp_path):
        path = tmp_path / "nested" / "embeddings.npy"
        _store_cache(path, EMBEDDINGS)
        assert path.exists()

    def test_a_none_path_disables_caching(self, tmp_path):
        _store_cache(None, EMBEDDINGS)
        assert _load_cache(None, 4) is None


class TestRebuildsRatherThanCrashing:
    """A cache is disposable, so an unreadable one must never abort a run."""

    def test_a_truncated_file_is_ignored(self, tmp_path):
        # What a Ctrl-C partway through writing used to leave behind: every
        # later run then aborted with an uncaught error until the user deleted
        # the cache by hand.
        path = tmp_path / "embeddings.npy"
        _store_cache(path, EMBEDDINGS)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
        assert _load_cache(path, 4) is None

    def test_a_file_of_garbage_is_ignored(self, tmp_path):
        path = tmp_path / "embeddings.npy"
        path.write_bytes(b"not a numpy array")
        assert _load_cache(path, 4) is None

    def test_an_empty_file_is_ignored(self, tmp_path):
        path = tmp_path / "embeddings.npy"
        path.write_bytes(b"")
        assert _load_cache(path, 4) is None

    def test_a_missing_file_is_ignored(self, tmp_path):
        assert _load_cache(tmp_path / "absent.npy", 4) is None

    def test_a_cache_for_a_different_chunk_count_is_ignored(self, tmp_path):
        path = tmp_path / "embeddings.npy"
        _store_cache(path, EMBEDDINGS)
        assert _load_cache(path, 99) is None


class TestAtomicity:
    def test_writing_leaves_no_temporary_files_behind(self, tmp_path):
        _store_cache(tmp_path / "embeddings.npy", EMBEDDINGS)
        assert [p.name for p in tmp_path.iterdir()] == ["embeddings.npy"]

    def test_a_failed_write_does_not_destroy_the_previous_cache(self, tmp_path, monkeypatch):
        path = tmp_path / "embeddings.npy"
        _store_cache(path, EMBEDDINGS)

        def explode(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(np, "save", explode)
        with pytest.raises(KeyboardInterrupt):
            _store_cache(path, EMBEDDINGS * 2)

        # The old cache survives intact and no debris is left next to it.
        assert np.array_equal(_load_cache(path, 4), EMBEDDINGS)
        assert [p.name for p in tmp_path.iterdir()] == ["embeddings.npy"]

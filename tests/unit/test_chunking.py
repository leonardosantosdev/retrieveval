"""Chunking tests, including the two invariants the module promises."""

import pytest

from retrieveval.config import ChunkingConfig
from retrieveval.ingestion.chunking import (
    DEFAULT_SEPARATORS,
    _atomise,
    chunk_document,
    chunk_documents,
    split_text,
)
from retrieveval.models import Document


class TestSplitText:
    def test_short_text_is_one_chunk(self):
        assert split_text("a short document", size=100, overlap=10) == ["a short document"]

    def test_splits_on_paragraph_boundaries_when_it_can(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        assert split_text(text, size=30, overlap=0) == [
            "First paragraph here.",
            "Second paragraph here.",
            "Third paragraph here.",
        ]

    def test_prefers_line_over_word_boundaries(self):
        text = "alpha beta gamma\ndelta epsilon zeta"
        assert split_text(text, size=20, overlap=0) == ["alpha beta gamma", "delta epsilon zeta"]

    def test_no_chunk_exceeds_the_requested_size(self):
        text = " ".join(f"word{i}" for i in range(200))
        for chunk in split_text(text, size=60, overlap=20):
            assert len(chunk) <= 60

    def test_a_word_longer_than_the_chunk_size_is_cut(self):
        chunks = split_text("x" * 130, size=50, overlap=0)
        assert [len(c) for c in chunks] == [50, 50, 30]

    def test_consecutive_chunks_share_context(self):
        text = " ".join(f"word{i}" for i in range(40))
        chunks = split_text(text, size=60, overlap=20)
        assert len(chunks) > 1
        for previous, following in zip(chunks, chunks[1:]):
            tail_words = set(previous.split())
            assert tail_words & set(following.split()), "expected overlapping words"

    def test_zero_overlap_repeats_nothing(self):
        text = " ".join(f"word{i}" for i in range(40))
        chunks = split_text(text, size=60, overlap=0)
        words = [word for chunk in chunks for word in chunk.split()]
        assert len(words) == len(set(words))

    @pytest.mark.parametrize(
        "size,overlap,message",
        [
            (0, 0, "chunk size must be positive"),
            (100, -1, "must not be negative"),
            (50, 50, "must be smaller than size"),
        ],
    )
    def test_rejects_invalid_parameters(self, size, overlap, message):
        with pytest.raises(ValueError, match=message):
            split_text("text", size=size, overlap=overlap)


class TestAtomiseInvariant:
    """The split step must not lose or alter any character."""

    @pytest.mark.parametrize(
        "text",
        [
            "one\n\ntwo\n\nthree",
            "a sentence. another sentence. a third one.",
            " ".join(f"word{i}" for i in range(50)),
            "x" * 200,
            "trailing whitespace   \n\n",
        ],
    )
    def test_pieces_reconstruct_the_source(self, text):
        assert "".join(_atomise(text, 30, DEFAULT_SEPARATORS)) == text


class TestChunkDocument:
    config = ChunkingConfig(size=40, overlap=0)

    def test_chunks_carry_provenance(self):
        document = Document(
            id="faq/billing.md",
            text="First paragraph here.\n\nSecond paragraph here.",
            source_path="/corpus/faq/billing.md",
        )
        chunks = chunk_document(document, self.config)
        assert [c.id for c in chunks] == ["faq/billing.md::chunk_0", "faq/billing.md::chunk_1"]
        assert [c.document_id for c in chunks] == ["faq/billing.md"] * 2
        assert [c.position for c in chunks] == [0, 1]

    def test_chunk_documents_preserves_corpus_order(self):
        documents = [
            Document(id="a.md", text="alpha", source_path="a.md"),
            Document(id="b.md", text="beta", source_path="b.md"),
        ]
        chunks = chunk_documents(documents, self.config)
        assert [c.document_id for c in chunks] == ["a.md", "b.md"]

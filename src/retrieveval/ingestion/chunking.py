"""Recursive character chunking.

``size`` and ``overlap`` are measured in **characters**, not tokens. Characters
keep the chunker dependency-free and tokenizer-agnostic, which matters because
the point of the benchmark is to compare retrievers on identical chunks -- a
tokenizer-defined chunk size would silently couple chunking to whichever
embedding model happens to be configured.

The splitter walks a list of separators from coarsest to finest (paragraph,
line, sentence, word) and only falls back to cutting mid-word when a single
word exceeds the chunk size. Two invariants hold and are covered by tests:

1. no chunk is longer than ``size``;
2. concatenating the raw pieces before merging reproduces the source text
   exactly, so no content is dropped by the split itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..config import ChunkingConfig
from ..models import Chunk, Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

__all__ = ["split_text", "chunk_document", "chunk_documents", "DEFAULT_SEPARATORS"]


def _split_keeping_separator(text: str, separator: str) -> list[str]:
    """Split on ``separator``, leaving it attached to the piece it followed."""
    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]]
    if parts[-1]:
        pieces.append(parts[-1])
    return [piece for piece in pieces if piece]


def _hard_cut(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _atomise(text: str, size: int, separators: Sequence[str]) -> list[str]:
    """Break ``text`` into pieces of at most ``size`` characters."""
    if not text:
        return []
    if len(text) <= size:
        return [text]
    if not separators:
        return _hard_cut(text, size)

    separator, rest = separators[0], separators[1:]
    pieces: list[str] = []
    for part in _split_keeping_separator(text, separator):
        if len(part) <= size:
            pieces.append(part)
        else:
            pieces.extend(_atomise(part, size, rest))
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    """The trailing ``overlap`` characters, advanced to a word boundary."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    if len(tail) < len(text):
        # The cut may have landed mid-word; start after the first space instead
        # so the carried context does not begin with a word fragment.
        _, space, remainder = tail.partition(" ")
        if space:
            return remainder
    return tail


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``size`` characters."""
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"chunk overlap ({overlap}) must be smaller than size ({size})")

    pieces = _atomise(text, size, DEFAULT_SEPARATORS)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current)
            current = _overlap_tail(current, overlap)
            if len(current) + len(piece) > size:
                # The carried overlap does not leave room for this piece; the
                # size guarantee wins over the overlap request.
                current = ""
        current += piece
    if current:
        chunks.append(current)

    return [stripped for chunk in chunks if (stripped := chunk.strip())]


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Chunk one document, tagging each chunk with its source and position."""
    texts = split_text(document.text, config.size, config.overlap)
    return [
        Chunk(
            id=Chunk.make_id(document.id, position),
            document_id=document.id,
            text=text,
            position=position,
        )
        for position, text in enumerate(texts)
    ]


def chunk_documents(documents: Iterable[Document], config: ChunkingConfig) -> list[Chunk]:
    """Chunk a whole corpus, preserving document order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, config))
    return chunks

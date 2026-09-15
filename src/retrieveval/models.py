"""Core data types shared across the pipeline.

These are deliberately plain dataclasses: they cross every layer boundary
(ingestion -> chunking -> retrieval -> evaluation -> reporting), so they must
stay free of behaviour specific to any one layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A source document, after text extraction."""

    id: str
    """Identity of the document, stable across runs. Relevance labels refer to
    this value, so it is the corpus-relative path (e.g. ``faq/billing.md``)."""

    text: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text carrying its provenance."""

    id: str
    document_id: str
    text: str
    position: int
    """0-based index of this chunk within its document."""

    @staticmethod
    def make_id(document_id: str, position: int) -> str:
        return f"{document_id}::chunk_{position}"


@dataclass(frozen=True)
class SearchResult:
    """One ranked hit returned by a retriever."""

    chunk: Chunk
    score: float
    rank: int
    """1-based position in the returned ranking."""

    @property
    def document_id(self) -> str:
        return self.chunk.document_id


@dataclass(frozen=True)
class EvalQuery:
    """A labeled evaluation example.

    ``relevant_documents`` holds document ids; ``relevant_chunks`` is optional
    and, when present, takes precedence during judging.
    """

    query: str
    relevant_documents: frozenset[str]
    relevant_chunks: frozenset[str] = field(default_factory=frozenset)

    @property
    def has_chunk_labels(self) -> bool:
        return bool(self.relevant_chunks)

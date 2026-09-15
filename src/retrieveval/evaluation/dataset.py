"""Loading and validating the labeled evaluation dataset.

The dataset is a first-class input: if labels are wrong, every number in the
report is wrong in a way no retriever change can explain. So validation is
strict and errors name the offending entry.

The most common mistake by far is an id that does not match a corpus document
-- usually a bare filename where the corpus id is a relative path. That case
gets a suggested correction rather than just a rejection.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..models import EvalQuery

ALLOWED_KEYS = frozenset({"query", "relevant_documents", "relevant_chunks", "id", "notes"})

__all__ = ["DatasetError", "load_dataset", "validate_against_corpus"]

EXPECTED_SHAPE = """expected a JSON array of objects, for example:

[
  {"query": "How do I cancel my subscription?", "relevant_documents": ["cancellation.md"]}
]"""


class DatasetError(Exception):
    """Raised when the evaluation dataset is malformed or inconsistent with the corpus."""


def _string_list(value: Any, key: str, where: str) -> list[str]:
    if not isinstance(value, list):
        raise DatasetError(f"{where}: {key} must be a list of strings, got {value!r}")
    items = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise DatasetError(f"{where}: {key} entries must be non-empty strings, got {entry!r}")
        items.append(entry.strip())
    return items


def _parse_entry(raw: Any, where: str) -> EvalQuery:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: expected an object, got {type(raw).__name__}")

    unknown = sorted(set(raw) - ALLOWED_KEYS)
    if unknown:
        raise DatasetError(
            f"{where}: unknown key(s) {', '.join(unknown)}; "
            f"allowed keys are {', '.join(sorted(ALLOWED_KEYS))}"
        )

    query = raw.get("query")
    if not isinstance(query, str) or not query.strip():
        raise DatasetError(f"{where}: 'query' must be a non-empty string, got {query!r}")

    if "relevant_documents" not in raw:
        raise DatasetError(f"{where}: missing required key 'relevant_documents'")
    documents = _string_list(raw["relevant_documents"], "relevant_documents", where)
    if not documents:
        raise DatasetError(
            f"{where}: 'relevant_documents' is empty; a query with no relevant document "
            "cannot be scored and should be removed from the dataset"
        )

    chunks = _string_list(raw.get("relevant_chunks", []), "relevant_chunks", where)
    return EvalQuery(
        query=query.strip(),
        relevant_documents=frozenset(documents),
        relevant_chunks=frozenset(chunks),
    )


def load_dataset(path: str | Path) -> list[EvalQuery]:
    """Read and validate an evaluation dataset file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetError(f"evaluation dataset not found: {dataset_path}")
    if not dataset_path.is_file():
        raise DatasetError(f"evaluation dataset path is not a file: {dataset_path}")
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{dataset_path} is not valid JSON: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{dataset_path} is not valid UTF-8: {exc}") from exc

    if not isinstance(raw, list):
        raise DatasetError(f"{dataset_path}: {EXPECTED_SHAPE}")
    if not raw:
        raise DatasetError(f"{dataset_path} contains no queries")

    return [_parse_entry(entry, f"{dataset_path} entry {i}") for i, entry in enumerate(raw)]


def _suggest(unknown: str, known: Sequence[str]) -> str:
    matches = difflib.get_close_matches(unknown, known, n=1, cutoff=0.6)
    if matches:
        return f"{unknown!r} (did you mean {matches[0]!r}?)"
    tail_matches = [name for name in known if name.endswith("/" + unknown)]
    if tail_matches:
        return f"{unknown!r} (did you mean {tail_matches[0]!r}?)"
    return repr(unknown)


def validate_against_corpus(
    queries: Iterable[EvalQuery],
    document_ids: Iterable[str],
    chunk_ids: Iterable[str] | None = None,
) -> None:
    """Check that every label refers to something the corpus actually contains."""
    # The queries are walked twice, once per label kind. Materialising first
    # means a generator caller does not silently skip the chunk-label pass.
    queries = list(queries)
    known_documents = set(document_ids)
    missing = sorted(
        {
            document
            for query in queries
            for document in query.relevant_documents
            if document not in known_documents
        }
    )
    if missing:
        ordered = sorted(known_documents)
        listed = ", ".join(_suggest(name, ordered) for name in missing[:10])
        extra = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        message = f"the dataset references document(s) that are not in the corpus: {listed}{extra}."
        if ordered:
            message += f" Document ids are corpus-relative paths, e.g. {ordered[0]!r}."
        raise DatasetError(message)

    if chunk_ids is None:
        return
    known_chunks = set(chunk_ids)
    missing_chunks = sorted(
        {
            chunk
            for query in queries
            for chunk in query.relevant_chunks
            if chunk not in known_chunks
        }
    )
    if missing_chunks:
        listed = ", ".join(missing_chunks[:10])
        extra = f" (and {len(missing_chunks) - 10} more)" if len(missing_chunks) > 10 else ""
        raise DatasetError(
            f"the dataset references chunk id(s) that this chunking configuration did not "
            f"produce: {listed}{extra}. Chunk ids depend on chunking.size and "
            "chunking.overlap, so chunk-level labels must be regenerated when those change."
        )

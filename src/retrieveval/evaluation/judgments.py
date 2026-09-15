"""Turning search results into a judged ranking.

Retrievers return *chunks*, but relevance labels are normally written against
*documents*. This module bridges the two, and it is where V1's one significant
semantic decision lives:

**A result counts as relevant only the first time its document appears.**

A retriever that returns five chunks of the same relevant document has found
one relevant document, not five. Grading every chunk of it as relevant would
let DCG exceed the ideal DCG and push nDCG above 1.0. So repeated chunks from
an already-credited document are graded 0.0 while *keeping their rank
position* -- they are not removed from the ranking. Positions are preserved
because they reflect what the user actually receives: if the top three chunks
are near-duplicates, that is a real cost, and both MRR and nDCG should see it.

The consequence to keep in mind when reading a report: ``Recall@5`` means
"fraction of the relevant documents reachable within the top 5 retrieved
chunks", not "within the top 5 documents".

When a dataset provides chunk-level labels, they take precedence and the
de-duplication rule is a no-op, since chunk ids are unique within a ranking.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import EvalQuery, SearchResult

__all__ = ["grade_ranking", "count_relevant"]


def grade_ranking(results: Sequence[SearchResult], query: EvalQuery) -> list[float]:
    """Grade each result 1.0 or 0.0, in rank order, applying the rule above."""
    if query.has_chunk_labels:
        relevant = query.relevant_chunks
        identify = lambda result: result.chunk.id  # noqa: E731
    else:
        relevant = query.relevant_documents
        identify = lambda result: result.document_id  # noqa: E731

    credited: set[str] = set()
    grades: list[float] = []
    for result in results:
        key = identify(result)
        if key in relevant and key not in credited:
            credited.add(key)
            grades.append(1.0)
        else:
            grades.append(0.0)
    return grades


def count_relevant(query: EvalQuery) -> int:
    """How many relevant items exist for this query -- the denominator for recall."""
    return len(query.relevant_chunks if query.has_chunk_labels else query.relevant_documents)

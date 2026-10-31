"""Retrieval metrics (minimal for now; lesson 10 adds Recall@k, MRR, ...)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from parsrag.eval.dataset import QAItem
from parsrag.types import SearchResult

Retriever = Callable[[str, int], Sequence[SearchResult]]


def hit_rate(items: Sequence[QAItem], retrieve: Retriever, k: int = 3) -> float:
    """Fraction of answerable questions whose source document appears in the top ``k``.

    ``retrieve(question, k)`` must return search results whose chunks carry a
    ``doc_id``. Unanswerable questions (``doc_id is None``) are skipped.
    """
    answerable = [item for item in items if item.doc_id is not None]
    if not answerable:
        return 0.0
    hits = sum(
        any(r.chunk.doc_id == item.doc_id for r in retrieve(item.question, k))
        for item in answerable
    )
    return hits / len(answerable)

"""Lesson 05 starter (part 1 of 2) — retrieval metrics.

Fill in every function marked TODO, then check your work:

    python -m grader run 05

The docstrings are the specification; the data classes and hit_rate are done.
Reference solution: src/parsrag/eval/retrieval.py (try not to peek!).
Part 2 is tfidf.py in this folder.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from parsrag.eval.dataset import QAItem
from parsrag.normalize import normalize
from parsrag.types import Chunk, SearchResult

Retriever = Callable[[str, int], Sequence[SearchResult]]
Level = Literal["doc", "passage"]

# ------------------------------------------------------------------- metrics


def dedupe(ids: Iterable[Hashable]) -> list[Hashable]:
    """Keep the first occurrence of every id, preserving order."""
    # TODO: implement
    raise NotImplementedError("TODO: dedupe")


def recall_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    """Fraction of ``relevant`` ids found in ``ranked[:k]`` (0.0 if nothing is relevant).

    Use sets: len(set(ranked[:k]) & set(relevant)) / len(set(relevant)); 0.0 for k <= 0.
    """
    # TODO: implement
    raise NotImplementedError("TODO: recall_at_k")


def reciprocal_rank(
    ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int | None = None
) -> float:
    """``1 / rank`` of the first relevant id within the top ``k`` (all if ``None``), else 0."""
    # TODO: implement
    raise NotImplementedError("TODO: reciprocal_rank")


def ndcg_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    DCG  = sum over positions i = 0..k-1 of ranked[:k] that are relevant: 1 / log2(i + 2)
    IDCG = the same sum for a perfect ranking: i = 0 .. min(k, len(set(relevant))) - 1
    Return DCG / IDCG (0.0 if nothing is relevant or k <= 0).
    """
    # TODO: implement
    raise NotImplementedError("TODO: ndcg_at_k")


# --------------------------------------------------------------- relevance

_STRIP = re.compile(r"[\s‌]+")


def match_key(text: str) -> str:
    """Normalise and drop whitespace/ZWNJ so evidence matching ignores formatting."""
    return _STRIP.sub("", normalize(text))


def context_text(chunk: Chunk) -> str:
    """The text an LLM would see for this chunk (a sentence window, if there is one)."""
    window = chunk.metadata.get("window")
    return window if isinstance(window, str) else chunk.text


def contains_evidence(chunk: Chunk, item: QAItem) -> bool:
    """True if ``chunk`` comes from the answer's document and contains its evidence span.

    False if item.doc_id or item.evidence is None, or chunk.doc_id differs.
    Otherwise compare with match_key on both sides, using context_text(chunk)
    (so sentence windows count): match_key(evidence) in match_key(context).
    """
    # TODO: implement
    raise NotImplementedError("TODO: contains_evidence")


# --------------------------------------------------------------- benchmark


@dataclass
class QuestionResult:
    id: str
    rank: int | None  # 1-based rank of the first relevant item, None if not retrieved


@dataclass
class RetrievalReport:
    """Mean metrics over the answerable questions of a QA set."""

    level: str
    ks: tuple[int, ...]
    recall: dict[int, float]
    ndcg: dict[int, float]
    mrr: float
    per_question: list[QuestionResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.per_question)

    def row(self) -> dict[str, float]:
        """Flat ``{"R@1": ..., "R@3": ..., "MRR": ..., "nDCG@3": ...}`` for tables."""
        out = {f"R@{k}": v for k, v in self.recall.items()}
        out["MRR"] = self.mrr
        out.update({f"nDCG@{k}": v for k, v in self.ndcg.items()})
        return out


def ranked_ids(results: Sequence[SearchResult], item: QAItem, level: Level) -> list[Hashable]:
    """Turn search results into a ranked id list for one question.

    ``doc``: distinct document ids. ``passage``: every relevant chunk becomes
    the single id ``"*"`` (deduplicated), other chunks keep their own id.
    """
    if level == "doc":
        return dedupe(r.chunk.doc_id for r in results)
    return dedupe("*" if contains_evidence(r.chunk, item) else r.chunk.id for r in results)


def evaluate_retrieval(
    items: Sequence[QAItem],
    retrieve: Retriever,
    ks: Sequence[int] = (1, 3, 5),
    *,
    level: Level = "doc",
    depth: int | None = None,
) -> RetrievalReport:
    """Run ``retrieve(question, depth)`` for every answerable question and average the metrics.

    ``depth`` (default ``4 * max(ks)``) is how many chunks to fetch; several
    chunks often come from the same document, so document-level metrics need
    more than ``k`` chunks to see ``k`` distinct documents. MRR is computed
    over the whole retrieved list.

    Steps: sort/dedupe ks (ValueError if empty or not positive); depth defaults
    to 4 * max(ks); keep answerable items; for level="passage" every item needs
    an evidence (else ValueError). For each item:
      ranked = ranked_ids(retrieve(item.question, depth), item, level)
      relevant = {item.doc_id} for "doc", {"*"} for "passage"
      add recall_at_k / ndcg_at_k for every k, and reciprocal_rank;
      per_question gets QuestionResult(item.id, round(1 / rr) if rr else None).
    Divide the sums by the number of answerable items (at least 1).
    """
    # TODO: implement
    raise NotImplementedError("TODO: evaluate_retrieval")


def hit_rate(items: Sequence[QAItem], retrieve: Retriever, k: int = 3) -> float:
    """Fraction of answerable questions whose source document appears in the top ``k`` chunks.

    The simple metric of lessons 01–02, kept for comparison. Unanswerable
    questions (``doc_id is None``) are skipped.
    """
    answerable = [item for item in items if item.doc_id is not None]
    if not answerable:
        return 0.0
    hits = sum(
        any(r.chunk.doc_id == item.doc_id for r in retrieve(item.question, k))
        for item in answerable
    )
    return hits / len(answerable)


__all__ = [
    "QuestionResult",
    "RetrievalReport",
    "Retriever",
    "contains_evidence",
    "context_text",
    "dedupe",
    "evaluate_retrieval",
    "hit_rate",
    "match_key",
    "ndcg_at_k",
    "ranked_ids",
    "recall_at_k",
    "reciprocal_rank",
]


# Imported for you — you will need them.
_ = (math,)

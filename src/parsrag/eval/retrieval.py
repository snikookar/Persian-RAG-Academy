"""Retrieval metrics and a reusable benchmark (lessons 04–05).

Classic ranking metrics over a ranked list of ids and a set of relevant ids:

* :func:`recall_at_k` — what fraction of the relevant items appear in the top ``k``?
* :func:`reciprocal_rank` — ``1 / rank`` of the first relevant item (0 if none);
  its mean over questions is **MRR**.
* :func:`ndcg_at_k` — rewards relevant items near the top, with a logarithmic
  discount: ``Σ rel_i / log2(i + 1)`` divided by the best possible score.

:func:`evaluate_retrieval` runs a retriever over a QA set at two levels:

* ``level="doc"`` — relevant = the document that contains the answer.
* ``level="passage"`` — relevant = a retrieved chunk that actually *contains*
  the answer's evidence span (needs ``QAItem.evidence``). This is stricter and
  is what matters for generation: the LLM only sees chunks, not documents.

With one relevant item per question, Recall@k equals "hit@k" (was the answer
found in the top k?) and nDCG@k reduces to ``1 / log2(rank + 1)``.
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
    seen: set[Hashable] = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def recall_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    """Fraction of ``relevant`` ids found in ``ranked[:k]`` (0.0 if nothing is relevant)."""
    if not relevant or k <= 0:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(
    ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int | None = None
) -> float:
    """``1 / rank`` of the first relevant id within the top ``k`` (all if ``None``), else 0."""
    for rank, item in enumerate(ranked[:k] if k else ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(set(relevant)))))
    return dcg / ideal


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
    """True if ``chunk`` comes from the answer's document and contains its evidence span."""
    if item.doc_id is None or item.evidence is None or chunk.doc_id != item.doc_id:
        return False
    return match_key(item.evidence) in match_key(context_text(chunk))


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
    """
    ks = tuple(sorted(set(ks)))
    if not ks or ks[0] <= 0:
        raise ValueError("ks must contain positive integers")
    depth = depth or 4 * ks[-1]
    answerable = [i for i in items if i.doc_id is not None]
    if level == "passage" and any(i.evidence is None for i in answerable):
        raise ValueError("passage-level evaluation needs an evidence span for every question")
    recall = dict.fromkeys(ks, 0.0)
    ndcg = dict.fromkeys(ks, 0.0)
    rr = 0.0
    per_question = []
    for item in answerable:
        ranked = ranked_ids(retrieve(item.question, depth), item, level)
        relevant = {item.doc_id} if level == "doc" else {"*"}
        for k in ks:
            recall[k] += recall_at_k(ranked, relevant, k)
            ndcg[k] += ndcg_at_k(ranked, relevant, k)
        item_rr = reciprocal_rank(ranked, relevant)
        rr += item_rr
        per_question.append(QuestionResult(item.id, round(1 / item_rr) if item_rr else None))
    n = max(len(answerable), 1)
    return RetrievalReport(
        level=level,
        ks=ks,
        recall={k: v / n for k, v in recall.items()},
        ndcg={k: v / n for k, v in ndcg.items()},
        mrr=rr / n,
        per_question=per_question,
    )


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

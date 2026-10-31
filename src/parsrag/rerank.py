"""Reranking: retrieve many, rerank few (lesson 08).

A first-stage retriever (lesson 07) is fast but crude: it scores each chunk
on its own with a cheap formula. A **reranker** looks at the query and one
candidate *together* and re-orders a short list — slower per item, so it only
sees the top ``fetch_k`` candidates::

    reranked = RerankingRetriever(hybrid, LexicalReranker(), fetch_k=20)
    reranked.search("پرسش", k=3)

* :class:`Reranker` — protocol: ``rerank(query, results, top_k=None)``.
* :class:`LexicalReranker` — offline and deterministic: weighted query-term
  coverage (with typo/prefix-tolerant matching) + term proximity + a prior
  from the first-stage rank. Used in tests and as the course default.
* :class:`CrossEncoderReranker` — a real cross-encoder via
  ``sentence-transformers`` (``st`` extra; lazy import; never used in CI).
* :class:`LLMReranker` — asks any :class:`~parsrag.llm.LLM` to grade each
  candidate 0–10 with a Persian prompt (``FakeLLM`` in tests).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from parsrag.llm import LLM, Message
from parsrag.retrieval import PERSIAN_STOPWORDS, PersianTokenizer, Retriever, Tokenizer
from parsrag.stores.base import Filter
from parsrag.types import SearchResult


@runtime_checkable
class Reranker(Protocol):
    """Re-orders candidates for a query; returns at most ``top_k`` of them."""

    def rerank(
        self, query: str, results: Sequence[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]: ...


def sort_by_scores(
    results: Sequence[SearchResult], scores: Sequence[float], top_k: int | None
) -> list[SearchResult]:
    """Return ``results`` re-scored with ``scores``, best first (ties keep the old order)."""
    if len(scores) != len(results):
        raise ValueError("need one score per result")
    order = sorted(range(len(results)), key=lambda i: -scores[i])
    out = [SearchResult(chunk=results[i].chunk, score=float(scores[i])) for i in order]
    return out[:top_k] if top_k is not None else out


# ------------------------------------------------------------------- lexical


def edit_distance_at_most_one(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` differ by at most one insertion, deletion or substitution."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    edited = False
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            if edited:
                return False
            edited = True
            if len(a) == len(b):
                i += 1
            j += 1
        else:
            i += 1
            j += 1
    return True


def term_match(query_term: str, token: str, min_len: int = 4) -> float:
    """1.0 for an exact match, ``0.7`` for a near match, else 0.

    A *near* match is one typo (``قصطی`` ~ ``قسطی``) or a shared prefix
    (``امتیازهایی`` ~ ``امتیاز``) between words of at least ``min_len`` letters.
    """
    if query_term == token:
        return 1.0
    if len(query_term) < min_len or len(token) < min_len:
        return 0.0
    if token.startswith(query_term) or query_term.startswith(token):
        return 0.7
    if edit_distance_at_most_one(query_term, token):
        return 0.7
    return 0.0


def min_window(positions: dict[str, list[int]]) -> int:
    """Length (in tokens) of the shortest window containing every key at least once."""
    if not positions:
        return 0
    events = sorted((p, term) for term, ps in positions.items() for p in ps)
    best = events[-1][0] - events[0][0] + 1
    counts: dict[str, int] = {}
    left = 0
    for pos, term in events:
        counts[term] = counts.get(term, 0) + 1
        while len(counts) == len(positions):
            best = min(best, pos - events[left][0] + 1)
            first = events[left][1]
            counts[first] -= 1
            if counts[first] == 0:
                del counts[first]
            left += 1
    return best


class LexicalReranker:
    """An offline cross-encoder stand-in that reads query and chunk together.

    ``score = coverage + proximity_weight · proximity + prior_weight · 1/rank``

    * **coverage** — the idf-weighted share of the query's content words found
      in the chunk (exact = 1, typo/prefix = 0.7). Pass ``idf`` (for example
      ``bm25.idf``) to weight rare words; without it every word counts 1.
    * **proximity** — ``matched / window``: matched words close together
      («هزینهٔ پیک در تهران») beat the same words spread over a paragraph.
    * **prior** — the first-stage rank, so near-ties keep the retriever's order.

    Both extra terms default to 0: lesson 08 measured them on ``qa.jsonl`` and
    neither improved it (proximity made it worse). Exact ties always keep the
    first-stage order.
    """

    def __init__(
        self,
        *,
        idf: Callable[[str], float] | None = None,
        tokenizer: Tokenizer | None = None,
        proximity_weight: float = 0.0,
        prior_weight: float = 0.0,
    ) -> None:
        self.idf = idf
        self.tokenizer: Tokenizer = tokenizer or PersianTokenizer(
            stem=True, stopwords=PERSIAN_STOPWORDS
        )
        self.proximity_weight = proximity_weight
        self.prior_weight = prior_weight

    def score(self, query: str, text: str) -> float:
        """Coverage + proximity for one (query, chunk text) pair, without the prior."""
        q_terms = list(dict.fromkeys(self.tokenizer(query)))
        if not q_terms:
            return 0.0
        tokens = self.tokenizer(text)
        weights = {t: max(self.idf(t), 1e-6) if self.idf else 1.0 for t in q_terms}
        total = sum(weights.values())
        covered = 0.0
        positions: dict[str, list[int]] = {}
        for term in q_terms:
            best = 0.0
            for pos, token in enumerate(tokens):
                m = term_match(term, token)
                if m > 0:
                    positions.setdefault(term, []).append(pos)
                    best = max(best, m)
            covered += weights[term] * best
        coverage = covered / total
        proximity = 0.0
        if len(positions) >= 2:
            proximity = len(positions) / min_window(positions)
        return coverage + self.proximity_weight * proximity

    def rerank(
        self, query: str, results: Sequence[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        scores = [
            self.score(query, r.chunk.text) + self.prior_weight / rank
            for rank, r in enumerate(results, start=1)
        ]
        return sort_by_scores(results, scores, top_k)


# --------------------------------------------------------------- cross-encoder


class CrossEncoderReranker:
    """A real cross-encoder (``pip install 'parsrag[st]'``; downloads a model).

    Multilingual rerankers that cover Persian include ``BAAI/bge-reranker-v2-m3``
    and ``jinaai/jina-reranker-v2-base-multilingual``. Imported lazily; never
    used in tests or CI.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        batch_size: int = 16,
        model: Any | None = None,
    ) -> None:
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoderReranker needs sentence-transformers. "
                    "Install it with: pip install 'parsrag[st]'"
                ) from exc
            model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model

    def rerank(
        self, query: str, results: Sequence[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        return sort_by_scores(results, [float(s) for s in scores], top_k)


# ------------------------------------------------------------------------ LLM

LLM_RERANK_SYSTEM = (
    "تو یک ارزیاب دقیق برای موتور جست‌وجو هستی. فقط بگو متن داده‌شده چقدر برای "
    "پاسخ دادن به پرسش مفید است."
)
LLM_RERANK_TEMPLATE = """پرسش: {query}

متن:
{passage}

به این متن از ۰ (بی‌ربط) تا ۱۰ (پاسخ کامل پرسش در آن هست) امتیاز بده.
فقط یک سطر به این شکل بنویس:
امتیاز: <عدد>"""

_SCORE = re.compile(r"(?:امتیاز|score)\s*[:：]\s*([0-9۰-۹٠-٩]+(?:[.٫][0-9۰-۹٠-٩]+)?)", re.I)
_ANY_NUMBER = re.compile(r"[0-9۰-۹٠-٩]+(?:[.٫][0-9۰-۹٠-٩]+)?")
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٫", "01234567890123456789.")


def parse_score(text: str, max_score: float = 10.0) -> float | None:
    """Pull ``امتیاز: 7`` (or the first number) out of an LLM reply, clipped to ``[0, max]``."""
    m = _SCORE.search(text) or _ANY_NUMBER.search(text)
    if m is None:
        return None
    value = float((m.group(1) if m.re is _SCORE else m.group(0)).translate(_DIGITS))
    return min(max(value, 0.0), max_score)


class LLMReranker:
    """Pointwise LLM reranking: one call per candidate, ``امتیاز: 0..10`` each.

    Cost grows with ``len(results)`` — 20 candidates means 20 LLM calls per
    question. Unparseable replies get ``default_score``. Ties (common with
    integer grades) keep the first-stage order.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        system_prompt: str = LLM_RERANK_SYSTEM,
        template: str = LLM_RERANK_TEMPLATE,
        default_score: float = 0.0,
        max_chars: int = 1500,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.template = template
        self.default_score = default_score
        self.max_chars = max_chars

    def messages(self, query: str, passage: str) -> list[Message]:
        user = self.template.format(query=query, passage=passage[: self.max_chars])
        return [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user),
        ]

    def rerank(
        self, query: str, results: Sequence[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        scores = []
        for r in results:
            parsed = parse_score(self.llm.generate(self.messages(query, r.chunk.text)))
            scores.append(self.default_score if parsed is None else parsed)
        return sort_by_scores(results, scores, top_k)


# ------------------------------------------------------------------- pipeline


class RerankingRetriever:
    """Fetch ``fetch_k`` candidates from ``retriever``, rerank, keep ``k``."""

    def __init__(self, retriever: Retriever, reranker: Reranker, *, fetch_k: int = 20) -> None:
        if fetch_k <= 0:
            raise ValueError("fetch_k must be positive")
        self.retriever = retriever
        self.reranker = reranker
        self.fetch_k = fetch_k

    def search(self, query: str, k: int = 4, *, filter: Filter | None = None) -> list[SearchResult]:
        if k <= 0:
            return []
        candidates = self.retriever.search(query, max(self.fetch_k, k), filter=filter)
        return self.reranker.rerank(query, candidates, top_k=k)


__all__ = [
    "LLM_RERANK_SYSTEM",
    "LLM_RERANK_TEMPLATE",
    "CrossEncoderReranker",
    "LLMReranker",
    "LexicalReranker",
    "Reranker",
    "RerankingRetriever",
    "edit_distance_at_most_one",
    "min_window",
    "parse_score",
    "sort_by_scores",
    "term_match",
]

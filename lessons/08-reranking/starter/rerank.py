"""Lesson 08 starter — reranking.

Fill in every function marked TODO, then check your work:

    python -m grader run 08

The docstrings are the specification. The prompts, the cross-encoder wrapper
and the constructors are written for you. Reference solution:
src/parsrag/rerank.py (try not to peek until you have tried!).

Suggested order: sort_by_scores → edit_distance_at_most_one → term_match →
min_window → LexicalReranker.score/rerank → parse_score → LLMReranker.rerank →
RerankingRetriever.search.
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
    # TODO: implement
    raise NotImplementedError("TODO: sort_by_scores")


# ------------------------------------------------------------------- lexical


def edit_distance_at_most_one(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` differ by at most one insertion, deletion or substitution."""
    # TODO: implement
    raise NotImplementedError("TODO: edit_distance_at_most_one")


def term_match(query_term: str, token: str, min_len: int = 4) -> float:
    """1.0 for an exact match, ``0.7`` for a near match, else 0.

    A *near* match is one typo (``قصطی`` ~ ``قسطی``) or a shared prefix
    (``امتیازهایی`` ~ ``امتیاز``) between words of at least ``min_len`` letters.
    """
    # TODO: implement
    raise NotImplementedError("TODO: term_match")


def min_window(positions: dict[str, list[int]]) -> int:
    """Length (in tokens) of the shortest window containing every key at least once."""
    # TODO: implement
    raise NotImplementedError("TODO: min_window")


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
        # TODO: implement
        raise NotImplementedError("TODO: LexicalReranker.score")

    def rerank(
        self, query: str, results: Sequence[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        # TODO: implement
        raise NotImplementedError("TODO: LexicalReranker.rerank")


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
    # TODO: implement
    raise NotImplementedError("TODO: parse_score")


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
        # TODO: implement
        raise NotImplementedError("TODO: LLMReranker.rerank")


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
        # TODO: implement
        raise NotImplementedError("TODO: RerankingRetriever.search")


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

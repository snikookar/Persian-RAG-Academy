"""Lexical, dense and hybrid retrieval (lesson 07).

* :func:`tokenize` — Persian-aware tokens: normalise (lesson 02), lowercase,
  keep ZWNJ-joined words together (``کتاب‌ها`` is one token), optionally drop
  stop words and strip a few suffixes with :func:`light_stem`.
* :class:`BM25Retriever` — Okapi BM25 written from scratch over a list of chunks.
* :class:`DenseRetriever` — wraps any :class:`~parsrag.stores.VectorStore`
  (which already wraps an :class:`~parsrag.embeddings.Embedding`).
* :func:`reciprocal_rank_fusion` / :func:`weighted_score_fusion` and
  :class:`HybridRetriever` — combine several ranked lists into one.

Every retriever has the same method as a vector store::

    retriever.search("پرسش", k=5, filter={"doc_id": "kahroba-faq"}) -> list[SearchResult]

so they can be nested (a hybrid of hybrids) and evaluated with
:func:`parsrag.eval.evaluate_retrieval` via ``retriever.search``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Literal, Protocol, runtime_checkable

from parsrag.normalize import ZWNJ, normalize
from parsrag.stores.base import Filter, VectorStore, matches_filter
from parsrag.types import Chunk, SearchResult

Tokenizer = Callable[[str], list[str]]

# ------------------------------------------------------------------ tokenising

_TOKEN = re.compile(r"[\w‌]+")

_STOPWORDS_TEXT = """
    و در به از که را با این آن برای تا یا هم اما اگر نیز است هست بود شد شود می‌شود
    می کند کرد باید هر چه چی آیا ای یک بر پس ها های ی را هایی خود من تو او ما شما
    آن‌ها ایشان دارد دارند داشت نه بله چند چقدر چطور کجا کی کدام
"""
#: A deliberately small list of very frequent Persian function words.
PERSIAN_STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_TEXT.split())

#: Suffixes removed by :func:`light_stem`, longest first.
SUFFIXES: tuple[str, ...] = ("هایی", "های", "ترین", "ها", "تر", "ی")


def light_stem(token: str, min_stem: int = 3) -> str:
    """Strip one plural/comparative/ezafe-like suffix, very conservatively.

    * after a ZWNJ anything in :data:`SUFFIXES` goes: ``کتاب‌های`` → ``کتاب``;
    * attached suffixes only if at least ``min_stem`` letters remain:
      ``مرجوعی`` → ``مرجوع``, ``بیشترین`` → ``بیش``, but ``تنها`` stays.

    This is *not* a real stemmer (Persian morphology needs one, see the
    lesson's reading list); it is the cheapest thing that might help BM25.
    """
    if ZWNJ in token:
        head, _, tail = token.rpartition(ZWNJ)
        if tail in SUFFIXES and len(head) >= 2:
            return head
        return token
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_stem:
            return token[: -len(suffix)]
    return token


def tokenize(
    text: str,
    *,
    normalizer: Callable[[str], str] | None = normalize,
    stem: bool = False,
    stopwords: Iterable[str] | None = None,
    split_zwnj: bool = False,
) -> list[str]:
    """Split ``text`` into lowercase tokens.

    ``split_zwnj=True`` treats ZWNJ as a separator (``کتاب‌ها`` → ``کتاب``,
    ``ها``) — another cheap way to match inflected forms, measured in the lesson.
    """
    if normalizer is not None:
        text = normalizer(text)
    text = text.lower()
    if split_zwnj:
        text = text.replace(ZWNJ, " ")
    tokens = [t.strip(ZWNJ) for t in _TOKEN.findall(text)]
    tokens = [t for t in tokens if t]
    if stopwords is not None:
        stop = stopwords if isinstance(stopwords, (set, frozenset)) else set(stopwords)
        tokens = [t for t in tokens if t not in stop]
    if stem:
        tokens = [light_stem(t) for t in tokens]
    return tokens


class PersianTokenizer:
    """A configured :func:`tokenize`: ``PersianTokenizer(stem=True)("کتاب‌ها")``."""

    def __init__(
        self,
        *,
        stem: bool = False,
        stopwords: Iterable[str] | None = None,
        split_zwnj: bool = False,
        normalizer: Callable[[str], str] | None = normalize,
    ) -> None:
        self.stem = stem
        self.stopwords = frozenset(stopwords) if stopwords is not None else None
        self.split_zwnj = split_zwnj
        self.normalizer = normalizer

    def __call__(self, text: str) -> list[str]:
        return tokenize(
            text,
            normalizer=self.normalizer,
            stem=self.stem,
            stopwords=self.stopwords,
            split_zwnj=self.split_zwnj,
        )


# ---------------------------------------------------------------- the protocol


@runtime_checkable
class Retriever(Protocol):
    """Anything that returns ranked chunks for a query (vector stores included)."""

    def search(
        self, query: str, k: int = 4, *, filter: Filter | None = None
    ) -> list[SearchResult]: ...


# ------------------------------------------------------------------------ BM25


def bm25_idf(n_docs: int, df: int) -> float:
    """``log(1 + (N - df + 0.5) / (df + 0.5))`` — never negative (the Lucene variant)."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


class BM25Retriever:
    """Okapi BM25 over chunks.

    For a query with terms ``t`` and a chunk ``d``::

        score(d) = Σ_t idf(t) · tf(t, d) · (k1 + 1) / (tf(t, d) + k1 · (1 - b + b · |d| / avgdl))

    ``k1`` controls how fast repeated terms saturate, ``b`` how strongly long
    chunks are penalised. Only chunks sharing at least one term with the query
    are returned (score > 0). Use the *same* tokenizer for chunks and queries.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk] = (),
        *,
        tokenizer: Tokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")
        self.tokenizer: Tokenizer = tokenizer or PersianTokenizer()
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[int]] = {}
        self.add(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def avgdl(self) -> float:
        return sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def df(self, term: str) -> int:
        """Number of chunks containing ``term``."""
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        return bm25_idf(len(self._chunks), self.df(term))

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Index more chunks (ids must be unique)."""
        seen = {c.id for c in self._chunks}
        for chunk in chunks:
            if chunk.id in seen:
                raise ValueError(f"duplicate chunk id: {chunk.id!r}")
            seen.add(chunk.id)
            tokens = self.tokenizer(chunk.text)
            tf = Counter(tokens)
            index = len(self._chunks)
            self._chunks.append(chunk)
            self._tfs.append(tf)
            self._lengths.append(len(tokens))
            for term in tf:
                self._postings.setdefault(term, []).append(index)

    def scores(self, query: str) -> dict[int, float]:
        """``{chunk index: BM25 score}`` for every chunk that shares a term with ``query``."""
        avgdl = self.avgdl or 1.0
        out: dict[int, float] = {}
        for term in set(self.tokenizer(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for i in postings:
                tf = self._tfs[i][term]
                norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / avgdl)
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 4, *, filter: Filter | None = None) -> list[SearchResult]:
        """The ``k`` best-scoring chunks (ties keep insertion order)."""
        if k <= 0:
            return []
        scored = [
            (score, i)
            for i, score in self.scores(query).items()
            if matches_filter(self._chunks[i], filter)
        ]
        scored.sort(key=lambda p: (-p[0], p[1]))
        return [SearchResult(chunk=self._chunks[i], score=s) for s, i in scored[:k]]


# ----------------------------------------------------------------------- dense


class DenseRetriever:
    """A vector store plus the query preprocessing that matches its index.

    The store embeds the query; this wrapper only makes sure the query is
    normalised the same way the chunks were (lesson 05's rule: one
    ``preprocess`` for both sides).
    """

    def __init__(
        self, store: VectorStore, *, preprocess: Callable[[str], str] | None = normalize
    ) -> None:
        self.store = store
        self.preprocess = preprocess

    def search(self, query: str, k: int = 4, *, filter: Filter | None = None) -> list[SearchResult]:
        if self.preprocess is not None:
            query = self.preprocess(query)
        return self.store.search(query, k, filter=filter)


# ---------------------------------------------------------------------- fusion


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[SearchResult]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    Each chunk gets ``Σ_lists w / (k + rank)`` (ranks start at 1). Only *ranks*
    are used, so BM25 scores (0…20) and cosine scores (-1…1) never need to be
    put on the same scale. Ties are broken by first appearance.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    weights = list(weights) if weights is not None else [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("need one weight per result list")
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results, w in zip(result_lists, weights, strict=True):
        for rank, r in enumerate(results, start=1):
            fused[r.chunk.id] = fused.get(r.chunk.id, 0.0) + w / (k + rank)
            chunks.setdefault(r.chunk.id, r.chunk)
    order = sorted(fused, key=lambda cid: -fused[cid])  # stable: first appearance wins ties
    out = [SearchResult(chunk=chunks[cid], score=fused[cid]) for cid in order]
    return out[:limit] if limit is not None else out


Normalization = Literal["minmax", "max", "none"]


def normalize_scores(
    results: Sequence[SearchResult], method: Normalization = "minmax"
) -> list[float]:
    """Rescale one list's scores to ``[0, 1]`` (``minmax``) or divide by the max (``max``)."""
    scores = [r.score for r in results]
    if not scores or method == "none":
        return scores
    hi, lo = max(scores), min(scores)
    if method == "max":
        return [s / hi if hi > 0 else 0.0 for s in scores]
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def weighted_score_fusion(
    result_lists: Sequence[Sequence[SearchResult]],
    weights: Sequence[float] | None = None,
    *,
    normalization: Normalization = "minmax",
    limit: int | None = None,
) -> list[SearchResult]:
    """``Σ w · normalised score`` — a chunk missing from a list gets 0 from it.

    Unlike RRF this uses the score *values*, so it is sensitive to how each
    list's scores are distributed (one very confident BM25 hit can dominate).
    """
    weights = list(weights) if weights is not None else [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("need one weight per result list")
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results, w in zip(result_lists, weights, strict=True):
        for r, s in zip(results, normalize_scores(results, normalization), strict=True):
            fused[r.chunk.id] = fused.get(r.chunk.id, 0.0) + w * s
            chunks.setdefault(r.chunk.id, r.chunk)
    order = sorted(fused, key=lambda cid: -fused[cid])
    out = [SearchResult(chunk=chunks[cid], score=fused[cid]) for cid in order]
    return out[:limit] if limit is not None else out


Fusion = Literal["rrf", "weighted"]


class HybridRetriever:
    """Ask several retrievers for ``fetch_k`` candidates each, then fuse.

    >>> hybrid = HybridRetriever([bm25, dense])  # doctest: +SKIP
    >>> weighted = HybridRetriever(  # doctest: +SKIP
    ...     [bm25, dense], fusion="weighted", weights=[0.3, 0.7]
    ... )
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        fusion: Fusion = "rrf",
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        fetch_k: int | None = None,
        normalization: Normalization = "minmax",
    ) -> None:
        if not retrievers:
            raise ValueError("need at least one retriever")
        if fusion not in ("rrf", "weighted"):
            raise ValueError("fusion must be 'rrf' or 'weighted'")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("need one weight per retriever")
        self.retrievers = list(retrievers)
        self.fusion = fusion
        self.weights = list(weights) if weights is not None else None
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k
        self.normalization = normalization

    def search(self, query: str, k: int = 4, *, filter: Filter | None = None) -> list[SearchResult]:
        if k <= 0:
            return []
        fetch = max(self.fetch_k or 20, k)
        lists = [r.search(query, fetch, filter=filter) for r in self.retrievers]
        if self.fusion == "rrf":
            return reciprocal_rank_fusion(lists, k=self.rrf_k, weights=self.weights, limit=k)
        return weighted_score_fusion(lists, self.weights, normalization=self.normalization, limit=k)


__all__ = [
    "PERSIAN_STOPWORDS",
    "SUFFIXES",
    "BM25Retriever",
    "DenseRetriever",
    "Fusion",
    "HybridRetriever",
    "PersianTokenizer",
    "Retriever",
    "Tokenizer",
    "bm25_idf",
    "light_stem",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "tokenize",
    "weighted_score_fusion",
]

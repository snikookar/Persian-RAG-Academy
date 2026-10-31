"""Text embeddings.

* :class:`Embedding` — protocol: ``embed(texts) -> (n, dim) float32 array`` and
  ``embed_query(text) -> (dim,) array``. Vectors are L2-normalised.
* :class:`HashingEmbedding` — deterministic, offline, zero-download. Hashes
  word unigrams and character n-grams into a fixed-size vector. Character
  n-grams make it reasonably robust for Persian morphology (``کتاب``/``کتاب‌ها``
  share most n-grams) while staying fully reproducible.
* :class:`TfidfEmbedding` — a classic sparse baseline (lesson 05): needs
  :meth:`~TfidfEmbedding.fit` on the corpus first.
* :class:`SentenceTransformerEmbedding` — real neural embeddings behind the
  ``st`` extra (lazy import; never used in tests).
* :class:`OpenAICompatibleEmbedding` — any OpenAI-compatible ``/embeddings``
  endpoint behind the ``openai`` extra (lazy import; mocked in tests).
"""

from __future__ import annotations

import math
import os
import re
import zlib
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

_TOKEN = re.compile(r"\w+")


@runtime_checkable
class Embedding(Protocol):
    """Anything that maps texts to fixed-size, L2-normalised vectors."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> FloatArray: ...

    def embed_query(self, text: str) -> FloatArray: ...


def l2_normalize(matrix: FloatArray) -> FloatArray:
    """Normalise each row to unit length (zero rows stay zero)."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """``a·b / (|a| |b|)``: 1 = same direction, 0 = unrelated, -1 = opposite.

    Returns 0.0 if either vector is all zeros.
    """
    va, vb = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb) / denom if denom else 0.0


class HashingEmbedding:
    """Feature-hashing embedding over word unigrams + character n-grams.

    Each feature is hashed with CRC32 (stable across processes and platforms,
    unlike Python's salted ``hash``) into one of ``dim`` buckets; a second hash
    bit chooses the sign so collisions tend to cancel out. Counts are damped
    with ``1 + log(tf)`` and the vector is L2-normalised, so the dot product of
    two embeddings is their cosine similarity.

    Note that it does **not** normalise Persian text itself — that is the job
    of :mod:`parsrag.normalize` (lesson 02), and keeping the two separate lets
    us measure how much normalisation matters.
    """

    def __init__(
        self,
        dim: int = 1024,
        ngram_range: tuple[int, int] = (2, 4),
        word_weight: float = 1.0,
        char_weight: float = 1.0,
        lowercase: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        lo, hi = ngram_range
        if not 1 <= lo <= hi:
            raise ValueError("ngram_range must satisfy 1 <= min <= max")
        self._dim = dim
        self.ngram_range = ngram_range
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.lowercase = lowercase

    @property
    def dim(self) -> int:
        return self._dim

    def features(self, text: str) -> Counter[str]:
        """Return the bag of (prefixed) features for ``text``."""
        if self.lowercase:
            text = text.lower()
        feats: Counter[str] = Counter()
        lo, hi = self.ngram_range
        for token in _TOKEN.findall(text):
            if self.word_weight:
                feats[f"w:{token}"] += 1
            if self.char_weight:
                padded = f"<{token}>"
                for n in range(lo, hi + 1):
                    for i in range(len(padded) - n + 1):
                        feats[f"c:{padded[i : i + n]}"] += 1
        return feats

    def _vector(self, text: str) -> FloatArray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for feat, count in self.features(text).items():
            h = zlib.crc32(feat.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
            weight = self.word_weight if feat.startswith("w:") else self.char_weight
            vec[h % self._dim] += sign * weight * (1.0 + math.log(count))
        return vec

    def embed(self, texts: Sequence[str]) -> FloatArray:
        if isinstance(texts, str):
            raise TypeError("embed() expects a sequence of strings; use embed_query() for one")
        if len(texts) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> FloatArray:
        return self.embed([text])[0]


class SentenceTransformerEmbedding:
    """Neural embeddings via ``sentence-transformers`` (``pip install 'parsrag[st]'``).

    The library is imported lazily on construction, so importing
    :mod:`parsrag.embeddings` never requires it. Many multilingual models that
    handle Persian well (e.g. the E5 family) expect ``query:`` / ``passage:``
    prefixes; pass them via ``query_prefix`` / ``document_prefix``.

    Lesson 05 compares this class against :class:`HashingEmbedding`.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedding needs sentence-transformers. "
                "Install it with: pip install 'parsrag[st]'"
            ) from exc
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self._model: Any = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: Sequence[str]) -> FloatArray:
        vectors = self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), self._dim)

    def embed(self, texts: Sequence[str]) -> FloatArray:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> FloatArray:
        return self._encode([self.query_prefix + text])[0]


class TfidfEmbedding:
    """TF-IDF vectors over the corpus vocabulary — the classic pre-neural baseline.

    A term gets weight ``(1 + log tf) * idf`` with ``idf = log((1 + N) / (1 + df)) + 1``
    (N documents, df = documents containing the term), and every vector is
    L2-normalised. Rare terms («پیشتاز») therefore count more than common
    ones («است»). Unlike :class:`HashingEmbedding` it must see the corpus first::

        emb = TfidfEmbedding().fit(chunk_texts)
        store = InMemoryVectorStore(emb)

    ``analyzer="word"`` uses whole words; ``"char"`` uses character n-grams
    inside words (``ngram_range``), which is more robust to Persian morphology
    and spelling variants. Terms never seen during ``fit`` are ignored.
    """

    def __init__(
        self,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (2, 4),
        min_df: int = 1,
        lowercase: bool = True,
    ) -> None:
        if analyzer not in ("word", "char"):
            raise ValueError("analyzer must be 'word' or 'char'")
        lo, hi = ngram_range
        if not 1 <= lo <= hi:
            raise ValueError("ngram_range must satisfy 1 <= min <= max")
        if min_df < 1:
            raise ValueError("min_df must be >= 1")
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.lowercase = lowercase
        self.vocabulary: dict[str, int] = {}
        self.idf: FloatArray = np.zeros(0, dtype=np.float32)

    def terms(self, text: str) -> list[str]:
        """The terms of ``text`` (words, or padded character n-grams of each word)."""
        if self.lowercase:
            text = text.lower()
        words = _TOKEN.findall(text)
        if self.analyzer == "word":
            return words
        lo, hi = self.ngram_range
        out = []
        for word in words:
            padded = f"<{word}>"
            for n in range(lo, hi + 1):
                out.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
        return out

    @property
    def fitted(self) -> bool:
        return bool(self.vocabulary)

    @property
    def dim(self) -> int:
        if not self.fitted:
            raise RuntimeError("TfidfEmbedding is not fitted yet: call .fit(texts) first")
        return len(self.vocabulary)

    def fit(self, texts: Sequence[str]) -> TfidfEmbedding:
        """Learn the vocabulary and IDF weights from ``texts``. Returns ``self``."""
        if isinstance(texts, str):
            raise TypeError("fit() expects a sequence of strings")
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(self.terms(text)))
        kept = sorted(term for term, count in df.items() if count >= self.min_df)
        if not kept:
            raise ValueError("no terms to fit: the texts are empty")
        self.vocabulary = {term: i for i, term in enumerate(kept)}
        n = len(texts)
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[term])) + 1.0 for term in kept], dtype=np.float32
        )
        return self

    def embed(self, texts: Sequence[str]) -> FloatArray:
        if isinstance(texts, str):
            raise TypeError("embed() expects a sequence of strings; use embed_query() for one")
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(self.terms(text)).items():
                col = self.vocabulary.get(term)
                if col is not None:
                    matrix[row, col] = (1.0 + math.log(count)) * self.idf[col]
        return l2_normalize(matrix)

    def embed_query(self, text: str) -> FloatArray:
        return self.embed([text])[0]


class OpenAICompatibleEmbedding:
    """Embeddings from any OpenAI-compatible ``POST /embeddings`` endpoint.

    Works with hosted APIs and local servers that speak the same protocol
    (for example Ollama's ``/v1`` endpoint or vLLM). ``httpx`` is imported
    lazily (``pip install "parsrag[openai]"``); tests inject a fake ``client``
    so no request is ever sent. The dimension is discovered from the first
    response unless you pass ``dim``.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        dim: int | None = None,
        batch_size: int = 64,
        query_prefix: str = "",
        document_prefix: str = "",
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._dim = dim
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError(
                    "OpenAICompatibleEmbedding needs httpx. "
                    "Install it with: pip install 'parsrag[openai]'"
                ) from exc
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._request(["dimension probe"]).shape[1])
        return self._dim

    def _request(self, texts: list[str]) -> FloatArray:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self._get_client().post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            rows = sorted(data["data"], key=lambda d: d.get("index", 0))
            return np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response from {self.base_url}: {data!r}") from exc

    def _encode(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [
            self._request(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        matrix = np.vstack(parts)
        if self._dim is None:
            self._dim = int(matrix.shape[1])
        return l2_normalize(matrix)

    def embed(self, texts: Sequence[str]) -> FloatArray:
        if isinstance(texts, str):
            raise TypeError("embed() expects a sequence of strings; use embed_query() for one")
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> FloatArray:
        return self._encode([self.query_prefix + text])[0]

"""Text embeddings.

* :class:`Embedding` — protocol: ``embed(texts) -> (n, dim) float32 array`` and
  ``embed_query(text) -> (dim,) array``. Vectors are L2-normalised.
* :class:`HashingEmbedding` — deterministic, offline, zero-download. Hashes
  word unigrams and character n-grams into a fixed-size vector. Character
  n-grams make it reasonably robust for Persian morphology (``کتاب``/``کتاب‌ها``
  share most n-grams) while staying fully reproducible.
* :class:`SentenceTransformerEmbedding` — real neural embeddings behind the
  ``st`` extra (lazy import; never used in tests).
"""

from __future__ import annotations

import math
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

"""Lesson 05 starter (part 2 of 2) — cosine similarity and a TF-IDF embedding.

Fill in every function marked TODO, then check your work:

    python -m grader run 05

Reference solution: TfidfEmbedding and cosine_similarity in
src/parsrag/embeddings.py (try not to peek!).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from parsrag.embeddings import FloatArray, l2_normalize

_TOKEN = re.compile(r"\w+")


def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """``a·b / (|a| |b|)``: 1 = same direction, 0 = unrelated, -1 = opposite.

    Returns 0.0 if either vector is all zeros.
    """
    # TODO: implement
    raise NotImplementedError("TODO: cosine_similarity")


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
        """The terms of ``text`` (words, or padded character n-grams of each word).

        Lower-case if self.lowercase; words = _TOKEN.findall(text).
        "word": return the words. "char": for every word, pad it as "<word>" and
        add all its n-grams for n in ngram_range (inclusive), in order.
        """
        # TODO: implement
        raise NotImplementedError("TODO: TfidfEmbedding.terms")

    @property
    def fitted(self) -> bool:
        return bool(self.vocabulary)

    @property
    def dim(self) -> int:
        if not self.fitted:
            raise RuntimeError("TfidfEmbedding is not fitted yet: call .fit(texts) first")
        return len(self.vocabulary)

    def fit(self, texts: Sequence[str]) -> TfidfEmbedding:
        """Learn the vocabulary and IDF weights from ``texts``. Returns ``self``.

        * TypeError if ``texts`` is a single str
        * df[term] = number of texts containing the term (count each term once per text)
        * keep terms with df >= min_df, sorted; ValueError if none are left
        * self.vocabulary = {term: column}; N = len(texts)
        * self.idf[column] = log((1 + N) / (1 + df)) + 1   (np.float32 array)
        """
        # TODO: implement
        raise NotImplementedError("TODO: TfidfEmbedding.fit")

    def embed(self, texts: Sequence[str]) -> FloatArray:
        """Return an (len(texts), dim) float32 matrix of L2-normalised TF-IDF rows.

        * TypeError if ``texts`` is a single str
        * for each text and each (term, count) in Counter(self.terms(text)):
          if the term is in self.vocabulary (column c):
          matrix[row, c] = (1 + log(count)) * self.idf[c]
        * return l2_normalize(matrix)
        """
        # TODO: implement
        raise NotImplementedError("TODO: TfidfEmbedding.embed")

    def embed_query(self, text: str) -> FloatArray:
        return self.embed([text])[0]


# Imported for you — you will need them.
_ = (math, Counter, l2_normalize)

"""A tiny in-memory vector store with exact cosine-similarity search."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from parsrag.embeddings import Embedding, FloatArray, l2_normalize
from parsrag.types import Chunk, SearchResult


class InMemoryVectorStore:
    """Keeps chunk vectors in a NumPy matrix and does brute-force search.

    Fine for thousands of chunks; lesson 06 swaps in FAISS / pgvector.
    """

    def __init__(self, embedding: Embedding) -> None:
        self.embedding = embedding
        self._chunks: list[Chunk] = []
        self._ids: set[str] = set()
        self._matrix: FloatArray = np.zeros((0, embedding.dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Embed and store chunks. Chunk ids must be unique."""
        new = list(chunks)
        if not new:
            return
        seen = set(self._ids)
        for chunk in new:
            if chunk.id in seen:
                raise ValueError(f"duplicate chunk id: {chunk.id!r}")
            seen.add(chunk.id)
        vectors = l2_normalize(self.embedding.embed([c.text for c in new]))
        self._matrix = np.vstack([self._matrix, vectors])
        self._chunks.extend(new)
        self._ids = seen

    def search_by_vector(self, vector: FloatArray, k: int = 4) -> list[SearchResult]:
        """Return the ``k`` most cosine-similar chunks to ``vector``."""
        if k <= 0 or not self._chunks:
            return []
        query = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self._matrix @ query
        k = min(k, len(self._chunks))
        # Stable sort: ties keep insertion order, so results are deterministic.
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in order]

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        """Embed ``query`` and return the ``k`` most similar chunks."""
        return self.search_by_vector(self.embedding.embed_query(query), k)

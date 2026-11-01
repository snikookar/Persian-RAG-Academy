"""A small in-memory vector store with exact cosine-similarity search."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from parsrag.embeddings import Embedding, FloatArray, l2_normalize
from parsrag.stores.base import Filter, matches_filter
from parsrag.types import Chunk, SearchResult

FORMAT_VERSION = 1


def prepare_vectors(
    embedding: Embedding, chunks: list[Chunk], vectors: FloatArray | None
) -> FloatArray:
    """Embed ``chunks`` (or validate precomputed ``vectors``) and L2-normalise them."""
    if vectors is None:
        vectors = embedding.embed([c.text for c in chunks])
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(chunks), embedding.dim):
        raise ValueError(
            f"expected vectors of shape {(len(chunks), embedding.dim)}, got {vectors.shape}"
        )
    return l2_normalize(vectors)


def check_new_ids(existing: set[str], chunks: list[Chunk]) -> set[str]:
    """Raise ``ValueError`` on duplicate ids; return the updated id set."""
    seen = set(existing)
    for chunk in chunks:
        if chunk.id in seen:
            raise ValueError(f"duplicate chunk id: {chunk.id!r}")
        seen.add(chunk.id)
    return seen


def save_chunks(directory: Path, chunks: list[Chunk], info: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    (directory / "store.json").write_text(
        json.dumps({"format": FORMAT_VERSION, **info}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_chunks(directory: Path, dim: int) -> tuple[list[Chunk], dict[str, object]]:
    info = json.loads((directory / "store.json").read_text(encoding="utf-8"))
    if info.get("dim") != dim:
        raise ValueError(
            f"store was saved with dim={info.get('dim')} but the embedding has dim={dim}; "
            "load it with the same embedding model it was built with"
        )
    with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
    return chunks, info


class InMemoryVectorStore:
    """Keeps chunk vectors in a NumPy matrix and does brute-force (exact) search.

    Fine up to roughly a hundred thousand chunks; lesson 06 measures where
    FAISS starts to pay off and when you need a real database.
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

    def add(self, chunks: Iterable[Chunk], vectors: FloatArray | None = None) -> None:
        """Store chunks (embedded here unless ``vectors`` are given). Ids must be unique."""
        new = list(chunks)
        if not new:
            return
        seen = check_new_ids(self._ids, new)
        matrix = prepare_vectors(self.embedding, new, vectors)
        self._matrix = np.vstack([self._matrix, matrix])
        self._chunks.extend(new)
        self._ids = seen

    def delete(self, ids: Iterable[str]) -> int:
        """Remove chunks by id (unknown ids are ignored). Returns how many were removed."""
        gone = set(ids) & self._ids
        if not gone:
            return 0
        keep = [i for i, c in enumerate(self._chunks) if c.id not in gone]
        self._matrix = self._matrix[keep]
        self._chunks = [self._chunks[i] for i in keep]
        self._ids -= gone
        return len(gone)

    def vectors(self) -> FloatArray:
        """The stored (L2-normalised) vectors, one row per chunk in :attr:`chunks` order."""
        return self._matrix.copy()

    def search_by_vector(
        self, vector: FloatArray, k: int = 4, *, filter: Filter | None = None
    ) -> list[SearchResult]:
        """Return the ``k`` most cosine-similar chunks that pass ``filter``."""
        if k <= 0 or not self._chunks:
            return []
        query = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self._matrix @ query
        if filter is not None:
            mask = np.fromiter(
                (matches_filter(c, filter) for c in self._chunks), dtype=bool, count=len(self)
            )
            scores = np.where(mask, scores, -np.inf)
            k = min(k, int(mask.sum()))
        k = min(k, len(self._chunks))
        # Stable sort: ties keep insertion order, so results are deterministic.
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in order]

    def search(self, query: str, k: int = 4, *, filter: Filter | None = None) -> list[SearchResult]:
        """Embed ``query`` and return the ``k`` most similar chunks."""
        return self.search_by_vector(self.embedding.embed_query(query), k, filter=filter)

    # ------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        """Write ``vectors.npy``, ``chunks.jsonl`` and ``store.json`` to ``directory``."""
        path = Path(directory)
        save_chunks(path, self._chunks, {"kind": "memory", "dim": self.embedding.dim})
        np.save(path / "vectors.npy", self._matrix)

    @classmethod
    def load(cls, directory: str | Path, embedding: Embedding) -> InMemoryVectorStore:
        """Load a saved store; ``embedding`` must be the model it was built with."""
        path = Path(directory)
        chunks, _ = load_chunks(path, embedding.dim)
        store = cls(embedding)
        store.add(chunks, np.load(path / "vectors.npy"))
        return store

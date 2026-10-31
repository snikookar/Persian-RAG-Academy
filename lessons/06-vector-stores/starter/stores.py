"""Lesson 06 starter — vector stores.

Fill in every function marked TODO, then check your work:

    pip install -e ".[faiss]"     # for the FAISS checks
    python -m grader run 06

Everything else (protocol, id checks, the FAISS index factory, persistence of
the chunk list) is written for you. Reference solution: src/parsrag/stores/
(try not to peek until you have tried!).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from parsrag.embeddings import Embedding, FloatArray, l2_normalize
from parsrag.types import Chunk, SearchResult

# ============================================================ filters & protocol

Filter = Mapping[str, Any] | Callable[[Chunk], bool]


def chunk_field(chunk: Chunk, key: str) -> Any:
    """``doc_id``/``id`` come from the chunk itself, everything else from its metadata."""
    if key in ("doc_id", "id"):
        return getattr(chunk, key)
    return chunk.metadata.get(key)


def matches_filter(chunk: Chunk, flt: Filter | None) -> bool:
    """True if ``chunk`` passes ``flt`` (``None`` lets everything through).

    * None -> True;  a callable -> bool(flt(chunk))
    * a dict: every key must match. Get the value with chunk_field(chunk, key);
      if the wanted value is a list/tuple/set/frozenset the value must be IN it,
      otherwise it must be EQUAL to it.
    """
    # TODO: implement
    raise NotImplementedError("TODO: matches_filter")


@runtime_checkable
class VectorStore(Protocol):
    """What every parsrag vector store can do."""

    def __len__(self) -> int: ...

    def add(self, chunks: Iterable[Chunk], vectors: FloatArray | None = None) -> None: ...

    def search(
        self, query: str, k: int = 4, *, filter: Filter | None = None
    ) -> list[SearchResult]: ...

    def search_by_vector(
        self, vector: FloatArray, k: int = 4, *, filter: Filter | None = None
    ) -> list[SearchResult]: ...


# ===================================================================== in-memory

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

    def search_by_vector(
        self, vector: FloatArray, k: int = 4, *, filter: Filter | None = None
    ) -> list[SearchResult]:
        """Return the ``k`` most cosine-similar chunks that pass ``filter``.

        * [] if k <= 0 or the store is empty
        * query = l2_normalize(vector as float32, shape (1, dim))[0]; scores = matrix @ query
        * with a filter: a boolean mask via matches_filter for every chunk; set the
          scores of rejected chunks to -inf and lower k to the number of survivors
        * order = np.argsort(-scores, kind="stable")[:k]   (stable: ties stay in order)
        * return SearchResult(chunk=..., score=float(score)) for each index in order
        """
        # TODO: implement
        raise NotImplementedError("TODO: InMemoryVectorStore.search_by_vector")

    def search(self, query: str, k: int = 4, *, filter: Filter | None = None) -> list[SearchResult]:
        """Embed ``query`` and return the ``k`` most similar chunks."""
        return self.search_by_vector(self.embedding.embed_query(query), k, filter=filter)

    # ------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        """Write ``vectors.npy``, ``chunks.jsonl`` and ``store.json`` to ``directory``.

        save_chunks(path, self._chunks, {"kind": "memory", "dim": self.embedding.dim})
        writes the last two; add np.save(path / "vectors.npy", self._matrix).
        """
        # TODO: implement
        raise NotImplementedError("TODO: InMemoryVectorStore.save")

    @classmethod
    def load(cls, directory: str | Path, embedding: Embedding) -> InMemoryVectorStore:
        """Load a saved store; ``embedding`` must be the model it was built with.

        chunks, _ = load_chunks(path, embedding.dim)  (raises if the dim differs);
        create cls(embedding) and add(chunks, np.load(path / "vectors.npy")).
        """
        # TODO: implement
        raise NotImplementedError("TODO: InMemoryVectorStore.load")


# ========================================================================= FAISS

IndexKind = Literal["flat", "hnsw", "ivf"]


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FaissVectorStore needs faiss. Install it with: pip install 'parsrag[faiss]'"
        ) from exc
    return faiss


class FaissVectorStore:
    """Vector store on a FAISS index (exact ``flat`` or approximate ``hnsw`` / ``ivf``)."""

    def __init__(
        self,
        embedding: Embedding,
        index: IndexKind = "flat",
        *,
        m: int = 32,
        ef_construction: int = 80,
        ef_search: int = 64,
        nlist: int = 64,
        nprobe: int = 8,
    ) -> None:
        self._faiss = _import_faiss()
        if index not in ("flat", "hnsw", "ivf"):
            raise ValueError("index must be 'flat', 'hnsw' or 'ivf'")
        self.embedding = embedding
        self.kind: IndexKind = index
        self.params = {"m": m, "ef_construction": ef_construction, "ef_search": ef_search,
                       "nlist": nlist, "nprobe": nprobe}  # fmt: skip
        self._chunks: list[Chunk] = []
        self._ids: set[str] = set()
        self.index = self._new_index()

    def _new_index(self) -> Any:
        faiss, dim, p = self._faiss, self.embedding.dim, self.params
        ip = faiss.METRIC_INNER_PRODUCT
        if self.kind == "flat":
            return faiss.IndexFlatIP(dim)
        if self.kind == "hnsw":
            index = faiss.IndexHNSWFlat(dim, p["m"], ip)
            index.hnsw.efConstruction = p["ef_construction"]
            index.hnsw.efSearch = p["ef_search"]
            return index
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, p["nlist"], ip)
        index.nprobe = p["nprobe"]
        self._quantizer = quantizer  # the IVF index does not own it: keep it alive
        return index

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: Iterable[Chunk], vectors: FloatArray | None = None) -> None:
        """Store chunks (embedded here unless ``vectors`` are given). Ids must be unique.

        An IVF index is trained on the first batch, which must contain at
        least ``nlist`` vectors (FAISS recommends ~40 per cell).
        """
        new = list(chunks)
        if not new:
            return
        seen = check_new_ids(self._ids, new)
        matrix = np.ascontiguousarray(prepare_vectors(self.embedding, new, vectors))
        if not self.index.is_trained:
            if len(matrix) < self.params["nlist"]:
                raise ValueError(
                    f"IVF needs at least nlist={self.params['nlist']} vectors to train, "
                    f"got {len(matrix)}; lower nlist or add more chunks at once"
                )
            self.index.train(matrix)
        self.index.add(matrix)
        if self.kind == "ivf":
            self.index.make_direct_map()  # lets reconstruct() find vectors by id
        self._chunks.extend(new)
        self._ids = seen

    def _exact_filtered(self, query: FloatArray, k: int, flt: Filter | None) -> list[SearchResult]:
        """Brute-force fallback: score every chunk that passes ``flt``."""
        ids = [i for i, c in enumerate(self._chunks) if matches_filter(c, flt)]
        if not ids:
            return []
        vectors = self.index.reconstruct_batch(np.asarray(ids, dtype=np.int64))
        scores = vectors @ query[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(chunk=self._chunks[ids[j]], score=float(scores[j])) for j in order]

    def search_by_vector(
        self, vector: FloatArray, k: int = 4, *, filter: Filter | None = None
    ) -> list[SearchResult]:
        """Search the FAISS index; apply ``filter`` afterwards, fetching more if needed.

        * [] if k <= 0 or the store is empty; query = l2_normalize(... shape (1, dim))
        * fetch = k without a filter, else min(len(self), 4 * k)
        * loop: scores, ids = self.index.search(query, min(fetch, len(self)))
          keep (score, id) pairs with id >= 0 whose chunk passes matches_filter;
          if you have >= k results: return the first k;
          if fetch >= len(self) (you searched everything and are still short):
          return self._exact_filtered(query, k, filter) — approximate indexes
          can miss rare matches; otherwise fetch *= 2 and search again
        """
        # TODO: implement
        raise NotImplementedError("TODO: FaissVectorStore.search_by_vector")

    def search(self, query: str, k: int = 4, *, filter: Filter | None = None) -> list[SearchResult]:
        return self.search_by_vector(self.embedding.embed_query(query), k, filter=filter)

    # ------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        """Write ``index.faiss``, ``chunks.jsonl`` and ``store.json``."""
        path = Path(directory)
        info = {"kind": f"faiss-{self.kind}", "dim": self.embedding.dim, "params": self.params}
        save_chunks(path, self._chunks, info)
        self._faiss.write_index(self.index, str(path / "index.faiss"))

    @classmethod
    def load(cls, directory: str | Path, embedding: Embedding) -> FaissVectorStore:
        path = Path(directory)
        chunks, info = load_chunks(path, embedding.dim)
        kind = str(info["kind"]).removeprefix("faiss-")
        store = cls(embedding, kind, **info["params"])  # type: ignore[arg-type]
        store.index = store._faiss.read_index(str(path / "index.faiss"))
        if kind == "ivf":
            store.index.nprobe = store.params["nprobe"]
            store.index.make_direct_map()
        store._chunks = chunks
        store._ids = {c.id for c in chunks}
        return store

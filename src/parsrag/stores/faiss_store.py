"""FAISS-backed vector store (``pip install 'parsrag[faiss]'``).

Three index types, all on L2-normalised vectors with inner product (= cosine):

* ``"flat"`` — exact search, like :class:`~parsrag.stores.memory.InMemoryVectorStore`
  but in optimised C++ (SIMD, multithreading).
* ``"hnsw"`` — a navigable small-world *graph*: approximate, very fast, no
  training. ``m`` = links per node; ``ef_search`` = how many candidates to
  explore per query (higher = more accurate, slower).
* ``"ivf"`` — an *inverted file*: vectors are clustered into ``nlist`` cells
  (needs training) and a query only scans the ``nprobe`` closest cells.

FAISS has no metadata, so filters are applied *after* searching: we fetch more
candidates than ``k`` and keep doubling until enough of them pass the filter.
An approximate index may still miss matches of a very selective filter (a graph
search does not visit every node), so as a last resort we score the matching
chunks exactly, with vectors reconstructed from the index.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from parsrag.embeddings import Embedding, FloatArray, l2_normalize
from parsrag.stores.base import Filter, matches_filter
from parsrag.stores.memory import check_new_ids, load_chunks, prepare_vectors, save_chunks
from parsrag.types import Chunk, SearchResult

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
        if k <= 0 or not self._chunks:
            return []
        query = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        fetch = k if filter is None else min(len(self), 4 * k)
        while True:
            scores, ids = self.index.search(query, min(fetch, len(self)))
            results = [
                SearchResult(chunk=self._chunks[i], score=float(s))
                for s, i in zip(scores[0], ids[0], strict=True)
                if i >= 0 and matches_filter(self._chunks[i], filter)
            ]
            if len(results) >= k:
                return results[:k]
            if fetch >= len(self):  # searched "everything" and still short
                return self._exact_filtered(query, k, filter)
            fetch *= 2  # not enough survived the filter: look further

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

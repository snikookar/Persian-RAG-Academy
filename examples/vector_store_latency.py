"""Exact vs approximate search: in-memory NumPy vs FAISS flat / HNSW / IVF (lesson 06).

    pip install -e ".[faiss]"
    python examples/vector_store_latency.py                  # n = 2,000 and 10,000
    python examples/vector_store_latency.py --sizes 50000    # try bigger on your machine

Synthetic data: unit vectors drawn around 50 random cluster centres (real
embeddings are clustered too, which is what makes IVF work). For every store
we report build time, mean single-query latency (k=10, one thread) and
Recall@10 against exact search. Timings depend on your CPU — rerun locally.
"""

from __future__ import annotations

import os

# One thread everywhere (NumPy's BLAS and FAISS) so the comparison is fair.
# Must happen before NumPy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from parsrag.embeddings import FloatArray, l2_normalize  # noqa: E402
from parsrag.stores import FaissVectorStore, InMemoryVectorStore  # noqa: E402
from parsrag.types import Chunk  # noqa: E402


class FixedDim:
    """A stand-in 'embedding' for precomputed vectors: only ``dim`` is used."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> FloatArray:
        raise NotImplementedError("vectors are passed in directly")

    def embed_query(self, text: str) -> FloatArray:
        raise NotImplementedError("vectors are passed in directly")


def clustered(n: int, dim: int, rng: np.random.Generator, centres: FloatArray) -> FloatArray:
    labels = rng.integers(0, len(centres), size=n)
    noise = rng.normal(scale=0.6 / np.sqrt(dim) * 4, size=(n, dim))
    return l2_normalize((centres[labels] + noise).astype(np.float32))


def run(
    n: int, dim: int, queries: int, k: int, seed: int = 0
) -> list[tuple[str, float, float, float]]:
    rng = np.random.default_rng(seed)
    centres = l2_normalize(rng.normal(size=(50, dim)).astype(np.float32))
    data = clustered(n, dim, rng, centres)
    qs = clustered(queries, dim, rng, centres)
    chunks = [Chunk(id=str(i), doc_id=str(i), text="") for i in range(n)]
    emb = FixedDim(dim)
    nlist = max(8, n // 100)
    factories: list[tuple[str, Callable[[], object]]] = [
        ("in-memory (NumPy, exact)", lambda: InMemoryVectorStore(emb)),
        ("FAISS flat (exact)", lambda: FaissVectorStore(emb, "flat")),
        ("FAISS HNSW (m=32, ef=64)", lambda: FaissVectorStore(emb, "hnsw", m=32, ef_search=64)),
        (
            f"FAISS IVF (nlist={nlist}, nprobe=8)",
            lambda: FaissVectorStore(emb, "ivf", nlist=nlist, nprobe=8),
        ),
    ]
    exact: list[set[str]] = []
    rows = []
    for name, make in factories:
        start = time.perf_counter()
        store = make()
        store.add(chunks, data)  # type: ignore[attr-defined]
        build = time.perf_counter() - start
        store.search_by_vector(qs[0], k)  # type: ignore[attr-defined]  # warm-up
        found = []
        start = time.perf_counter()
        for q in qs:
            found.append({r.chunk.id for r in store.search_by_vector(q, k)})  # type: ignore[attr-defined]
        latency_ms = (time.perf_counter() - start) / queries * 1000
        if not exact:
            exact = found
        recall = float(np.mean([len(f & e) / k for f, e in zip(found, exact, strict=True)]))
        rows.append((name, build, latency_ms, recall))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[2_000, 10_000])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    import faiss

    faiss.omp_set_num_threads(1)  # single-thread for a fair, reproducible comparison
    for n in args.sizes:
        print(f"\nn = {n:,} vectors, dim = {args.dim}, {args.queries} queries, k = {args.k}")
        print(f"{'store':<34} {'build (s)':>9} {'query (ms)':>10} {'recall@10':>9}")
        for name, build, latency, recall in run(n, args.dim, args.queries, args.k):
            print(f"{name:<34} {build:>9.3f} {latency:>10.3f} {recall:>9.3f}")


if __name__ == "__main__":
    main()

"""Vector stores (lesson 06).

* :class:`InMemoryVectorStore` — exact NumPy search; part of the core install.
* :class:`~parsrag.stores.faiss_store.FaissVectorStore` — FAISS flat/HNSW/IVF
  (``faiss`` extra).
* :class:`~parsrag.stores.pgvector_store.PgVectorStore` — PostgreSQL + pgvector
  (``pgvector`` extra).

The optional ones are imported lazily: ``from parsrag.stores import FaissVectorStore``
works without FAISS installed until you actually create one.
"""

from __future__ import annotations

from typing import Any

from parsrag.stores.base import Filter, VectorStore, chunk_field, matches_filter
from parsrag.stores.memory import InMemoryVectorStore

__all__ = [
    "FaissVectorStore",
    "Filter",
    "InMemoryVectorStore",
    "PgVectorStore",
    "VectorStore",
    "chunk_field",
    "matches_filter",
]


def __getattr__(name: str) -> Any:
    if name == "FaissVectorStore":
        from parsrag.stores.faiss_store import FaissVectorStore

        return FaissVectorStore
    if name == "PgVectorStore":
        from parsrag.stores.pgvector_store import PgVectorStore

        return PgVectorStore
    raise AttributeError(f"module 'parsrag.stores' has no attribute {name!r}")

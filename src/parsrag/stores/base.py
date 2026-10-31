"""The vector-store protocol and metadata filters (lesson 06).

Every store in :mod:`parsrag.stores` offers the same small interface, so the
rest of the pipeline never cares where the vectors live::

    store.add(chunks)                       # embed and store (or pass vectors=)
    store.search("پرسش", k=4)               # embed the query and search
    store.search("پرسش", k=4, filter={"doc_id": "kahroba-faq"})
    len(store)

A **filter** is either a dict or a function:

* ``{"doc_id": "a"}`` — equality on ``chunk.doc_id`` or on a ``chunk.metadata`` key;
* ``{"section": ["الف", "ب"]}`` — a list/tuple/set means "any of these";
* ``lambda chunk: chunk.metadata["page"] > 1`` — any predicate (in-memory and
  FAISS only; a database needs filters it can translate to SQL).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from parsrag.embeddings import FloatArray
from parsrag.types import Chunk, SearchResult

Filter = Mapping[str, Any] | Callable[[Chunk], bool]


def chunk_field(chunk: Chunk, key: str) -> Any:
    """``doc_id``/``id`` come from the chunk itself, everything else from its metadata."""
    if key in ("doc_id", "id"):
        return getattr(chunk, key)
    return chunk.metadata.get(key)


def matches_filter(chunk: Chunk, flt: Filter | None) -> bool:
    """True if ``chunk`` passes ``flt`` (``None`` lets everything through)."""
    if flt is None:
        return True
    if callable(flt):
        return bool(flt(chunk))
    for key, wanted in flt.items():
        value = chunk_field(chunk, key)
        if isinstance(wanted, (list, tuple, set, frozenset)):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


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

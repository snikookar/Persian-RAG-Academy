"""PostgreSQL + pgvector store (``pip install 'parsrag[pgvector]'``).

Use a real database when you need what a NumPy matrix or a FAISS file cannot
give you: many processes reading and writing at once, transactions, backups,
access control, and SQL filters next to the vectors. The table looks like::

    id text PRIMARY KEY, doc_id text, text text, metadata jsonb, embedding vector(dim)

and search is ``ORDER BY embedding <=> query LIMIT k`` (``<=>`` = cosine
distance; score = 1 - distance). An HNSW index makes it approximate and fast.

Filters must be dicts — they are translated to SQL (``metadata @> ...``).
Tests talk to a real database only when ``PARSRAG_TEST_DATABASE_URL`` is set.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from parsrag.embeddings import Embedding, FloatArray, l2_normalize
from parsrag.stores.base import Filter
from parsrag.stores.memory import prepare_vectors
from parsrag.types import Chunk, SearchResult

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")
_COLUMNS = ("id", "doc_id")


def filter_to_sql(flt: Filter | None) -> tuple[str, list[Any]]:
    """Translate a dict filter into a SQL ``WHERE`` fragment and its parameters.

    >>> filter_to_sql({"doc_id": "a", "section": ["x", "y"], "page": 2})
    ('doc_id = %s AND metadata->>%s = ANY(%s) AND metadata @> %s::jsonb',
     ['a', 'section', ['x', 'y'], '{"page": 2}'])
    """
    if flt is None or not flt:
        return "TRUE", []
    if callable(flt) or not isinstance(flt, Mapping):
        raise TypeError("PgVectorStore filters must be dicts (they are translated to SQL)")
    clauses: list[str] = []
    params: list[Any] = []
    contains: dict[str, Any] = {}
    for key, wanted in flt.items():
        many = isinstance(wanted, (list, tuple, set, frozenset))
        if key in _COLUMNS:
            clauses.append(f"{key} = ANY(%s)" if many else f"{key} = %s")
            params.append(list(wanted) if many else wanted)
        elif many:
            clauses.append("metadata->>%s = ANY(%s)")
            params.extend([key, [str(v) for v in wanted]])
        else:
            contains[key] = wanted
    if contains:
        clauses.append("metadata @> %s::jsonb")
        params.append(json.dumps(contains, ensure_ascii=False))
    return " AND ".join(clauses), params


class PgVectorStore:
    """Vector store in a PostgreSQL table with the pgvector extension."""

    def __init__(
        self,
        embedding: Embedding,
        dsn: str,
        *,
        table: str = "parsrag_chunks",
        create: bool = True,
        hnsw: bool = True,
        connection: Any | None = None,
    ) -> None:
        if not _IDENTIFIER.fullmatch(table):
            raise ValueError(f"invalid table name: {table!r}")
        self.embedding = embedding
        self.table = table
        self._conn = connection if connection is not None else self._connect(dsn)
        if create:
            self.create_table(hnsw=hnsw)

    @staticmethod
    def _connect(dsn: str) -> Any:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise ImportError(
                "PgVectorStore needs psycopg and pgvector. "
                "Install them with: pip install 'parsrag[pgvector]'"
            ) from exc
        conn = psycopg.connect(dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def create_table(self, *, hnsw: bool = True) -> None:
        t, dim = self.table, self.embedding.dim
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {t} (id text PRIMARY KEY, doc_id text NOT NULL, "
            f"text text NOT NULL, metadata jsonb NOT NULL DEFAULT '{{}}', "
            f"embedding vector({dim}) NOT NULL)"
        )
        if hnsw:
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_embedding_hnsw ON {t} "
                "USING hnsw (embedding vector_cosine_ops)"
            )

    def drop(self) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS {self.table}")

    def __len__(self) -> int:
        row = self._conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()
        return int(row[0])

    def add(self, chunks: Iterable[Chunk], vectors: FloatArray | None = None) -> None:
        """Insert chunks; a duplicate id raises (the whole batch is rolled back)."""
        new = list(chunks)
        if not new:
            return
        ids = [c.id for c in new]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk id in batch")
        matrix = prepare_vectors(self.embedding, new, vectors)
        rows = [
            (c.id, c.doc_id, c.text, json.dumps(c.metadata, ensure_ascii=False), v)
            for c, v in zip(new, matrix, strict=True)
        ]
        with self._conn.transaction(), self._conn.cursor() as cur:
            existing = cur.execute(
                f"SELECT id FROM {self.table} WHERE id = ANY(%s)", (ids,)
            ).fetchall()
            if existing:
                raise ValueError(f"duplicate chunk id: {existing[0][0]!r}")
            cur.executemany(
                f"INSERT INTO {self.table} (id, doc_id, text, metadata, embedding) "
                "VALUES (%s, %s, %s, %s::jsonb, %s)",
                rows,
            )

    def delete(self, ids: Iterable[str]) -> int:
        """Delete chunks by id. Returns how many rows were removed."""
        wanted = list(ids)
        if not wanted:
            return 0
        cur = self._conn.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (wanted,))
        return int(cur.rowcount or 0)

    def search_by_vector(
        self, vector: FloatArray, k: int = 4, *, filter: Filter | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            return []
        where, params = filter_to_sql(filter)
        query = l2_normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        rows = self._conn.execute(
            f"SELECT id, doc_id, text, metadata, 1 - (embedding <=> %s) AS score "
            f"FROM {self.table} WHERE {where} ORDER BY embedding <=> %s LIMIT %s",
            [query, *params, query, k],
        ).fetchall()
        return [
            SearchResult(
                chunk=Chunk(id=r[0], doc_id=r[1], text=r[2], metadata=_as_dict(r[3])),
                score=float(r[4]),
            )
            for r in rows
        ]

    def search(self, query: str, k: int = 4, *, filter: Filter | None = None) -> list[SearchResult]:
        return self.search_by_vector(self.embedding.embed_query(query), k, filter=filter)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else json.loads(value)

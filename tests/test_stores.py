from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from parsrag import Chunk, HashingEmbedding, InMemoryVectorStore
from parsrag.stores import VectorStore, matches_filter
from parsrag.stores.pgvector_store import filter_to_sql

DB_URL = os.environ.get("PARSRAG_TEST_DATABASE_URL")
TEXTS = {
    "ship": ("kahroba-shipping", "ارسال سفارش با پیک در تهران", {"page": 1, "section": "ارسال"}),
    "lib": ("sepidar-hours", "ساعت کاری کتابخانهٔ عمومی", {"page": 1, "section": "ساعت"}),
    "pay": ("kahroba-payment", "پرداخت اقساطی و درگاه بانکی", {"page": 2, "section": "پرداخت"}),
    "lib2": ("sepidar-rules", "قوانین امانت کتاب در کتابخانه", {"page": 3, "section": "امانت"}),
}


def _chunks() -> list[Chunk]:
    return [Chunk(id=k, doc_id=d, text=t, metadata=m) for k, (d, t, m) in TEXTS.items()]


def _stores() -> list[str]:
    kinds = ["memory"]
    try:
        import faiss  # noqa: F401

        kinds += ["faiss-flat", "faiss-hnsw"]
    except ImportError:
        pass
    return kinds


def _make(kind: str):  # type: ignore[no-untyped-def]
    emb = HashingEmbedding()
    if kind == "memory":
        return InMemoryVectorStore(emb)
    from parsrag.stores import FaissVectorStore

    return FaissVectorStore(emb, kind.removeprefix("faiss-"))  # type: ignore[arg-type]


# ----------------------------------------------------------------- filters


def test_matches_filter() -> None:
    chunk = _chunks()[0]
    assert matches_filter(chunk, None)
    assert matches_filter(chunk, {"doc_id": "kahroba-shipping"})
    assert matches_filter(chunk, {"page": 1, "section": ["ارسال", "x"]})
    assert not matches_filter(chunk, {"page": 2})
    assert not matches_filter(chunk, {"missing": 1})
    assert matches_filter(chunk, lambda c: c.metadata["page"] < 2)


# ------------------------------------------------- behaviour of every store


@pytest.fixture(params=_stores())
def store(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    s = _make(request.param)
    s.add(_chunks())
    return s


def test_store_protocol_and_search(store) -> None:  # type: ignore[no-untyped-def]
    assert isinstance(store, VectorStore)
    assert len(store) == 4
    results = store.search("ساعت کاری کتابخانه", k=2)
    assert results[0].chunk.id == "lib" and len(results) == 2
    assert results[0].score >= results[1].score


def test_store_filters(store) -> None:  # type: ignore[no-untyped-def]
    only = store.search("کتابخانه", k=4, filter={"doc_id": "sepidar-rules"})
    assert [r.chunk.id for r in only] == ["lib2"]
    pages = store.search("کتابخانه", k=4, filter=lambda c: c.metadata["page"] >= 2)
    assert {r.chunk.id for r in pages} == {"pay", "lib2"}
    assert store.search("کتابخانه", k=3, filter={"section": ["ارسال", "پرداخت"]})[0].chunk.id in {
        "ship",
        "pay",
    }
    assert store.search("کتابخانه", k=4, filter={"doc_id": "nope"}) == []


def test_store_rejects_duplicates_atomically(store) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="duplicate"):
        store.add([Chunk(id="new", doc_id="x", text="الف"), Chunk(id="lib", doc_id="x", text="ب")])
    assert len(store) == 4


def test_precomputed_vectors_are_checked(store) -> None:  # type: ignore[no-untyped-def]
    emb = store.embedding
    vec = emb.embed(["متن تازه"])
    store.add([Chunk(id="v", doc_id="v", text="متن تازه")], vec * 3)  # normalised on the way in
    assert store.search_by_vector(vec[0], k=1)[0].chunk.id == "v"
    with pytest.raises(ValueError, match="shape"):
        store.add([Chunk(id="w", doc_id="w", text="?")], np.zeros((1, 7), dtype=np.float32))


@pytest.mark.parametrize("kind", _stores())
def test_save_and_load_round_trip(kind: str, tmp_path: Path) -> None:
    store = _make(kind)
    store.add(_chunks())
    store.save(tmp_path / "idx")
    loaded = type(store).load(tmp_path / "idx", HashingEmbedding())
    assert len(loaded) == 4
    a, b = store.search("پرداخت قسطی", k=3), loaded.search("پرداخت قسطی", k=3)
    assert [r.chunk.id for r in a] == [r.chunk.id for r in b]
    assert loaded.chunks[0].metadata == {"page": 1, "section": "ارسال"}
    with pytest.raises(ValueError, match="dim"):
        type(store).load(tmp_path / "idx", HashingEmbedding(dim=64))


# -------------------------------------------------------------------- FAISS


def test_faiss_requires_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faiss", None)
    from parsrag.stores.faiss_store import FaissVectorStore

    with pytest.raises(ImportError, match=r"parsrag\[faiss\]"):
        FaissVectorStore(HashingEmbedding())


def test_faiss_ivf_and_approximate_recall() -> None:
    pytest.importorskip("faiss", reason="pip install -e '.[faiss]'")
    from parsrag.stores import FaissVectorStore

    rng = np.random.default_rng(0)
    centres = rng.normal(size=(20, 64)).astype(np.float32)
    data = centres[rng.integers(0, 20, 2000)] + rng.normal(scale=0.3, size=(2000, 64))
    queries = centres[rng.integers(0, 20, 50)] + rng.normal(scale=0.3, size=(50, 64))
    chunks = [Chunk(id=str(i), doc_id="d", text="") for i in range(2000)]
    emb = HashingEmbedding(dim=64)
    exact = InMemoryVectorStore(emb)
    exact.add(chunks, data.astype(np.float32))
    for kind, kwargs in (("hnsw", {}), ("ivf", {"nlist": 20, "nprobe": 4})):
        approx = FaissVectorStore(emb, kind, **kwargs)  # type: ignore[arg-type]
        approx.add(chunks, data.astype(np.float32))
        recall = np.mean([
            len({r.chunk.id for r in approx.search_by_vector(q, 10)}
                & {r.chunk.id for r in exact.search_by_vector(q, 10)}) / 10
            for q in queries
        ])  # fmt: skip
        assert recall > 0.9, (kind, recall)
    with pytest.raises(ValueError, match="nlist"):
        FaissVectorStore(emb, "ivf", nlist=64).add(chunks[:10], data[:10].astype(np.float32))
    with pytest.raises(ValueError):
        FaissVectorStore(emb, "lsh")  # type: ignore[arg-type]


# ----------------------------------------------------------------- pgvector


def test_filter_to_sql() -> None:
    assert filter_to_sql(None) == ("TRUE", [])
    sql, params = filter_to_sql({"doc_id": "a", "section": ["x", "y"], "page": 2})
    assert sql == "doc_id = %s AND metadata->>%s = ANY(%s) AND metadata @> %s::jsonb"
    assert params == ["a", "section", ["x", "y"], '{"page": 2}']
    assert filter_to_sql({"id": ["a", "b"]}) == ("id = ANY(%s)", [["a", "b"]])
    with pytest.raises(TypeError):
        filter_to_sql(lambda c: True)


def test_pgvector_rejects_bad_table_name() -> None:
    from parsrag.stores import PgVectorStore

    with pytest.raises(ValueError, match="table"):
        PgVectorStore(HashingEmbedding(), "", table="x; DROP TABLE y", connection=object())


@pytest.mark.skipif(
    not DB_URL,
    reason="PgVectorStore tests need PARSRAG_TEST_DATABASE_URL=postgresql://... (with pgvector)",
)
def test_pgvector_round_trip() -> None:
    pytest.importorskip("psycopg", reason="pip install -e '.[pgvector]'")
    from parsrag.stores import PgVectorStore

    table = f"parsrag_test_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(HashingEmbedding(), DB_URL or "", table=table)
    try:
        store.add(_chunks())
        assert len(store) == 4
        assert store.search("ساعت کاری کتابخانه", k=1)[0].chunk.id == "lib"
        only = store.search("کتابخانه", k=4, filter={"doc_id": "sepidar-rules"})
        assert [r.chunk.id for r in only] == ["lib2"]
        pages = store.search("کتابخانه", k=4, filter={"section": ["ارسال", "پرداخت"]})
        assert {r.chunk.id for r in pages} == {"ship", "pay"}
        with pytest.raises(ValueError, match="duplicate"):
            store.add([Chunk(id="lib", doc_id="x", text="ب")])
        assert len(store) == 4
    finally:
        store.drop()


def test_latency_example_runs_small() -> None:
    pytest.importorskip("faiss", reason="pip install -e '.[faiss]'")
    import importlib.util

    from grader.paths import REPO_ROOT

    path = REPO_ROOT / "examples" / "vector_store_latency.py"
    spec = importlib.util.spec_from_file_location("vector_store_latency", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["vector_store_latency"] = module
    spec.loader.exec_module(module)
    rows = module.run(n=1000, dim=32, queries=20, k=10)
    assert [r[0].split()[0] for r in rows] == ["in-memory", "FAISS", "FAISS", "FAISS"]
    assert rows[0][3] == rows[1][3] == 1.0  # both exact
    assert all(r[3] > 0.8 for r in rows)

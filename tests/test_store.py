from __future__ import annotations

import numpy as np
import pytest

from parsrag import Chunk, HashingEmbedding, InMemoryVectorStore


def _chunks() -> list[Chunk]:
    texts = {
        "ship": "ارسال سفارش با پیک در تهران",
        "lib": "ساعت کاری کتابخانهٔ عمومی",
        "pay": "پرداخت اقساطی و درگاه بانکی",
    }
    return [Chunk(id=k, doc_id=k, text=v) for k, v in texts.items()]


def test_add_and_search() -> None:
    store = InMemoryVectorStore(HashingEmbedding())
    store.add(_chunks())
    assert len(store) == 3
    results = store.search("کتابخانه کی باز است", k=2)
    assert results[0].chunk.id == "lib"
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_k_larger_than_store_and_non_positive_k() -> None:
    store = InMemoryVectorStore(HashingEmbedding())
    assert store.search("هر چیزی") == []
    store.add(_chunks())
    assert len(store.search("پیک", k=10)) == 3
    assert store.search("پیک", k=0) == []


def test_duplicate_ids_rejected_atomically() -> None:
    store = InMemoryVectorStore(HashingEmbedding())
    store.add(_chunks())
    with pytest.raises(ValueError, match="duplicate"):
        store.add([Chunk(id="new", doc_id="x", text="الف"), Chunk(id="lib", doc_id="x", text="ب")])
    assert len(store) == 3  # nothing partially added
    with pytest.raises(ValueError):
        store.add([Chunk(id="a", doc_id="x", text="الف"), Chunk(id="a", doc_id="x", text="ب")])


def test_add_empty_is_noop() -> None:
    store = InMemoryVectorStore(HashingEmbedding())
    store.add([])
    assert len(store) == 0


def test_search_by_vector_normalizes_query() -> None:
    emb = HashingEmbedding()
    store = InMemoryVectorStore(emb)
    store.add(_chunks())
    q = emb.embed_query("پرداخت اقساطی")
    a = store.search_by_vector(q, k=1)
    b = store.search_by_vector(q * 5.0, k=1)
    assert a[0].chunk.id == b[0].chunk.id == "pay"
    assert a[0].score == pytest.approx(b[0].score)
    assert a[0].score <= 1.0 + 1e-6


def test_ties_are_deterministic() -> None:
    store = InMemoryVectorStore(HashingEmbedding())
    store.add([Chunk(id=str(i), doc_id="d", text="یکسان") for i in range(5)])
    assert [r.chunk.id for r in store.search("یکسان", k=5)] == ["0", "1", "2", "3", "4"]
    assert np.allclose([r.score for r in store.search("یکسان", k=5)], 1.0)

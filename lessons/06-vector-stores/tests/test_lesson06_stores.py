"""Lesson 06 — vector stores. Run with: python -m grader run 06"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from parsrag import Chunk, HashingEmbedding
from parsrag.eval import evaluate_retrieval
from parsrag.eval.benchmark import index_documents, load_fixture_benchmark
from parsrag.normalize import normalize

hint = pytest.mark.hint
EXTRA = "pip install -e '.[faiss]'  (بستهٔ اختیاری faiss نصب نیست)"

CHUNKS = [
    Chunk(
        id="ship",
        doc_id="kahroba-shipping",
        text="ارسال سفارش با پیک در تهران",
        metadata={"page": 1, "section": "ارسال"},
    ),
    Chunk(
        id="lib",
        doc_id="sepidar-hours",
        text="ساعت کاری کتابخانهٔ عمومی",
        metadata={"page": 1, "section": "ساعت"},
    ),
    Chunk(
        id="pay",
        doc_id="kahroba-payment",
        text="پرداخت اقساطی و درگاه بانکی",
        metadata={"page": 2, "section": "پرداخت"},
    ),
    Chunk(
        id="lib2",
        doc_id="sepidar-rules",
        text="قوانین امانت کتاب در کتابخانه",
        metadata={"page": 3, "section": "امانت"},
    ),
]


# --------------------------------------------------------------------- filters


@hint(
    fa="برای هر کلید مقدار را با chunk_field بخوانید؛ اگر مقدار خواسته‌شده فهرست بود، «عضو بودن» را بررسی کنید.",
    en="Read each value with chunk_field; a list/tuple/set means 'any of these'.",
)
@pytest.mark.parametrize(
    ("flt", "expected"),
    [
        (None, True),
        ({"doc_id": "kahroba-shipping"}, True),
        ({"doc_id": "x"}, False),
        ({"page": 1, "section": ["ارسال", "x"]}, True),
        ({"page": 1, "section": ["x"]}, False),
        ({"missing": 1}, False),
        (lambda c: c.metadata["page"] < 2, True),
        (lambda c: c.id == "nope", False),
    ],
)
def test_matches_filter(vs: ModuleType, flt: object, expected: bool) -> None:
    """matches_filter supports equality, 'any of' lists and predicates"""
    assert vs.matches_filter(CHUNKS[0], flt) is expected


# ------------------------------------------------------------------ in-memory


@pytest.fixture
def memory(vs: ModuleType):  # type: ignore[no-untyped-def]
    store = vs.InMemoryVectorStore(HashingEmbedding())
    store.add(CHUNKS)
    return store


@hint(
    fa="امتیاز = ماتریس @ پرسشِ نرمال‌شده؛ مرتب‌سازی پایدار (kind='stable') تا تساوی‌ها ترتیب درج را حفظ کنند.",
    en="scores = matrix @ normalised query; use a stable sort so ties keep insertion order.",
)
def test_memory_search(vs: ModuleType, memory) -> None:  # type: ignore[no-untyped-def]
    """InMemoryVectorStore.search returns the most similar chunks, best first"""
    results = memory.search("ساعت کاری کتابخانه", k=2)
    assert results[0].chunk.id == "lib" and len(results) == 2
    assert results[0].score >= results[1].score
    assert memory.search("هر چیزی", k=0) == []
    assert len(memory.search("پیک", k=10)) == 4
    same = vs.InMemoryVectorStore(HashingEmbedding())
    same.add([Chunk(id=str(i), doc_id="d", text="یکسان") for i in range(4)])
    assert [r.chunk.id for r in same.search("یکسان", k=4)] == ["0", "1", "2", "3"]


@hint(
    fa="امتیاز تکه‌های ردشده را ‎-inf کنید و k را به تعداد تکه‌های قبول‌شده محدود کنید.",
    en="Set rejected chunks' scores to -inf and cap k at the number that passed.",
)
def test_memory_filters(memory) -> None:  # type: ignore[no-untyped-def]
    """InMemoryVectorStore.search applies metadata filters before ranking"""
    assert [
        r.chunk.id for r in memory.search("کتابخانه", k=4, filter={"doc_id": "sepidar-rules"})
    ] == ["lib2"]
    assert {
        r.chunk.id for r in memory.search("کتابخانه", k=4, filter=lambda c: c.metadata["page"] >= 2)
    } == {"pay", "lib2"}
    assert memory.search("کتابخانه", k=4, filter={"doc_id": "nope"}) == []


@hint(
    fa="save_chunks و np.save برای ذخیره؛ load_chunks و np.load و add(chunks, vectors) برای بارگذاری.",
    en="save_chunks + np.save to save; load_chunks + np.load + add(chunks, vectors) to load.",
)
def test_memory_save_and_load(vs: ModuleType, memory, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """InMemoryVectorStore.save/load round-trips vectors, chunks and metadata"""
    memory.save(tmp_path / "idx")
    assert {p.name for p in (tmp_path / "idx").iterdir()} == {
        "vectors.npy",
        "chunks.jsonl",
        "store.json",
    }
    loaded = vs.InMemoryVectorStore.load(tmp_path / "idx", HashingEmbedding())
    assert len(loaded) == 4 and loaded.chunks[2].metadata == {"page": 2, "section": "پرداخت"}
    assert [r.chunk.id for r in loaded.search("پرداخت قسطی", k=3)] == [
        r.chunk.id for r in memory.search("پرداخت قسطی", k=3)
    ]
    with pytest.raises(ValueError):
        vs.InMemoryVectorStore.load(tmp_path / "idx", HashingEmbedding(dim=64))


# ----------------------------------------------------------------------- FAISS


@hint(
    fa="FAISS فیلتر ندارد: k×4 نتیجه بگیرید، فیلتر کنید و اگر کم بود تعداد را دو برابر کنید.",
    en="FAISS has no filters: fetch 4k results, filter, and double the fetch if too few survive.",
)
@pytest.mark.parametrize("kind", ["flat", "hnsw", "ivf"])
def test_faiss_search_and_filters(vs: ModuleType, kind: str) -> None:
    """FaissVectorStore finds the same answers as exact search, with filters"""
    pytest.importorskip("faiss", reason=EXTRA)
    rng = np.random.default_rng(1)
    emb = HashingEmbedding(dim=32)
    data = rng.normal(size=(500, 32)).astype(np.float32)
    chunks = [
        Chunk(id=str(i), doc_id=f"d{i % 5}", text="", metadata={"even": i % 2 == 0})
        for i in range(500)
    ]
    exact = vs.InMemoryVectorStore(emb)
    exact.add(chunks, data)
    store = vs.FaissVectorStore(emb, kind, nlist=8, nprobe=8)
    store.add(chunks, data)
    q = data[42] + 0.01
    assert store.search_by_vector(q, 1)[0].chunk.id == "42"
    filtered = store.search_by_vector(q, 5, filter={"doc_id": "d3"})
    assert len(filtered) == 5 and all(r.chunk.doc_id == "d3" for r in filtered)
    rare = store.search_by_vector(q, 3, filter=lambda c: c.id in {"7", "300", "499"})
    assert {r.chunk.id for r in rare} == {"7", "300", "499"}
    assert [r.chunk.id for r in store.search_by_vector(q, 1, filter={"even": False})] == [
        exact.search_by_vector(q, 1, filter={"even": False})[0].chunk.id
    ]


# ------------------------------------------------------------------ the payoff


def test_faiss_matches_exact_retrieval_on_fixtures(vs: ModuleType) -> None:
    """swapping the store does not change retrieval quality on the course corpus"""
    pytest.importorskip("faiss", reason=EXTRA)
    docs, qa = load_fixture_benchmark()
    exact = index_documents(docs, HashingEmbedding())
    hnsw = vs.FaissVectorStore(HashingEmbedding(), "hnsw")
    hnsw.add(exact.chunks)

    def mrr(store: object) -> float:
        return evaluate_retrieval(
            qa, lambda q, k: store.search(normalize(q), k), (3,), level="passage"
        ).mrr  # type: ignore[attr-defined]

    assert mrr(hnsw) == pytest.approx(mrr(exact), abs=0.02)

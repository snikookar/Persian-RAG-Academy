from __future__ import annotations

import pytest

from parsrag import HashingEmbedding, InMemoryVectorStore
from parsrag.retrieval import (
    PERSIAN_STOPWORDS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    PersianTokenizer,
    Retriever,
    light_stem,
    normalize_scores,
    reciprocal_rank_fusion,
    tokenize,
    weighted_score_fusion,
)
from parsrag.types import Chunk, SearchResult


def chunks() -> list[Chunk]:
    texts = ["هزینهٔ پیک در تهران", "ارسال رایگان برای خریدهای بالا", "جریمهٔ دیرکرد کتاب"]
    return [Chunk(id=f"c{i}", doc_id=f"d{i}", text=t) for i, t in enumerate(texts)]


def test_tokenizer_variants() -> None:
    assert tokenize("خریدهای بزرگ‌تر", stem=True) == ["خرید", "بزرگ"]
    assert "است" in PERSIAN_STOPWORDS
    assert PersianTokenizer(stopwords=PERSIAN_STOPWORDS)("این کتاب است") == ["کتاب"]
    assert tokenize("كتاب", normalizer=None) == ["كتاب"]
    assert light_stem("‌ها") == "‌ها"  # nothing left before the ZWNJ


def test_bm25_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        BM25Retriever(b=2.0)
    empty = BM25Retriever()
    assert empty.search("هر چیزی") == [] and empty.avgdl == 0.0


def test_bm25_rare_terms_weigh_more() -> None:
    bm25 = BM25Retriever(chunks())
    assert bm25.idf("پیک") > 0 and bm25.df("پیک") == 1 and bm25.df("نیست") == 0
    assert bm25.search("کرایهٔ پیک", 1)[0].chunk.id == "c0"


def test_retrievers_share_one_protocol() -> None:
    store = InMemoryVectorStore(HashingEmbedding(dim=64))
    store.add(chunks())
    bm25 = BM25Retriever(chunks())
    for r in (store, bm25, DenseRetriever(store), HybridRetriever([bm25, store])):
        assert isinstance(r, Retriever)
        assert r.search("پیک تهران", 2)[0].chunk.id == "c0"
    assert HybridRetriever([bm25]).search("x", 0) == []


def test_hybrid_validation_and_filters() -> None:
    bm25 = BM25Retriever(chunks())
    with pytest.raises(ValueError):
        HybridRetriever([])
    with pytest.raises(ValueError):
        HybridRetriever([bm25], fusion="max")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HybridRetriever([bm25], weights=[1.0, 2.0])
    hybrid = HybridRetriever([bm25, bm25], fusion="weighted")
    assert [r.chunk.id for r in hybrid.search("کتاب پیک", 3, filter={"doc_id": "d2"})] == ["c2"]


def test_fusion_edge_cases() -> None:
    r = [SearchResult(chunk=c, score=s) for c, s in zip(chunks(), [4.0, 2.0, 0.0], strict=True)]
    assert normalize_scores(r, "max") == [1.0, 0.5, 0.0]
    assert normalize_scores(r, "none") == [4.0, 2.0, 0.0]
    assert normalize_scores([]) == []
    assert reciprocal_rank_fusion([]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([r], k=-1)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([r], weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        weighted_score_fusion([r], [1.0, 1.0])
    assert len(weighted_score_fusion([r, r], limit=2)) == 2

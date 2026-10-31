from __future__ import annotations

import importlib.util

import pytest

from parsrag import FakeLLM
from parsrag.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    LLMReranker,
    Reranker,
    RerankingRetriever,
)
from parsrag.retrieval import BM25Retriever
from parsrag.types import Chunk, SearchResult


def results() -> list[SearchResult]:
    texts = ["ساعت کاری", "هزینهٔ پیک در تهران", "پیک"]
    return [
        SearchResult(chunk=Chunk(id=f"c{i}", doc_id="d", text=t), score=0.0)
        for i, t in enumerate(texts)
    ]


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        self.pairs.extend(pairs)
        return [float(len(text)) for _, text in pairs]


def test_cross_encoder_with_injected_model() -> None:
    model = FakeCrossEncoder()
    ce = CrossEncoderReranker(model=model)
    out = ce.rerank("پیک", results(), top_k=2)
    assert [r.chunk.id for r in out] == ["c1", "c0"]
    assert model.pairs[0] == ("پیک", "ساعت کاری")
    assert ce.rerank("پیک", []) == []


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="sentence-transformers is installed",
)
def test_cross_encoder_needs_the_st_extra() -> None:
    with pytest.raises(ImportError, match=r"parsrag\[st\]"):
        CrossEncoderReranker()


def test_rerankers_share_the_protocol() -> None:
    for r in (LexicalReranker(), LLMReranker(FakeLLM(["5"])), CrossEncoderReranker(model=object())):
        assert isinstance(r, Reranker)


def test_llm_reranker_default_score_and_truncation() -> None:
    llm = FakeLLM(["no idea"])
    out = LLMReranker(llm, default_score=1.5, max_chars=3).rerank("q", results())
    assert {r.score for r in out} == {1.5}
    assert "ساعت\n" not in llm.calls[0][-1].content and "ساع" in llm.calls[0][-1].content


def test_reranking_retriever_over_bm25_with_idf() -> None:
    chunks = [r.chunk for r in results()]
    bm25 = BM25Retriever(chunks)
    pipe = RerankingRetriever(bm25, LexicalReranker(idf=bm25.idf), fetch_k=5)
    assert pipe.search("هزینهٔ پیک تهران", 1)[0].chunk.id == "c1"
    with pytest.raises(ValueError):
        RerankingRetriever(bm25, LexicalReranker(), fetch_k=0)

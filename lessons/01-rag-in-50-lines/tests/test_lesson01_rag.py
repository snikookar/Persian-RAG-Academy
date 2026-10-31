"""Lesson 01 — RAG in 50 lines. Run with: python -m grader run 01"""

from __future__ import annotations

from types import ModuleType

import pytest

from grader.paths import CORPUS, QA_FILE
from parsrag import FakeLLM, HashingEmbedding, Message
from parsrag.eval import hit_rate, load_qa
from parsrag.types import Chunk, SearchResult

hint = pytest.mark.hint


def _result(doc_id: str, text: str, score: float = 1.0) -> SearchResult:
    return SearchResult(chunk=Chunk(id=f"{doc_id}#0", doc_id=doc_id, text=text), score=score)


# ------------------------------------------------------------------- chunking


@hint(
    fa="گام (step) برابر size - overlap است: text[i:i+size] برای i در range(0, ..., step).",
    en="The step is size - overlap: text[i:i+size] for i in range(0, ..., step).",
)
@pytest.mark.parametrize(
    ("text", "size", "overlap", "expected"),
    [
        ("abcdefghij", 4, 0, ["abcd", "efgh", "ij"]),
        ("abcdefghij", 4, 1, ["abcd", "defg", "ghij"]),
        ("abcdefg", 4, 1, ["abcd", "defg"]),
        ("کتابخانه", 3, 1, ["کتا", "ابخ", "خان", "نه"]),
        ("کوتاه", 100, 10, ["کوتاه"]),
        ("", 10, 2, []),
        ("   ", 10, 2, []),
    ],
)
def test_chunk_text(
    rag: ModuleType, text: str, size: int, overlap: int, expected: list[str]
) -> None:
    """chunk_text makes fixed-size overlapping windows"""
    assert rag.chunk_text(text, size, overlap) == expected


@hint(
    fa="اگر size <= 0 یا overlap منفی یا overlap >= size بود، ValueError بدهید.",
    en="Raise ValueError when size <= 0, overlap < 0 or overlap >= size.",
)
@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_chunk_text_rejects_bad_arguments(rag: ModuleType, size: int, overlap: int) -> None:
    """chunk_text rejects impossible size/overlap values"""
    with pytest.raises(ValueError):
        rag.chunk_text("متن", size, overlap)


def test_chunks_cover_the_whole_text(rag: ModuleType) -> None:
    """chunk_text never loses text"""
    text = (CORPUS / "kahroba-faq.md").read_text(encoding="utf-8").strip()
    chunks = rag.chunk_text(text, 120, 20)
    rebuilt = chunks[0] + "".join(c[20:] for c in chunks[1:])
    assert rebuilt == text


# -------------------------------------------------------------------- loading


def test_load_documents(rag: ModuleType) -> None:
    """load_documents reads the fixture corpus (id = file name, title = first heading)"""
    docs = rag.load_documents(CORPUS)
    assert len(docs) == 25
    by_id = {d.id: d for d in docs}
    assert by_id["kahroba-club"].title == "باشگاه مشتریان کهربا"
    assert not by_id["kahroba-club"].text.startswith("#")


# --------------------------------------------------------------------- prompt


@hint(
    fa="دو پیام بسازید: system با SYSTEM_PROMPT و user شامل متن‌های شماره‌دار و «پرسش: ...» در انتها.",
    en="Return two messages: system (SYSTEM_PROMPT) and user (numbered context + 'پرسش: ...' last).",
)
def test_build_prompt(rag: ModuleType) -> None:
    """build_prompt puts numbered context and the question into chat messages"""
    results = [_result("a", "متن اول"), _result("b", "متن دوم")]
    messages = rag.build_prompt("سؤال من چیست؟", results)
    messages = [m if isinstance(m, Message) else Message.model_validate(m) for m in messages]
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == rag.SYSTEM_PROMPT
    user = messages[1].content
    assert "[1] (a) متن اول" in user
    assert "[2] (b) متن دوم" in user
    assert user.rstrip().endswith("پرسش: سؤال من چیست؟")
    assert user.index("متن اول") < user.index("متن دوم") < user.index("پرسش:")


# ------------------------------------------------------------------- pipeline


@pytest.fixture
def pipeline(rag: ModuleType):  # type: ignore[no-untyped-def]
    p = rag.NaiveRAG(HashingEmbedding(), FakeLLM(mode="extractive"), chunk_size=400, k=3)
    p.index(rag.load_documents(CORPUS))
    return p


@hint(
    fa="هر سند را با chunk_text تکه کنید و Chunk(id=f'{doc.id}#{i}', doc_id=doc.id, ...) بسازید.",
    en="Chunk each document with chunk_text and build Chunk(id=f'{doc.id}#{i}', doc_id=doc.id, ...).",
)
def test_index_creates_chunks(rag: ModuleType) -> None:
    """NaiveRAG.index chunks every document into the store"""
    p = rag.NaiveRAG(HashingEmbedding(), FakeLLM(), chunk_size=400, chunk_overlap=50)
    docs = rag.load_documents(CORPUS)
    n = p.index(docs)
    expected = sum(len(rag.chunk_text(d.text, 400, 50)) for d in docs)
    assert n == expected == len(p.store)
    assert {c.doc_id for c in p.store.chunks} == {d.id for d in docs}
    assert all(c.id.startswith(f"{c.doc_id}#") for c in p.store.chunks)


@hint(
    fa="retrieve فقط self.store.search(question, k) را صدا می‌زند.",
    en="retrieve() simply calls self.store.search(question, k).",
)
def test_retrieve(pipeline) -> None:  # type: ignore[no-untyped-def]
    """NaiveRAG.retrieve returns the k most similar chunks, best first"""
    results = pipeline.retrieve("هزینهٔ پیک در تهران چقدر است؟")
    assert len(results) == 3
    assert results[0].chunk.doc_id == "kahroba-shipping"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert len(pipeline.retrieve("پیک", k=1)) == 1


@hint(
    fa="answer: بازیابی → build_prompt → llm.generate → Answer(text=..., sources=...).",
    en="answer(): retrieve → build_prompt → llm.generate → Answer(text=..., sources=...).",
)
def test_answer(pipeline) -> None:  # type: ignore[no-untyped-def]
    """NaiveRAG.answer generates from retrieved context"""
    answer = pipeline.answer("هزینهٔ پیک در تهران چقدر است؟")
    assert "45" in answer.text
    assert answer.sources[0].chunk.doc_id == "kahroba-shipping"
    sent = pipeline.llm.calls[-1]
    assert sent[-1].content.rstrip().endswith("پرسش: هزینهٔ پیک در تهران چقدر است؟")


def test_end_to_end_hit_rate(pipeline) -> None:  # type: ignore[no-untyped-def]
    """the naive pipeline finds the right document for most fixture questions"""
    qa = load_qa(QA_FILE)
    assert hit_rate(qa, pipeline.retrieve, k=3) >= 0.8

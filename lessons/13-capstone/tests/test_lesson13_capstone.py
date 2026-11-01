"""Lesson 13 — capstone: RAG over your own documents. Run with: python -m grader run 13"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from grader.paths import FIXTURES, LESSONS, QA_FILE, REPO_ROOT
from parsrag.eval.benchmark import load_fixture_benchmark
from parsrag.eval.dataset import QAItem, load_qa
from parsrag.types import Document

hint = pytest.mark.hint

PIK = "هزینهٔ پیک در تهران چقدر است؟"
SHOP = Document(
    id="shop",
    title="راهنمای فروشگاه",
    text="هزينهٔ پيک در تهران ۲۰۰ هزار تومان است.\n\nکالا را تا ۷ روز پس از تحویل می‌توانید مرجوع کنید.",
)
LIB = Document(id="lib", text="کتابخانه از ساعت ۸ صبح تا ۸ شب باز است.")
EXAMPLE = LESSONS / "13-capstone" / "my_questions.example.jsonl"


@pytest.fixture(scope="module")
def course(rg: ModuleType) -> Any:
    return rg.PersianRAG.course_default()


# ---------------------------------------------------------------- indexing


@hint(
    fa="chunk_document با heading_context؛ اسناد و قطعه‌ها را نگه دارید؛ store.add و bm25.add؛ index_version += 1.",
    en="chunk_document with heading_context; keep documents and chunks; store.add and bm25.add; index_version += 1.",
)
def test_index_documents(rg: ModuleType) -> None:
    """documents are chunked (with their title) into the vector store and BM25"""
    rag = rg.PersianRAG([SHOP, LIB])
    assert set(rag.documents) == {"shop", "lib"} and rag.index_version == 1
    assert len(rag) == len(rag.store) == len(rag.bm25) >= 2
    first = next(c for c in rag.chunks if c.doc_id == "shop")
    assert first.text.startswith("راهنمای فروشگاه") and "هزینهٔ پیک" in first.text  # normalised
    rag.add_documents([Document(id="x", text="متن تازه")])
    assert rag.index_version == 2 and "x" in rag.documents


@hint(
    fa="شناسهٔ تکراری (در نمایه یا در همین دسته) → ValueError با «documents already indexed».",
    en="A repeated id (already indexed or within the batch) → ValueError 'documents already indexed'.",
)
def test_duplicates_are_rejected(rg: ModuleType) -> None:
    """re-adding a document is an error, not a silent duplicate"""
    rag = rg.PersianRAG([SHOP])
    with pytest.raises(ValueError, match="already indexed"):
        rag.add_documents([SHOP])
    with pytest.raises(ValueError, match="already indexed"):
        rg.PersianRAG([LIB, LIB])
    assert len(rag.documents) == 1


@hint(
    fa="add_documents مسیر پوشه را با loaders درس ۰۳ بار می‌کند و _index شما را صدا می‌زند.",
    en="add_documents loads folders with lesson 03's loaders and calls your _index.",
)
def test_index_a_folder(rg: ModuleType, tmp_path: Path) -> None:
    """a folder of your own files becomes a searchable index"""
    (tmp_path / "cafe.md").write_text("# کافه\nکافه جمعه‌ها تعطیل است.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "menu.txt").write_text("قیمت چای ۲۰ هزار تومان است.", encoding="utf-8")
    rag = rg.PersianRAG.from_directory(tmp_path)
    assert set(rag.documents) == {"cafe", "menu"}
    assert rag.search("قیمت چای", k=1)[0].chunk.doc_id == "menu"


# --------------------------------------------------------------- retrieval


@hint(
    fa="hybrid = HybridRetriever([bm25, DenseRetriever(store, preprocess=...)])؛ retriever = RerankingRetriever(hybrid, reranker, fetch_k).",
    en="hybrid = HybridRetriever([bm25, DenseRetriever(store, preprocess=...)]); retriever = RerankingRetriever(hybrid, reranker, fetch_k).",
)
def test_retrieval_chain(rg: ModuleType, course: Any) -> None:
    """RRF(BM25, dense) → lexical rerank of 20, as chosen in lessons 07–08"""
    from parsrag.rerank import RerankingRetriever
    from parsrag.retrieval import HybridRetriever

    assert isinstance(course.hybrid, HybridRetriever) and len(course.hybrid.retrievers) == 2
    r = course.retriever
    assert isinstance(r, RerankingRetriever) and r.fetch_k == 20
    assert r.search(PIK, 1)[0].chunk.doc_id == "kahroba-shipping"
    plain = rg.PersianRAG([SHOP, LIB], reranker=None)
    assert isinstance(plain.retriever, HybridRetriever)


# ----------------------------------------------------------------- answers


@hint(
    fa="اول _check_index؛ بعد generator، retriever.search و answer_from؛ زمان هر مرحله با perf_counter.",
    en="First _check_index; then generator, retriever.search and answer_from; time each stage with perf_counter.",
)
def test_ask_with_citations_and_timings(rg: ModuleType, course: Any) -> None:
    """ask returns a cited answer and per-stage timings"""
    timings = rg.Timings()
    answer = course.ask(PIK, timings=timings)
    assert not answer.abstained and "200" in answer.text
    assert answer.citations[0].doc_id == "kahroba-shipping"
    assert timings.llm_calls == 1 and timings.total_ms >= timings.retrieve_ms > 0
    with pytest.raises(rg.EmptyIndexError):
        rg.PersianRAG().ask(PIK)


@hint(
    fa="متن اصلی سند را با start/end برش بزنید، نه متن نرمال‌شدهٔ قطعه را.",
    en="Slice the original document text with start/end, not the normalised chunk text.",
)
def test_passage_is_the_original_text(rg: ModuleType) -> None:
    """a citation links to the exact, original source text"""
    rag = rg.PersianRAG([SHOP, LIB])
    answer = rag.ask("هزینهٔ پیک در تهران چقدر است؟")
    passage = rag.passage(answer.citations[0])
    assert passage.startswith("هزينهٔ پيک")  # Arabic letters as written in the source
    from parsrag.generate import Citation

    assert rag.passage(Citation(number=1, chunk_id="?", doc_id="nope")) == ""
    no_offsets = Citation(number=1, chunk_id="lib#0", doc_id="lib")
    assert "کتابخانه" in rag.passage(no_offsets)


@hint(
    fa="retriever_confidence(self.bm25) برای هر پرسش، بعد calibrate_threshold؛ min_score را تنظیم کنید.",
    en="retriever_confidence(self.bm25) per question, then calibrate_threshold; set min_score.",
)
def test_calibrate_and_abstain(rg: ModuleType) -> None:
    """the «نمی‌دانم» threshold is calibrated on a question file"""
    rag = rg.PersianRAG.course_default(calibrate=False)
    assert rag.min_score is None
    threshold = rag.calibrate(str(QA_FILE))
    assert rag.min_score == threshold == pytest.approx(8.423, abs=1e-3)
    answer = rag.ask("قیمت بلیت هواپیما به استانبول چقدر است؟")
    assert answer.abstained and answer.abstain_reason == "low_score"


@hint(
    fa="اگر عددها با baseline یکی نیستند، ترتیب قطعه‌ها، heading_context یا آستانه را با مرجع مقایسه کنید.",
    en="If the numbers differ from the baseline, compare chunk order, heading_context or the threshold with the reference.",
)
def test_your_facade_reproduces_the_eval_gate(course: Any) -> None:
    """your PersianRAG gives exactly the committed lesson 10 baseline"""
    baseline = json.loads((REPO_ROOT / "docs/eval/baseline.json").read_text(encoding="utf-8"))
    report = course.evaluate(str(QA_FILE), n_resamples=10)
    for key in ("retrieval_r@1", "retrieval_mrr", "answer_f1", "abstention_f1"):
        assert report.metrics[key] == pytest.approx(baseline["metrics"]["original"][key], abs=1e-3)


# ------------------------------------------------------------ your own set


@hint(
    fa="ترتیب بررسی‌ها برای هر پرسش: تکراری، خالی، بی‌پاسخ با شاهد، doc_id ناشناخته، بدون شاهد، شاهدِ پیدانشده (با match_key).",
    en="Per item: duplicate, empty, unanswerable with evidence, unknown doc_id, no evidence, evidence not found (match_key).",
)
def test_validate_qa(rg: ModuleType) -> None:
    """mistakes in a hand-written question file are reported"""
    items = [
        QAItem(
            id="1", question="پیک؟", answer="۲۰۰", doc_id="shop", evidence="هزینهٔ پیک در تهران 200"
        ),
        QAItem(id="2", question="؟", answer="-", doc_id="shop"),
        QAItem(id="2", question=" ", answer="-", doc_id=None, evidence="x"),
        QAItem(id="3", question="پیک؟", answer="-", doc_id="shop", evidence="۵۰ هزار تومان"),
        QAItem(id="4", question="پیک؟", answer="-", doc_id="other", evidence="x"),
    ]
    assert rg.validate_qa(items, [SHOP]) == [
        "2: answerable but has no evidence span",
        "2: duplicate id",
        "2: empty question",
        "2: unanswerable (no doc_id) but has evidence",
        "3: evidence not found in 'shop': '۵۰ هزار تومان'",
        "4: unknown doc_id 'other'",
    ]


@hint(
    fa="فایل نمونه باید بدون مشکل باشد؛ اگر خطا گرفتید، validate_qa شما احتمالاً match_key را روی هر دو طرف اجرا نمی‌کند.",
    en="The example file is valid; if this fails, your validate_qa probably does not apply match_key to both sides.",
)
def test_example_question_file(rg: ModuleType) -> None:
    """the lesson's example question file is valid and covers all three kinds of questions"""
    docs, _ = load_fixture_benchmark(FIXTURES)
    items = load_qa(EXAMPLE)
    assert rg.validate_qa(items, docs) == []
    assert len(items) >= 12 and sum(not q.answerable for q in items) >= 3


# ------------------------------------------------------------- end to end


@hint(
    fa="save/load و API از _index و retriever شما استفاده می‌کنند؛ اگر این رد شد، آن‌ها را بررسی کنید.",
    en="save/load and the API use your _index and retriever; if this fails, check those.",
)
def test_save_load_and_serve(rg: ModuleType, course: Any, tmp_path: Path) -> None:
    """your index can be saved, loaded and served by the API"""
    course.save(tmp_path / "idx")
    loaded = rg.PersianRAG.load(tmp_path / "idx", llm=course.llm)
    assert loaded.ask(PIK) == course.ask(PIK)
    pytest.importorskip("fastapi", reason="API check needs: pip install -e '.[api]'")
    from fastapi.testclient import TestClient

    from parsrag.api import Settings, create_app

    client = TestClient(create_app(loaded, Settings(rate_limit=0)))
    data = client.post("/ask", json={"question": PIK}).json()
    assert "200" in data["text"] and "هزینهٔ پیک در تهران" in data["citations"][0]["passage"]
    assert 'dir="rtl"' in client.get("/").text

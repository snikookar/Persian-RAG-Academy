"""The PersianRAG facade and its configuration (lesson 13)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from parsrag import FakeLLM, PersianRAG, TfidfEmbedding
from parsrag.chunking import RecursiveChunker
from parsrag.config import RAGConfig, load_config, make_embedding, make_llm
from parsrag.eval.benchmark import FIXTURES, load_fixture_benchmark
from parsrag.llm import OpenAICompatibleLLM
from parsrag.rag import EmptyIndexError, Timings, answer_payload
from parsrag.types import Document

SHOP = Document(
    id="shop",
    title="راهنمای فروشگاه",
    text="# ارسال\nهزینهٔ پیک در تهران ۴۵ هزار تومان است.\n\n# مرجوعی\n"
    "کالا را تا ۷ روز پس از تحویل می‌توانید مرجوع کنید.",
)
LIB = Document(id="lib", text="کتابخانه از ساعت ۸ صبح تا ۸ شب باز است.")


@pytest.fixture(scope="module")
def course() -> PersianRAG:
    return PersianRAG.course_default()


def test_empty_index_raises() -> None:
    rag = PersianRAG()
    assert len(rag) == 0
    with pytest.raises(EmptyIndexError):
        rag.ask("هزینهٔ پیک؟")
    with pytest.raises(EmptyIndexError):
        rag.search("پیک")


def test_add_documents_ask_and_cite() -> None:
    rag = PersianRAG([SHOP, LIB])
    assert set(rag.documents) == {"shop", "lib"} and rag.index_version == 1
    timings = Timings()
    ans = rag.ask("هزینهٔ پیک در تهران چقدر است؟", timings=timings)
    assert not ans.abstained
    assert "۴۵" in ans.text or "45" in ans.text
    assert ans.citations and ans.citations[0].doc_id == "shop"
    assert "هزینهٔ پیک در تهران" in rag.passage(ans.citations[0])
    assert timings.llm_calls == 1 and timings.total_ms >= timings.retrieve_ms >= 0


def test_duplicate_documents_are_rejected() -> None:
    rag = PersianRAG([SHOP])
    with pytest.raises(ValueError, match="already indexed"):
        rag.add_documents([SHOP])
    with pytest.raises(ValueError, match="already indexed"):
        PersianRAG([LIB, LIB])


def test_add_texts_and_paths(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# الف\nساعت کاری کتابخانه ۹ صبح است.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("قیمت بلیت ۲۰ هزار تومان است.", encoding="utf-8")
    rag = PersianRAG([str(tmp_path)])
    assert set(rag.documents) == {"a", "b"}
    rag.add_texts(["متن آزاد درباره‌ی باغ"], ids=["note"])
    assert "note" in rag.documents and rag.index_version == 2
    with pytest.raises(FileNotFoundError):
        rag.add_documents([tmp_path / "missing"])


def test_course_default_matches_lesson_10_pipeline(course: PersianRAG) -> None:
    from parsrag.eval.benchmark import load_fixture_benchmark

    _, qa = load_fixture_benchmark()
    report = course.evaluate(qa[:20], n_resamples=50)
    assert report.n_questions == 20
    assert course.min_score is not None and course.min_score > 0


def test_calibrated_abstention(course: PersianRAG) -> None:
    ans = course.ask("قیمت بلیت هواپیما به استانبول چقدر است؟")
    assert ans.abstained and ans.abstain_reason in ("low_score", "model")


def test_stream_yields_sources_tokens_done(course: PersianRAG) -> None:
    events = list(course.stream("هزینهٔ پیک در تهران چقدر است؟"))
    kinds = [e.type for e in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "token" in kinds
    text = "".join(e.data for e in events if e.type == "token")
    final = events[-1].data
    assert final.answer == text.strip()
    assert final == course.ask("هزینهٔ پیک در تهران چقدر است؟")


def test_stream_early_abstention(course: PersianRAG) -> None:
    events = list(course.stream("قیمت بلیت هواپیما به استانبول چقدر است؟"))
    assert [e.type for e in events] == ["sources", "done"]


def test_reranker_can_be_turned_off() -> None:
    rag = PersianRAG([SHOP, LIB], reranker=None)
    assert type(rag.retriever).__name__ == "HybridRetriever"
    assert rag.search("پیک", k=1)[0].chunk.doc_id == "shop"


def test_tfidf_embedding_is_refit_on_add() -> None:
    rag = PersianRAG([SHOP], embedding=TfidfEmbedding())
    rag.add_documents([LIB])
    assert len(rag.store) == len(rag) and rag.embedding.fitted  # type: ignore[attr-defined]
    assert rag.search("کتابخانه ساعت", k=1)[0].chunk.doc_id == "lib"


def test_save_and_load_round_trip(tmp_path: Path, course: PersianRAG) -> None:
    course.save(tmp_path / "idx")
    assert (tmp_path / "idx" / "store" / "vectors.npy").is_file()
    loaded = PersianRAG.load(tmp_path / "idx", llm=course.llm)
    assert len(loaded) == len(course) and loaded.min_score == course.min_score
    q = "جریمهٔ دیرکرد کتاب چقدر است؟"
    assert loaded.ask(q) == course.ask(q)
    rebuilt = PersianRAG.load(tmp_path / "idx", chunker=RecursiveChunker(200, 20))
    assert len(rebuilt) > len(course)  # different chunker → chunks rebuilt, not reused


def test_answer_payload_has_passages() -> None:
    rag = PersianRAG([SHOP])
    payload = answer_payload(rag, rag.ask("کالا را تا چند روز می‌توان مرجوع کرد؟"))
    assert payload.citations and "۷ روز" in payload.citations[0]["passage"]
    json.dumps(payload.model_dump(), ensure_ascii=False)


# ------------------------------------------------------------------ config


def test_load_config_defaults_and_mapping() -> None:
    cfg = load_config()
    assert cfg.retrieval.k == 4 and cfg.llm.provider == "fake"
    cfg = load_config({"retrieval": {"k": 2, "rerank": "none"}, "llm": {"provider": "fake"}})
    assert cfg.retrieval.k == 2
    with pytest.raises(ValueError):
        load_config({"retrieval": {"kk": 2}})


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib needs Python 3.11+")
def test_from_config_toml(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "shop.md").write_text(SHOP.text, encoding="utf-8")
    cfg = tmp_path / "parsrag.toml"
    cfg.write_text(
        '[documents]\npaths = ["docs"]\n\n[chunking]\nsize = 200\noverlap = 20\n\n'
        '[retrieval]\nk = 2\nrerank = "none"\nmin_score = 0.5\n',
        encoding="utf-8",
    )
    rag = PersianRAG.from_config(cfg)
    assert set(rag.documents) == {"shop"} and rag.k == 2 and rag.reranker is None
    assert rag.min_score == 0.5 and rag.config.resolve("docs") == docs
    assert not rag.ask("هزینهٔ پیک در تهران چقدر است؟").abstained


def test_from_config_json_and_overrides(tmp_path: Path) -> None:
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"retrieval": {"k": 3}}), encoding="utf-8")
    llm = FakeLLM(["پاسخ [1]"])
    rag = PersianRAG.from_config(cfg, llm=llm, k=1)
    assert rag.k == 1 and rag.llm is llm


def test_make_providers_lazily() -> None:
    cfg = RAGConfig.model_validate(
        {"llm": {"provider": "ollama", "model": "qwen2.5:7b"}, "embedding": {"provider": "tfidf"}}
    )
    llm = make_llm(cfg.llm)
    assert isinstance(llm, OpenAICompatibleLLM) and llm.base_url.endswith(":11434/v1")
    assert isinstance(make_embedding(cfg.embedding), TfidfEmbedding)
    with pytest.raises(ValueError):
        make_llm(RAGConfig.model_validate({"llm": {"provider": "openai"}}).llm)
    with pytest.raises(ValueError):
        make_embedding(RAGConfig.model_validate({"embedding": {"provider": "nope"}}).embedding)


def test_fixture_benchmark_unchanged() -> None:
    docs, qa = load_fixture_benchmark(FIXTURES)
    assert len(docs) == 25 and len(qa) == 66


def test_validate_qa_finds_problems() -> None:
    from parsrag.eval import QAItem, validate_qa

    docs = [SHOP]
    items = [
        QAItem(id="1", question="هزینهٔ پیک؟", answer="۴۵", doc_id="shop", evidence="۴۵ هزار تومان"),
        QAItem(id="2", question="؟", answer="-", doc_id="shop"),
        QAItem(id="2", question=" ", answer="-", doc_id=None, evidence="x"),
        QAItem(id="3", question="پیک؟", answer="-", doc_id="shop", evidence="۵۰ هزار تومان"),
    ]
    problems = validate_qa(items, docs)
    assert problems == [
        "2: answerable but has no evidence span",
        "2: duplicate id",
        "2: empty question",
        "2: unanswerable (no doc_id) but has evidence",
        "3: evidence not found in 'shop': '۵۰ هزار تومان'",
    ]

from __future__ import annotations

import json
from pathlib import Path

from grader.paths import CORPUS, QA_FILE
from parsrag import Chunk, FakeLLM, HashingEmbedding, SearchResult
from parsrag.doctor import extras_status, main, python_version_ok, smoke_test
from parsrag.eval import QAItem, hit_rate, load_qa
from parsrag.naive_rag import NaiveRAG, load_documents
from parsrag.normalize import normalize


def test_load_qa_fixture() -> None:
    items = load_qa(QA_FILE)
    assert len(items) >= 25
    assert len({i.id for i in items}) == len(items)
    assert sum(not i.answerable for i in items) >= 3
    doc_ids = {p.stem for p in CORPUS.glob("*.md")}
    assert {i.doc_id for i in items if i.answerable} <= doc_ids


def test_load_qa_skips_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "qa.jsonl"
    row = {"id": "x", "question": "q", "answer": "a", "doc_id": None}
    f.write_text("\n" + json.dumps(row) + "\n\n", encoding="utf-8")
    assert load_qa(f) == [QAItem(id="x", question="q", answer="a")]


def test_hit_rate() -> None:
    items = [
        QAItem(id="1", question="a", answer="", doc_id="A"),
        QAItem(id="2", question="b", answer="", doc_id="B"),
        QAItem(id="3", question="c", answer="", doc_id=None),  # ignored
    ]

    def retrieve(q: str, k: int) -> list[SearchResult]:
        return [SearchResult(chunk=Chunk(id="c", doc_id="A", text=""), score=1.0)][:k]

    assert hit_rate(items, retrieve, k=1) == 0.5
    assert hit_rate(items[2:], retrieve) == 0.0


def test_fixture_corpus_is_deliberately_messy() -> None:
    text = "".join(p.read_text(encoding="utf-8") for p in CORPUS.glob("*.md"))
    for bad in ("ي", "ك", "٣", "ـ", "َ"):
        assert bad in text


def test_normalization_gain_on_fixtures() -> None:
    """The numbers quoted in lessons/02-normalization/README.fa.md."""
    docs, qa = load_documents(CORPUS), load_qa(QA_FILE)
    scores = {}
    for name, pre in (("raw", None), ("normalized", normalize)):
        rag = NaiveRAG(HashingEmbedding(), FakeLLM(), chunk_size=300, preprocess=pre)
        rag.index(docs)
        scores[name] = (hit_rate(qa, rag.retrieve, 1), hit_rate(qa, rag.retrieve, 3))
    assert scores["raw"] == (19 / 24, 22 / 24)
    assert scores["normalized"] == (23 / 24, 24 / 24)


def test_doctor(capsys) -> None:  # type: ignore[no-untyped-def]
    assert python_version_ok()
    assert set(extras_status()) >= {"openai", "st", "loaders"}
    assert "ساعت شش" in smoke_test()
    assert main() == 0
    assert "Offline smoke test" in capsys.readouterr().out

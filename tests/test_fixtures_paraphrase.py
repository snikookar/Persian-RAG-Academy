"""The paraphrase question set (Day 3) stays consistent with the original set."""

from __future__ import annotations

from grader.paths import CORPUS, QA_FILE, QA_PARAPHRASE_FILE
from parsrag.eval import load_qa
from parsrag.eval.retrieval import match_key


def test_paraphrase_set_links_back_to_originals() -> None:
    originals = {q.id: q for q in load_qa(QA_FILE)}
    items = load_qa(QA_PARAPHRASE_FILE)
    assert len(items) >= 40
    assert len({i.id for i in items}) == len(items)
    assert sum(not i.answerable for i in items) >= 3
    for item in items:
        source = originals[item.source_id or ""]
        assert item.doc_id == source.doc_id
        assert item.evidence == source.evidence
        assert item.answer == source.answer
        assert item.question != source.question


def test_paraphrase_evidence_is_in_its_document() -> None:
    docs = {p.stem: p.read_text(encoding="utf-8") for p in CORPUS.glob("*.md")}
    for item in load_qa(QA_PARAPHRASE_FILE):
        if item.answerable:
            assert match_key(item.evidence or "") in match_key(docs[item.doc_id or ""]), item.id

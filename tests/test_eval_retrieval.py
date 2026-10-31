from __future__ import annotations

import math

import pytest

from parsrag.eval import (
    QAItem,
    contains_evidence,
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from parsrag.eval.retrieval import dedupe, match_key
from parsrag.types import Chunk, SearchResult


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"b"}, 1) == 0.0
    assert recall_at_k(["a", "b", "c"], {"b"}, 2) == 1.0
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 3) == 0.5
    assert recall_at_k(["a"], set(), 3) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert reciprocal_rank([], {"c"}) == 0.0


def test_ndcg() -> None:
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(
        (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )
    assert ndcg_at_k(["x"], {"a"}, 0) == 0.0


def test_dedupe_and_match_key() -> None:
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert match_key("كتاب ها ۱۲") == match_key("کتاب‌ها 12")


def _r(doc: str, text: str, i: int = 0) -> SearchResult:
    return SearchResult(chunk=Chunk(id=f"{doc}#{i}", doc_id=doc, text=text), score=1.0)


def test_contains_evidence_uses_window_and_doc() -> None:
    item = QAItem(id="q", question="?", answer="", doc_id="A", evidence="هزینهٔ پیک 45 هزار")
    assert contains_evidence(_r("A", "هزينهٔ پيک ۴۵ هزار تومان").chunk, item)
    assert not contains_evidence(_r("B", "هزینهٔ پیک 45 هزار").chunk, item)
    windowed = Chunk(id="A#1", doc_id="A", text="پیک", metadata={"window": "هزینهٔ پیک 45 هزار"})
    assert contains_evidence(windowed, item)


def test_evaluate_retrieval_doc_and_passage() -> None:
    items = [
        QAItem(id="1", question="x", answer="", doc_id="A", evidence="جواب"),
        QAItem(id="2", question="y", answer="", doc_id="B", evidence="پاسخ"),
        QAItem(id="3", question="z", answer="", doc_id=None),
    ]
    results = {
        "x": [_r("A", "بی‌ربط", 0), _r("A", "این جواب است", 1)],
        "y": [_r("C", "پاسخ"), _r("D", "هیچ")],
    }
    report = evaluate_retrieval(items, lambda q, k: results[q][:k], ks=(1, 2))
    assert report.n == 2
    assert report.recall == {1: 0.5, 2: 0.5}
    assert report.mrr == 0.5
    passage = evaluate_retrieval(items, lambda q, k: results[q][:k], ks=(1, 2), level="passage")
    assert passage.recall == {1: 0.0, 2: 0.5}
    assert passage.mrr == pytest.approx(0.25)
    assert [q.rank for q in passage.per_question] == [2, None]
    assert set(passage.row()) == {"R@1", "R@2", "MRR", "nDCG@1", "nDCG@2"}


def test_evaluate_retrieval_validates() -> None:
    with pytest.raises(ValueError):
        evaluate_retrieval([], lambda q, k: [], ks=(0,))
    items = [QAItem(id="1", question="x", answer="", doc_id="A")]
    with pytest.raises(ValueError, match="evidence"):
        evaluate_retrieval(items, lambda q, k: [], level="passage")

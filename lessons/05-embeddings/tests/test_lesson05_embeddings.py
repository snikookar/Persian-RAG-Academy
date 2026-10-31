"""Lesson 05 — embeddings and a retrieval benchmark. Run with: python -m grader run 05"""

from __future__ import annotations

import math
from types import ModuleType

import numpy as np
import pytest

from parsrag import HashingEmbedding
from parsrag.embeddings import Embedding
from parsrag.eval import QAItem
from parsrag.eval.benchmark import index_documents, load_fixture_benchmark
from parsrag.normalize import normalize
from parsrag.types import Chunk, SearchResult

hint = pytest.mark.hint


# --------------------------------------------------------------------- metrics


@hint(
    fa="ترتیب را نگه دارید و فقط اولین تکرار هر شناسه را بردارید.",
    en="Keep order; keep only the first occurrence.",
)
def test_dedupe(metrics: ModuleType) -> None:
    """dedupe keeps the first occurrence of every id"""
    assert metrics.dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert metrics.dedupe([]) == []


@hint(
    fa="Recall@k = |مرتبط‌ها در k تای اول| / |همهٔ مرتبط‌ها|.",
    en="Recall@k = |relevant in top k| / |relevant|.",
)
@pytest.mark.parametrize(
    ("ranked", "relevant", "k", "expected"),
    [
        (["a", "b", "c"], {"b"}, 1, 0.0),
        (["a", "b", "c"], {"b"}, 2, 1.0),
        (["a", "b", "c"], {"b", "z"}, 3, 0.5),
        (["a", "b"], set(), 2, 0.0),
        (["a", "b"], {"a"}, 0, 0.0),
    ],
)
def test_recall_at_k(
    metrics: ModuleType, ranked: list[str], relevant: set[str], k: int, expected: float
) -> None:
    """recall_at_k counts relevant ids in the top k"""
    assert metrics.recall_at_k(ranked, relevant, k) == pytest.approx(expected)


@hint(
    fa="۱ تقسیم بر رتبهٔ اولین نتیجهٔ مرتبط (رتبه از ۱ شروع می‌شود).",
    en="1 / rank of the first relevant id (ranks start at 1).",
)
@pytest.mark.parametrize(
    ("ranked", "k", "expected"),
    [
        (["c", "a"], None, 1.0),
        (["a", "b", "c"], None, 1 / 3),
        (["a", "b", "c"], 2, 0.0),
        ([], None, 0.0),
    ],
)
def test_reciprocal_rank(
    metrics: ModuleType, ranked: list[str], k: int | None, expected: float
) -> None:
    """reciprocal_rank is 1/rank of the first relevant id"""
    assert metrics.reciprocal_rank(ranked, {"c"}, k) == pytest.approx(expected)


@hint(
    fa="DCG = Σ 1/log2(i+2) برای جایگاه‌های مرتبط (i از ۰)؛ تقسیم بر بهترین DCG ممکن.",
    en="DCG = Σ 1/log2(i+2) over relevant positions (i from 0); divide by the ideal DCG.",
)
def test_ndcg_at_k(metrics: ModuleType) -> None:
    """ndcg_at_k rewards relevant ids near the top"""
    assert metrics.ndcg_at_k(["a", "b"], {"a"}, 2) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))
    both = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(both)
    assert metrics.ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert metrics.ndcg_at_k(["a"], set(), 1) == 0.0


@hint(
    fa="هر دو طرف را با match_key مقایسه کنید و برای پنجرهٔ جمله از context_text استفاده کنید.",
    en="Compare match_key on both sides, and use context_text for sentence windows.",
)
def test_contains_evidence(metrics: ModuleType) -> None:
    """contains_evidence matches the evidence span despite messy text"""
    item = QAItem(id="q", question="?", answer="", doc_id="A", evidence="هزینهٔ پیک 200 هزار")
    assert metrics.contains_evidence(
        Chunk(id="A#0", doc_id="A", text="هزينهٔ پيک ۲۰۰ هزار تومان"), item
    )
    assert not metrics.contains_evidence(
        Chunk(id="B#0", doc_id="B", text="هزینهٔ پیک 200 هزار"), item
    )
    window = Chunk(id="A#1", doc_id="A", text="پیک", metadata={"window": "هزینهٔ پیک 200 هزار"})
    assert metrics.contains_evidence(window, item)
    no_ev = QAItem(id="q", question="?", answer="", doc_id="A")
    assert not metrics.contains_evidence(Chunk(id="A#0", doc_id="A", text="هر چیزی"), no_ev)


def _r(doc: str, text: str, i: int = 0) -> SearchResult:
    return SearchResult(chunk=Chunk(id=f"{doc}#{i}", doc_id=doc, text=text), score=1.0)


@hint(
    fa="فقط پرسش‌های پاسخ‌دار را بشمارید؛ برای هر پرسش ranked_ids و مجموعهٔ مرتبط را بسازید و میانگین بگیرید.",
    en="Only answerable items count; build ranked_ids and the relevant set per item, then average.",
)
def test_evaluate_retrieval(metrics: ModuleType) -> None:
    """evaluate_retrieval averages the metrics at doc and passage level"""
    items = [
        QAItem(id="1", question="x", answer="", doc_id="A", evidence="جواب"),
        QAItem(id="2", question="y", answer="", doc_id="B", evidence="پاسخ"),
        QAItem(id="3", question="z", answer="", doc_id=None),
    ]
    results = {
        "x": [_r("A", "بی‌ربط", 0), _r("A", "این جواب است", 1)],
        "y": [_r("C", "پاسخ"), _r("D", "هیچ")],
    }

    def retrieve(q: str, k: int) -> list[SearchResult]:
        return results[q][:k]

    doc = metrics.evaluate_retrieval(items, retrieve, ks=(1, 2))
    assert (doc.recall, doc.mrr, len(doc.per_question)) == ({1: 0.5, 2: 0.5}, 0.5, 2)
    passage = metrics.evaluate_retrieval(items, retrieve, ks=(2, 1), level="passage")
    assert passage.recall == {1: 0.0, 2: 0.5}
    assert passage.mrr == pytest.approx(0.25)
    assert passage.ndcg[2] == pytest.approx(0.5 / math.log2(3))
    assert [q.rank for q in passage.per_question] == [2, None]
    with pytest.raises(ValueError):
        metrics.evaluate_retrieval(items, retrieve, ks=(0,))
    with pytest.raises(ValueError):
        metrics.evaluate_retrieval(
            [QAItem(id="1", question="x", answer="", doc_id="A")], retrieve, level="passage"
        )


# ----------------------------------------------------------------------- TF-IDF


@hint(
    fa="a·b / (|a| |b|)؛ اگر یکی از بردارها صفر بود، ۰ برگردانید.",
    en="a·b / (|a| |b|); return 0 if either vector is zero.",
)
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 0], [2, 0], 1.0),
        ([1, 0], [0, 3], 0.0),
        ([1, 1], [-1, -1], -1.0),
        ([3, 4], [4, 3], 0.96),
        ([0, 0], [1, 2], 0.0),
    ],
)
def test_cosine_similarity(
    tfidf: ModuleType, a: list[float], b: list[float], expected: float
) -> None:
    """cosine_similarity measures the angle between two vectors"""
    assert tfidf.cosine_similarity(a, b) == pytest.approx(expected)


@hint(
    fa="در حالت char هر کلمه را به شکل <کلمه> درآورید و همهٔ n-gramهای آن را برای n در ngram_range بسازید.",
    en="For 'char', pad each word as <word> and emit all its n-grams for n in ngram_range.",
)
def test_tfidf_terms(tfidf: ModuleType) -> None:
    """TfidfEmbedding.terms produces words or padded character n-grams"""
    assert tfidf.TfidfEmbedding("word").terms("کتاب خوب، Book") == ["کتاب", "خوب", "book"]
    assert tfidf.TfidfEmbedding("char", ngram_range=(2, 2)).terms("آب") == ["<آ", "آب", "ب>"]
    assert tfidf.TfidfEmbedding("char", ngram_range=(2, 3)).terms("آب") == [
        "<آ",
        "آب",
        "ب>",
        "<آب",
        "آب>",
    ]


@hint(
    fa="df = تعداد متن‌هایی که واژه را دارند (هر متن یک بار)؛ idf = log((1+N)/(1+df)) + 1.",
    en="df = number of texts containing the term (once per text); idf = log((1+N)/(1+df)) + 1.",
)
def test_tfidf_fit(tfidf: ModuleType) -> None:
    """fit learns a sorted vocabulary and IDF weights"""
    emb = tfidf.TfidfEmbedding().fit(["کتاب خوب است", "کتاب بد است", "باران باران"])
    assert list(emb.vocabulary) == sorted(emb.vocabulary)
    assert emb.dim == 5
    idf = dict(zip(emb.vocabulary, emb.idf, strict=True))
    assert idf["کتاب"] == pytest.approx(math.log(4 / 3) + 1)
    assert idf["باران"] == pytest.approx(math.log(4 / 2) + 1)
    assert tfidf.TfidfEmbedding(min_df=2).fit(["الف ب", "الف پ"]).dim == 1
    with pytest.raises(ValueError):
        tfidf.TfidfEmbedding().fit(["", " "])


@hint(
    fa="وزن هر واژه (1 + log(tf)) × idf است و سطرها باید طول ۱ داشته باشند (l2_normalize).",
    en="Weight = (1 + log tf) × idf; rows must have unit length (l2_normalize).",
)
def test_tfidf_embed(tfidf: ModuleType) -> None:
    """embed returns unit-length TF-IDF vectors; rare words weigh more"""
    emb = tfidf.TfidfEmbedding().fit(["کتاب خوب است", "کتاب بد است", "باران می‌بارد"])
    assert isinstance(emb, Embedding)
    vecs = emb.embed(["کتاب خوب", "باران باران باران"])
    assert vecs.shape == (2, emb.dim) and vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-5)
    v = vecs[0]
    assert v[emb.vocabulary["خوب"]] > v[emb.vocabulary["کتاب"]] > 0
    assert not emb.embed_query("ناشناخته").any()
    with pytest.raises(TypeError):
        emb.embed("یک رشته")


# ------------------------------------------------------------------ the payoff


def test_benchmark_compares_embeddings(metrics: ModuleType, tfidf: ModuleType) -> None:
    """your benchmark ranks the offline embeddings on the Persian fixtures"""
    docs, qa = load_fixture_benchmark()

    def run(embedding: Embedding) -> float:
        store = index_documents(docs, embedding)
        report = metrics.evaluate_retrieval(
            qa, lambda q, k: store.search(normalize(q), k), (1, 3), level="passage"
        )
        return report.mrr

    hashing = run(HashingEmbedding())
    word = run(tfidf.TfidfEmbedding("word"))
    char = run(tfidf.TfidfEmbedding("char"))
    assert 0.8 < word < hashing < char, f"MRR word={word:.3f} hashing={hashing:.3f} char={char:.3f}"

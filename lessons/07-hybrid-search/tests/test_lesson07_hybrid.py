"""Lesson 07 — BM25, dense and hybrid retrieval. Run with: python -m grader run 07"""

from __future__ import annotations

import math
from types import ModuleType

import pytest

from parsrag import HashingEmbedding
from parsrag.eval import evaluate_retrieval
from parsrag.eval.benchmark import index_documents, load_fixture_benchmark
from parsrag.types import Chunk, SearchResult

hint = pytest.mark.hint


def _chunks(*texts: str) -> list[Chunk]:
    return [Chunk(id=f"c{i}", doc_id=f"d{i % 2}", text=t) for i, t in enumerate(texts)]


def _results(*ids: str, scores: list[float] | None = None) -> list[SearchResult]:
    scores = scores or [float(len(ids) - i) for i in range(len(ids))]
    return [
        SearchResult(chunk=Chunk(id=i, doc_id="d", text=i), score=s)
        for i, s in zip(ids, scores, strict=True)
    ]


# ------------------------------------------------------------------ tokenizing


@hint(
    fa="اول normalize، بعد lower، بعد re.findall با الگوی [\\w‌]+ (نیم‌فاصله جزء کلمه است).",
    en="normalize, then lower(), then re.findall(r'[\\w\\u200c]+') — ZWNJ stays inside words.",
)
def test_tokenize_normalizes_and_keeps_zwnj_words(rt: ModuleType) -> None:
    """tokenize normalises, lowercases and keeps ZWNJ-joined words together"""
    assert rt.tokenize("كتاب هاي  من، PDF!") == ["کتاب‌های", "من", "pdf"]
    assert rt.tokenize("۱۲ هزار تومان") == ["12", "هزار", "تومان"]
    assert rt.tokenize("") == []


@hint(
    fa="split_zwnj یعنی نیم‌فاصله را با فاصله جایگزین کنید؛ stopwords را بعد از توکن‌سازی حذف کنید.",
    en="split_zwnj: replace ZWNJ with a space; drop stop words after tokenising.",
)
def test_tokenize_options(rt: ModuleType) -> None:
    """tokenize can split at ZWNJ and drop stop words"""
    assert rt.tokenize("کتاب‌ها را", split_zwnj=True) == ["کتاب", "ها", "را"]
    assert rt.tokenize("کتاب‌ها را به من بده", stopwords={"را", "به"}) == ["کتاب‌ها", "من", "بده"]


@hint(
    fa="بعد از نیم‌فاصله هر پسوند فهرست SUFFIXES حذف می‌شود؛ پسوند چسبیده فقط اگر دست‌کم ۳ حرف بماند.",
    en="After a ZWNJ any suffix in SUFFIXES goes; an attached suffix only if >= 3 letters remain.",
)
@pytest.mark.parametrize(
    ("token", "stem"),
    [
        ("کتاب‌ها", "کتاب"),
        ("کتاب‌های", "کتاب"),
        ("بزرگ‌ترین", "بزرگ"),
        ("مرجوعی", "مرجوع"),
        ("شلوغ‌ترین", "شلوغ"),
        ("کتابها", "کتاب"),
        ("تنها", "تنها"),
        ("می‌شود", "می‌شود"),
        ("کار", "کار"),
    ],
)
def test_light_stem(rt: ModuleType, token: str, stem: str) -> None:
    """light_stem strips one suffix, conservatively"""
    assert rt.light_stem(token) == stem


# ------------------------------------------------------------------------ BM25


@hint(
    fa="idf = log(1 + (N - df + 0.5) / (df + 0.5)) — هیچ‌وقت منفی نمی‌شود.",
    en="idf = log(1 + (N - df + 0.5) / (df + 0.5)) — never negative.",
)
def test_bm25_idf(rt: ModuleType) -> None:
    """bm25_idf uses the non-negative Lucene formula"""
    assert rt.bm25_idf(10, 1) == pytest.approx(math.log(1 + 9.5 / 1.5))
    assert rt.bm25_idf(10, 10) > 0
    assert rt.bm25_idf(10, 1) > rt.bm25_idf(10, 5)


@hint(
    fa="برای هر واژهٔ پرسش (یک بار): idf × tf × (k1+1) / (tf + k1 × (1 - b + b × |d| / avgdl)).",
    en="Per distinct query term: idf · tf · (k1+1) / (tf + k1 · (1 - b + b·|d|/avgdl)).",
)
def test_bm25_scores_match_the_formula(rt: ModuleType) -> None:
    """BM25 scores equal the textbook formula"""
    chunks = _chunks("پیک تهران پیک", "پست پیشتاز شهرستان", "ارسال رایگان تهران")
    bm25 = rt.BM25Retriever(chunks, k1=1.2, b=0.75)
    avgdl = 3.0

    def term(tf: int, dl: int, df: int) -> float:
        idf = math.log(1 + (3 - df + 0.5) / (df + 0.5))
        return idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))

    results = bm25.search("هزینهٔ پیک در تهران", k=5)
    assert [r.chunk.id for r in results] == ["c0", "c2"]
    assert results[0].score == pytest.approx(term(2, 3, 1) + term(1, 3, 2))
    assert results[1].score == pytest.approx(term(1, 3, 2))


@hint(
    fa="فقط قطعه‌هایی با امتیاز > ۰؛ مرتب‌سازی نزولی؛ k تا؛ فیلتر را با matches_filter اعمال کنید.",
    en="Only chunks with score > 0, sorted descending, at most k, filter via matches_filter.",
)
def test_bm25_search_k_filter_and_ids(rt: ModuleType) -> None:
    """BM25 search respects k, filters and rejects duplicate ids"""
    chunks = _chunks("کتاب امانت", "کتاب جریمه", "کتاب ساعت", "باشگاه")
    bm25 = rt.BM25Retriever(chunks)
    assert len(bm25) == 4
    assert len(bm25.search("کتاب", k=2)) == 2
    assert bm25.search("کتاب", k=0) == []
    assert bm25.search("ناموجود", k=3) == []
    assert {r.chunk.doc_id for r in bm25.search("کتاب", k=5, filter={"doc_id": "d1"})} == {"d1"}
    assert bm25.search("جریمهٔ کتاب", k=1)[0].chunk.id == "c1"
    with pytest.raises(ValueError):
        bm25.add([Chunk(id="c0", doc_id="x", text="تکراری")])


@hint(
    fa="همان توکنایزر برای قطعه‌ها و پرسش؛ پیش‌فرض PersianTokenizer() است.",
    en="Use the same tokenizer for chunks and queries; default to PersianTokenizer().",
)
def test_bm25_uses_its_tokenizer(rt: ModuleType) -> None:
    """BM25 with stemming matches inflected forms"""
    chunks = _chunks("کالای مرجوع", "کتاب")
    plain = rt.BM25Retriever(chunks)
    stem = rt.BM25Retriever(chunks, tokenizer=rt.PersianTokenizer(stem=True))
    assert plain.search("مرجوعی", k=3) == []
    assert stem.search("مرجوعی", k=3)[0].chunk.id == "c0"


# ----------------------------------------------------------------------- dense


@hint(
    fa="پرسش را با self.preprocess نرمال کنید و بعد store.search را صدا بزنید.",
    en="Apply self.preprocess to the query, then call store.search.",
)
def test_dense_retriever_normalizes_the_query(rt: ModuleType) -> None:
    """DenseRetriever normalises the query before searching the store"""
    seen: list[str] = []

    class SpyStore:
        def search(self, query: str, k: int = 4, *, filter: object = None) -> list[SearchResult]:
            seen.append(query)
            return _results("a")[:k]

    dense = rt.DenseRetriever(SpyStore())
    assert [r.chunk.id for r in dense.search("كتاب", k=1)] == ["a"]
    assert seen == ["کتاب"]


# ---------------------------------------------------------------------- fusion


@hint(
    fa="امتیاز هر قطعه = Σ w / (k + rank) با rank از ۱؛ بر اساس chunk.id جمع بزنید.",
    en="Score per chunk id = Σ w / (k + rank), rank starting at 1.",
)
def test_reciprocal_rank_fusion(rt: ModuleType) -> None:
    """RRF sums 1/(k + rank) over the lists"""
    fused = rt.reciprocal_rank_fusion([_results("a", "b", "c"), _results("b", "d")], k=60)
    assert [r.chunk.id for r in fused] == ["b", "a", "d", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    weighted = rt.reciprocal_rank_fusion(
        [_results("a", "b"), _results("b", "a")], k=0, weights=[3.0, 1.0], limit=1
    )
    assert [r.chunk.id for r in weighted] == ["a"]
    assert weighted[0].score == pytest.approx(3.0 + 0.5)


@hint(
    fa="minmax: (s - min) / (max - min)، و اگر همه برابرند ۱. سپس Σ w × امتیاز نرمال‌شده.",
    en="minmax: (s - min) / (max - min), all 1.0 if equal; then Σ w · normalised score.",
)
def test_weighted_score_fusion(rt: ModuleType) -> None:
    """weighted fusion min-max normalises each list before adding"""
    assert rt.normalize_scores(_results("a", "b", "c", scores=[10.0, 6.0, 2.0])) == [1.0, 0.5, 0.0]
    assert rt.normalize_scores(_results("a", "b", scores=[3.0, 3.0])) == [1.0, 1.0]
    bm25 = _results("a", "b", "c", scores=[12.0, 11.0, 1.0])
    dense = _results("c", "a", scores=[0.9, 0.1])
    fused = rt.weighted_score_fusion([bm25, dense], [0.5, 0.5])
    assert [r.chunk.id for r in fused] == ["a", "c", "b"]
    assert fused[0].score == pytest.approx(0.5)
    assert fused[1].score == pytest.approx(0.5 * 0.0 + 0.5 * 1.0)


@hint(
    fa="از هر بازیاب max(fetch_k, k) نتیجه بگیرید (پیش‌فرض ۲۰)، ادغام کنید و k تای اول را برگردانید.",
    en="Fetch max(fetch_k, k) (default 20) from each retriever, fuse, return the top k.",
)
def test_hybrid_retriever(rt: ModuleType) -> None:
    """HybridRetriever fetches from every retriever and fuses"""
    calls: list[int] = []

    class Fixed:
        def __init__(self, ids: tuple[str, ...]) -> None:
            self.ids = ids

        def search(self, query: str, k: int = 4, *, filter: object = None) -> list[SearchResult]:
            calls.append(k)
            return _results(*self.ids)[:k]

    hybrid = rt.HybridRetriever([Fixed(("a", "b", "c")), Fixed(("b", "d"))])
    assert [r.chunk.id for r in hybrid.search("q", k=2)] == ["b", "a"]
    assert calls == [20, 20]
    weighted = rt.HybridRetriever(
        [Fixed(("a", "b")), Fixed(("b", "a"))], fusion="weighted", weights=[0.9, 0.1]
    )
    assert weighted.search("q", k=1)[0].chunk.id == "a"


# ------------------------------------------------------------------ the payoff


def _mrr(qa: list, retriever: object) -> float:  # type: ignore[type-arg]
    return evaluate_retrieval(
        qa,
        lambda q, k: retriever.search(q, k),
        (1, 3),
        level="passage",  # type: ignore[attr-defined]
    ).mrr


def test_hybrid_beats_both_on_the_fixtures(rt: ModuleType) -> None:
    """on the course corpus, BM25 + dense with RRF beats each alone (and paraphrases hurt)"""
    docs, qa = load_fixture_benchmark()
    _, para = load_fixture_benchmark(qa_file="qa_paraphrase.jsonl")
    store = index_documents(docs, HashingEmbedding())
    bm25 = rt.BM25Retriever(store.chunks)
    dense = rt.DenseRetriever(store)
    hybrid = rt.HybridRetriever([bm25, dense])
    m_bm25, m_dense, m_hybrid = (_mrr(qa, r) for r in (bm25, dense, hybrid))
    assert m_bm25 > 0.83 and m_dense > 0.83
    assert m_hybrid > max(m_bm25, m_dense)
    assert _mrr(para, hybrid) < m_hybrid - 0.3

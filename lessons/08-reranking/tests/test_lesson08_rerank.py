"""Lesson 08 — reranking. Run with: python -m grader run 08"""

from __future__ import annotations

from types import ModuleType

import pytest

from parsrag import FakeLLM, HashingEmbedding, Message
from parsrag.eval import evaluate_retrieval
from parsrag.eval.benchmark import index_documents, load_fixture_benchmark
from parsrag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, PersianTokenizer
from parsrag.types import Chunk, SearchResult

hint = pytest.mark.hint


def _results(*texts: str) -> list[SearchResult]:
    return [
        SearchResult(chunk=Chunk(id=f"c{i}", doc_id="d", text=t), score=1.0 - i / 10)
        for i, t in enumerate(texts)
    ]


# --------------------------------------------------------------- building blocks


@hint(
    fa="ترتیب نزولی امتیازها؛ برای امتیازهای برابر ترتیب قبلی حفظ شود (sorted پایدار است).",
    en="Sort by score descending; equal scores keep their old order (sorted is stable).",
)
def test_sort_by_scores(rr: ModuleType) -> None:
    """sort_by_scores re-scores and re-orders, keeping ties stable"""
    res = _results("a", "b", "c")
    out = rr.sort_by_scores(res, [0.1, 0.5, 0.1], None)
    assert [r.chunk.id for r in out] == ["c1", "c0", "c2"]
    assert [r.score for r in out] == [0.5, 0.1, 0.1]
    assert len(rr.sort_by_scores(res, [3, 2, 1], 2)) == 2
    with pytest.raises(ValueError):
        rr.sort_by_scores(res, [1.0], None)


@hint(
    fa="دو اشاره‌گر روی دو رشته؛ اولین اختلاف را «مصرف» کنید و اگر اختلاف دوم بود False.",
    en="Walk both strings; allow one mismatch (substitute, or skip a char in the longer one).",
)
@pytest.mark.parametrize(
    ("a", "b", "ok"),
    [
        ("قسطی", "قصطی", True),
        ("تخفیف", "تحفیف", True),
        ("کتاب", "کتابی", True),
        ("کتاب", "کاب", True),
        ("کتاب", "کتاب", True),
        ("کتاب", "کباب‌ها", False),
        ("پیک", "پست", False),
    ],
)
def test_edit_distance_at_most_one(rr: ModuleType, a: str, b: str, ok: bool) -> None:
    """edit_distance_at_most_one tolerates exactly one typo"""
    assert rr.edit_distance_at_most_one(a, b) is ok


@hint(
    fa="یکسان → ۱٫۰؛ اگر هر دو دست‌کم ۴ حرف دارند و یکی پیشوند دیگری است یا یک غلط تایپی فاصله دارند → ۰٫۷.",
    en="Equal → 1.0; both ≥ 4 letters and prefix-related or one typo apart → 0.7; else 0.",
)
def test_term_match(rr: ModuleType) -> None:
    """term_match: exact 1.0, near 0.7, otherwise 0"""
    assert rr.term_match("پیک", "پیک") == 1.0
    assert rr.term_match("قصطی", "قسطی") == 0.7
    assert rr.term_match("امتیاز", "امتیازها") == 0.7
    assert rr.term_match("پیک", "پیکر") == 0.0  # too short for fuzzy matching
    assert rr.term_match("کرایه", "هزینه") == 0.0


@hint(
    fa="پنجرهٔ لغزان روی موقعیت‌های مرتب‌شده: سمت راست را جلو ببرید و تا همهٔ واژه‌ها هستند سمت چپ را جمع کنید.",
    en="Sliding window over sorted positions: extend right, shrink left while all terms are inside.",
)
def test_min_window(rr: ModuleType) -> None:
    """min_window finds the shortest span containing every term"""
    assert rr.min_window({"a": [0, 9], "b": [5], "c": [7, 20]}) == 5  # 5..9
    assert rr.min_window({"a": [3]}) == 1
    assert rr.min_window({}) == 0


# ------------------------------------------------------------------ rerankers


@hint(
    fa="coverage = Σ وزن × بهترین تطبیق / Σ وزن‌ها؛ proximity = تعداد واژه‌های پیداشده / min_window (اگر ≥ ۲ واژه).",
    en="coverage = Σ w·best match / Σ w; proximity = matched / min_window (only if ≥ 2 matched).",
)
def test_lexical_score(rr: ModuleType) -> None:
    """LexicalReranker.score = coverage + proximity_weight · proximity"""
    plain = rr.LexicalReranker()
    assert plain.score("هزینهٔ پیک تهران", "پیک تهران") == pytest.approx(2 / 3)
    assert plain.score("هزینهٔ پیک تهران", "هیچ ربطی ندارد") == 0.0
    assert plain.score("", "پیک") == 0.0
    prox = rr.LexicalReranker(proximity_weight=0.5)
    near = prox.score("پیک تهران", "پیک تهران رایگان است")
    far = prox.score("پیک تهران", "پیک و پست و باربری برای تهران")
    assert near == pytest.approx(1.0 + 0.5 * 2 / 2)
    assert far == pytest.approx(1.0 + 0.5 * 2 / 4)  # stop words are dropped first
    weighted = rr.LexicalReranker(idf=lambda t: 3.0 if t == "پیک" else 1.0)
    assert weighted.score("پیک تهران", "پیک") == pytest.approx(3 / 4)


@hint(
    fa="امتیاز نهایی = score(query, text) + prior_weight / rank (rank از ۱)؛ بعد sort_by_scores.",
    en="final = score(query, text) + prior_weight / rank (rank from 1); then sort_by_scores.",
)
def test_lexical_rerank(rr: ModuleType) -> None:
    """LexicalReranker moves the chunk that covers the question to the top"""
    res = _results("ساعت کاری فروشگاه", "هزینهٔ ارسال با پیک در تهران", "پیک")
    out = rr.LexicalReranker().rerank("هزینهٔ پیک در تهران چقدر است؟", res, top_k=2)
    assert [r.chunk.id for r in out] == ["c1", "c2"]
    tie = rr.LexicalReranker(prior_weight=0.2).rerank("پیک", _results("پیک", "پیک"))
    assert [r.chunk.id for r in tie] == ["c0", "c1"]
    assert tie[0].score == pytest.approx(1.2)
    assert rr.LexicalReranker().rerank("x", []) == []


@hint(
    fa="اول الگوی «امتیاز: عدد»، وگرنه اولین عدد؛ ارقام فارسی را به لاتین تبدیل و در [0, 10] محدود کنید.",
    en="Prefer 'امتیاز: N', else the first number; map Persian digits; clip to [0, 10].",
)
@pytest.mark.parametrize(
    ("reply", "score"),
    [
        ("امتیاز: 7", 7.0),
        ("امتیاز: ۸", 8.0),
        ("فکر می‌کنم ۹٫۵ است", 9.5),
        ("Score: 3", 3.0),
        ("امتیاز: 42", 10.0),
        ("نمی‌دانم", None),
    ],
)
def test_parse_score(rr: ModuleType, reply: str, score: float | None) -> None:
    """parse_score reads the grade from an LLM reply"""
    assert rr.parse_score(reply) == score


@hint(
    fa="برای هر نامزد یک بار llm.generate(self.messages(query, text))؛ پاسخ نامفهوم → default_score.",
    en="One llm.generate(self.messages(query, text)) per candidate; unparseable → default_score.",
)
def test_llm_reranker(rr: ModuleType) -> None:
    """LLMReranker asks the LLM once per candidate and sorts by its grades"""

    def judge(messages: list[Message]) -> str:
        return "امتیاز: 9" if "تهران" in messages[-1].content.split("متن:")[1] else "؟"

    llm = FakeLLM(handler=judge)
    out = rr.LLMReranker(llm).rerank("هزینهٔ پیک؟", _results("پست", "پیک تهران", "پیک"))
    assert [r.chunk.id for r in out] == ["c1", "c0", "c2"]
    assert out[0].score == 9.0 and out[1].score == 0.0
    assert len(llm.calls) == 3
    assert "هزینهٔ پیک؟" in llm.calls[0][-1].content


@hint(
    fa="max(fetch_k, k) نامزد از retriever بگیرید، بعد reranker.rerank(query, candidates, top_k=k).",
    en="Fetch max(fetch_k, k) candidates, then reranker.rerank(query, candidates, top_k=k).",
)
def test_reranking_retriever(rr: ModuleType) -> None:
    """RerankingRetriever = retrieve many, rerank few"""
    asked: list[int] = []

    class Fixed:
        def search(self, query: str, k: int = 4, *, filter: object = None) -> list[SearchResult]:
            asked.append(k)
            return _results("الف", "ب", "پیک تهران")[:k]

    pipe = rr.RerankingRetriever(Fixed(), rr.LexicalReranker(), fetch_k=20)
    assert [r.chunk.id for r in pipe.search("پیک تهران", k=1)] == ["c2"]
    assert asked == [20]
    assert pipe.search("پیک", k=0) == []


# ------------------------------------------------------------------ the payoff


def test_reranking_improves_r_at_1_on_the_fixtures(rr: ModuleType) -> None:
    """reranking 20 hybrid candidates improves R@1 on the original questions"""
    docs, qa = load_fixture_benchmark()
    store = index_documents(docs, HashingEmbedding())
    bm25 = BM25Retriever(store.chunks, tokenizer=PersianTokenizer(stem=True))
    hybrid = HybridRetriever([bm25, DenseRetriever(store)])
    reranked = rr.RerankingRetriever(hybrid, rr.LexicalReranker(), fetch_k=20)

    def r_at_1(retriever: object) -> float:
        return evaluate_retrieval(
            qa,
            lambda q, k: retriever.search(q, k),
            (1,),
            level="passage",
            depth=10,  # type: ignore[attr-defined]
        ).recall[1]

    assert r_at_1(reranked) > r_at_1(hybrid)

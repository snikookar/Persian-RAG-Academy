"""Lesson 09 — grounded generation with citations. Run with: python -m grader run 09"""

from __future__ import annotations

from types import ModuleType

import pytest

from parsrag import FakeLLM, HashingEmbedding
from parsrag.eval.benchmark import index_documents, load_fixture_benchmark
from parsrag.eval.retrieval import contains_evidence
from parsrag.rerank import LexicalReranker, RerankingRetriever
from parsrag.retrieval import (
    PERSIAN_STOPWORDS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    PersianTokenizer,
)
from parsrag.types import Chunk, SearchResult

hint = pytest.mark.hint

META = {"section": "شیوه‌های ارسال", "page": 2, "start": 10, "end": 60}


def _results() -> list[SearchResult]:
    return [
        SearchResult(
            chunk=Chunk(
                id="ship#0",
                doc_id="shipping",
                text="هزینهٔ پیک در تهران ۲۰۰ هزار تومان است.",
                metadata=META,
            ),
            score=12.0,
        ),
        SearchResult(
            chunk=Chunk(id="faq#3", doc_id="faq", text="پشتیبانی تا ۹ شب پاسخ می‌دهد."), score=3.0
        ),
    ]


class Fixed:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, query: str, k: int = 4, *, filter: object = None) -> list[SearchResult]:
        return self.results[:k]


# ------------------------------------------------------------------ prompting


@hint(
    fa="«[n] doc_id» و اگر section بود « › section» و اگر page بود « (صفحهٔ p)».",
    en="'[n] doc_id', plus ' › section' and ' (صفحهٔ p)' when present.",
)
def test_block_header_and_context(gen: ModuleType) -> None:
    """context blocks are numbered and labelled with their source"""
    res = _results()
    assert gen.block_header(1, res[0].chunk) == "[1] shipping › شیوه‌های ارسال (صفحهٔ 2)"
    assert gen.block_header(2, res[1].chunk) == "[2] faq"
    ctx = gen.build_context(res)
    assert ctx == (
        "[1] shipping › شیوه‌های ارسال (صفحهٔ 2)\nهزینهٔ پیک در تهران ۲۰۰ هزار تومان است."
        "\n\n[2] faq\nپشتیبانی تا ۹ شب پاسخ می‌دهد."
    )
    windowed = Chunk(
        id="x", doc_id="d", text="کوتاه", metadata={"window": "جملهٔ قبل. کوتاه. جملهٔ بعد."}
    )
    assert "جملهٔ قبل" in gen.build_context([SearchResult(chunk=windowed, score=1.0)])


@hint(
    fa="دو پیام: system با SYSTEM_PROMPT و user با USER_TEMPLATE.format(context=..., question=...).",
    en="Two messages: system = SYSTEM_PROMPT, user = USER_TEMPLATE.format(context=..., question=...).",
)
def test_build_messages(gen: ModuleType) -> None:
    """the prompt has the rules, the numbered context and the question"""
    msgs = gen.build_messages("هزینهٔ پیک؟", _results())
    assert [m.role for m in msgs] == ["system", "user"]
    assert "نمی‌دانم" in msgs[0].content and "[1]" in msgs[0].content
    assert msgs[1].content.startswith("متن‌ها:\n[1] shipping")
    assert msgs[1].content.endswith("پرسش: هزینهٔ پیک؟")


# ----------------------------------------------------------------- citations


@hint(
    fa="الگوی [عدد] با ارقام فارسی/لاتین و فهرست‌هایی مثل [1، 3]؛ هر شماره یک بار، به ترتیب اولین ظهور.",
    en="Match [n] with Persian/Latin digits and lists like [1، 3]; unique, in order of first appearance.",
)
@pytest.mark.parametrize(
    ("text", "cited"),
    [
        ("۲۰۰ هزار تومان است [1].", [1]),
        ("طبق [۲] و [1]، بله [2].", [2, 1]),
        ("بله [1، 3]", [1, 3]),
        ("بله [2,4]", [2, 4]),
        ("بدون ارجاع", []),
        ("آرایه [a] نیست", []),
    ],
)
def test_parse_citations(gen: ModuleType, text: str, cited: list[int]) -> None:
    """parse_citations finds [n] markers"""
    assert gen.parse_citations(text) == cited


@hint(
    fa="متن را normalize کنید و دنبال «نمی‌دانم» یا گونه‌های نزدیکش بگردید.",
    en="Normalise the text and look for «نمی‌دانم» or a close variant.",
)
@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        ("نمی‌دانم", True),
        ("نمیدانم.", True),
        ("متأسفم، نمي دانم.", True),
        ("این اطلاعات در متن‌ها نیست.", True),
        ("۲۰۰ هزار تومان [1]", False),
    ],
)
def test_is_refusal(gen: ModuleType, text: str, refusal: bool) -> None:
    """is_refusal recognises «نمی‌دانم» and its variants"""
    assert gen.is_refusal(text) is refusal


@hint(
    fa="valid = شماره‌های بین ۱ و n_context؛ invalid = بقیه؛ uncited = هیچ valid ندارد و پاسخ «نمی‌دانم» نیست.",
    en="valid = 1..n_context; invalid = the rest; uncited = no valid citation and not a refusal.",
)
def test_validate_citations(gen: ModuleType) -> None:
    """validate_citations flags made-up block numbers and uncited answers"""
    ok = gen.validate_citations("بله [1][2]", 2)
    assert ok.valid == [1, 2] and ok.invalid == [] and not ok.uncited
    bad = gen.validate_citations("بله [3]", 2)
    assert bad.valid == [] and bad.invalid == [3] and bad.uncited
    assert gen.validate_citations("بله.", 2).uncited
    assert not gen.validate_citations("نمی‌دانم", 2).uncited


@hint(
    fa="number، chunk_id، doc_id و section/page/start/end را از chunk.metadata بردارید؛ score هم از نتیجه.",
    en="Copy number, chunk_id, doc_id, metadata section/page/start/end, and the result score.",
)
def test_to_citation(gen: ModuleType) -> None:
    """citations map back to the exact source span"""
    c = gen.to_citation(1, _results()[0])
    assert (c.number, c.chunk_id, c.doc_id) == (1, "ship#0", "shipping")
    assert (c.section, c.page, c.start, c.end, c.score) == ("شیوه‌های ارسال", 2, 10, 60, 12.0)
    bare = gen.to_citation(2, _results()[1])
    assert bare.section is None and bare.page is None


# ----------------------------------------------------------------- generator


@hint(
    fa="اگر confidence(question, results) < min_score، بدون صدا زدن LLM با abstain_reason='low_score' برگردید.",
    en="If confidence(question, results) < min_score, return abstain_reason='low_score' without calling the LLM.",
)
def test_generator_abstains_on_low_score(gen: ModuleType) -> None:
    """a low retrieval score means «نمی‌دانم» without an LLM call"""
    llm = FakeLLM(["نباید صدا زده شود [1]"])
    g = gen.GroundedGenerator(Fixed(_results()), llm, min_score=20.0)
    a = g.answer("هزینهٔ پیک؟")
    assert a.abstained and a.abstain_reason == "low_score" and a.answer == "نمی‌دانم"
    assert a.top_score == 12.0 and llm.calls == []
    empty = gen.GroundedGenerator(Fixed([]), llm).answer("؟")
    assert empty.abstained and empty.abstain_reason == "no_results"


@hint(
    fa="پاسخ LLM اگر is_refusal بود → abstain_reason='model'؛ وگرنه validate_citations و to_citation برای شماره‌های معتبر.",
    en="A refusal reply → abstain_reason='model'; otherwise validate and map valid numbers with to_citation.",
)
def test_generator_answers_with_citations(gen: ModuleType) -> None:
    """answers keep valid citations and report invalid ones"""
    g = gen.GroundedGenerator(Fixed(_results()), FakeLLM(["۲۰۰ هزار تومان [1] [5]"]))
    a = g.answer("هزینهٔ پیک؟")
    assert not a.abstained and a.answer == "۲۰۰ هزار تومان [1] [5]"
    assert [c.chunk_id for c in a.citations] == ["ship#0"]
    assert a.invalid_citations == [5] and not a.uncited
    assert a.text == "۲۰۰ هزار تومان"
    refused = gen.GroundedGenerator(Fixed(_results()), FakeLLM(["نمی‌دانم."])).answer("؟")
    assert refused.abstained and refused.abstain_reason == "model"
    extractive = FakeLLM(mode="extractive", cite=True)
    b = gen.GroundedGenerator(Fixed(_results()), extractive).answer(
        "پشتیبانی تا چه ساعتی پاسخ می‌دهد؟"
    )
    assert b.answer == "پشتیبانی تا ۹ شب پاسخ می‌دهد. [2]"
    assert b.citations[0].doc_id == "faq"


# --------------------------------------------------------------- calibration


@hint(
    fa="مثبت = «باید نمی‌دانم بگوید» (پرسش بی‌پاسخ). tp = abstain و بی‌پاسخ، fp = abstain و پاسخ‌دار، ...",
    en="Positive = should abstain (unanswerable). tp = abstained & unanswerable, fp = abstained & answerable, ...",
)
def test_abstention_stats(gen: ModuleType) -> None:
    """abstention precision / recall / F1"""
    s = gen.abstention_stats([True, True, False, False, False], [False, True, False, True, True])
    assert (s.tp, s.fp, s.fn, s.tn) == (1, 1, 1, 2)
    assert s.precision == pytest.approx(0.5) and s.recall == pytest.approx(0.5)
    assert s.f1 == pytest.approx(0.5) and s.accuracy == pytest.approx(0.6)
    with pytest.raises(ValueError):
        gen.abstention_stats([True], [])


@hint(
    fa="آستانه‌های نامزد: -inf، وسط هر دو امتیاز متوالی، +inf؛ abstain اگر score < t؛ بیشترین F1، در تساوی کوچک‌ترین t.",
    en="Candidates: -inf, midpoints of consecutive distinct scores, +inf; abstain if score < t; best F1, ties → lowest t.",
)
def test_calibrate_threshold(gen: ModuleType) -> None:
    """calibrate_threshold picks the cut that best separates the two groups"""
    t, s = gen.calibrate_threshold([1.0, 2.0, 5.0, 6.0], [False, False, True, True])
    assert t == pytest.approx(3.5) and s.f1 == 1.0
    t2, s2 = gen.calibrate_threshold([5.0, 6.0], [True, True])
    assert t2 == float("-inf") and s2.f1 == 0.0


# ------------------------------------------------------------------ the payoff


def test_grounded_pipeline_on_the_fixtures(gen: ModuleType) -> None:
    """on the course corpus: calibrated abstention works and citations point at the evidence"""
    docs, qa = load_fixture_benchmark()
    store = index_documents(docs, HashingEmbedding())
    bm25 = BM25Retriever(store.chunks, tokenizer=PersianTokenizer(stem=True))
    retriever = RerankingRetriever(
        HybridRetriever([bm25, DenseRetriever(store)]), LexicalReranker()
    )

    def conf(question: str, results: object) -> float:
        return bm25.search(question, 1)[0].score

    t, stats = gen.calibrate_threshold(
        [conf(q.question, None) for q in qa], [q.answerable for q in qa]
    )
    assert stats.f1 >= 0.6
    llm = FakeLLM(mode="extractive", cite=True, ignore=PERSIAN_STOPWORDS)
    g = gen.GroundedGenerator(retriever, llm, min_score=t, confidence=conf)
    answered = [(q, g.answer(q.question)) for q in qa if q.answerable]
    answered = [(q, a) for q, a in answered if not a.abstained]
    assert len(answered) >= 50
    assert all(a.citations and not a.invalid_citations for _, a in answered)
    by_id = {r.chunk.id: r.chunk for _, a in answered for r in a.sources}
    supported = sum(
        any(contains_evidence(by_id[c.chunk_id], q) for c in a.citations) for q, a in answered
    )
    assert supported / len(answered) >= 0.8

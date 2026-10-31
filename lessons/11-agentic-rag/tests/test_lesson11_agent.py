"""Lesson 11 — agentic RAG with LangGraph. Run with: python -m grader run 11"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

from parsrag import PersianRAG
from parsrag.agent import OfflineAgentLLM

hint = pytest.mark.hint

PIK = "هزینهٔ پیک در تهران چقدر است؟"
NO_ANSWER = "قیمت بلیت هواپیما به استانبول چقدر است؟"
EXTRA = "LangGraph checks need: pip install -e '.[agent]'"


@pytest.fixture(scope="module")
def rag() -> PersianRAG:
    return PersianRAG.course_default()


def offline(rag: PersianRAG, **kwargs: Any) -> OfflineAgentLLM:
    return OfflineAgentLLM(vocabulary=rag.bm25._postings, **kwargs)


def hallucinating(rag: PersianRAG) -> OfflineAgentLLM:
    return offline(rag, overrides={"generate": lambda m: "هزینهٔ پیک ۹۹ هزار تومان است [1]."})


# ------------------------------------------------------------------ parsers


@hint(
    fa="اول بعد از هر «؟» ببُرید؛ بعد هر تکه را در «؛» و «و» ببُرید فقط اگر همهٔ طرف‌ها واژهٔ پرسشی دارند.",
    en="Split after every «؟»; then split at «؛» / «و» only if every side has a question word.",
)
def test_split_questions(ag: ModuleType) -> None:
    """multi-part questions are split, single ones are kept"""
    assert ag.split_questions("هزینهٔ پیک چقدر است و مرجوعی تا کی ممکن است؟") == [
        "هزینهٔ پیک چقدر است؟",
        "مرجوعی تا کی ممکن است؟",
    ]
    assert ag.split_questions("ساعت کاری چیست؟ جریمه چقدر است") == [
        "ساعت کاری چیست؟",
        "جریمه چقدر است؟",
    ]
    assert ag.split_questions("قوانین ارسال و مرجوعی چیست؟") == ["قوانین ارسال و مرجوعی چیست؟"]
    assert ag.split_questions("ساعت کار؛ قیمت چقدر است؟") == ["ساعت کار؛ قیمت چقدر است؟"]


@hint(
    fa="هر سطر را با _field بخوانید؛ مسیر را با _ROUTES نگاشت کنید؛ پیش‌فرض‌ها: documents، خود پرسش، [query].",
    en="Read each line with _field; map the route with _ROUTES; defaults: documents, the question, [query].",
)
def test_parse_analysis(ag: ModuleType) -> None:
    """the analyse step's reply is read into route, query, sub-questions"""
    a = ag.parse_analysis(
        "مسیر: اسناد\nبازنویسی: هزینهٔ پیک چقدر است؟\nزیرپرسش: الف؟\nزیرپرسش: ب؟", "x"
    )
    assert (a.route, a.query, a.sub_questions) == (
        "documents",
        "هزینهٔ پیک چقدر است؟",
        ["الف؟", "ب؟"],
    )
    b = ag.parse_analysis("مسیر: «ابهام»", "اون چنده؟")
    assert (b.route, b.query, b.sub_questions) == ("clarify", "اون چنده؟", ["اون چنده؟"])
    assert b.clarification
    c = ag.parse_analysis("مسیر: خارج\nپرسش روشن‌کننده: ", "q")
    assert c.route == "out_of_scope" and c.clarification is None
    assert ag.parse_analysis("جواب بی‌ربط", "q").route == "documents"


@hint(
    fa="با _GRADE همهٔ «[n] بله/خیر» را پیدا کنید؛ ارقام فارسی را تبدیل کنید؛ بقیه False.",
    en="Find every '[n] بله/خیر' with _GRADE; convert Persian digits; everything else is False.",
)
def test_parse_grades(ag: ModuleType) -> None:
    """grading replies become one bool per passage"""
    assert ag.parse_grades("[1] بله\n[۲]: خیر\n[3] yes\n[9] بله", 4) == [True, False, True, False]
    assert ag.parse_grades("نمی‌دانم", 2) == [False, False]


# ------------------------------------------------------------------ routers


@hint(
    fa="مسیر documents → retrieve، clarify → clarify، out_of_scope → refuse.",
    en="documents → retrieve, clarify → clarify, out_of_scope → refuse.",
)
def test_route_after_analyze(ag: ModuleType, rag: PersianRAG) -> None:
    """the first router sends each route to the right node"""
    agent = ag.AgenticRAG(rag)
    assert agent.route_after_analyze({"route": "documents"}) == "retrieve"
    assert agent.route_after_analyze({"route": "clarify"}) == "clarify"
    assert agent.route_after_analyze({"route": "out_of_scope"}) == "refuse"


@hint(
    fa="متن مرتبط → generate؛ وگرنه تا can_retry درست است rewrite؛ وگرنه refuse.",
    en="Relevant passages → generate; else rewrite while can_retry; else refuse.",
)
def test_route_after_grade(ag: ModuleType, rag: PersianRAG) -> None:
    """no relevant passage → rewrite and retry, but only a bounded number of times"""
    agent = ag.AgenticRAG(rag, max_retries=2)
    assert agent.route_after_grade({"relevant": [{"x": 1}], "attempt": 3}) == "generate"
    assert agent.route_after_grade({"relevant": [], "attempt": 1}) == "rewrite"
    assert agent.route_after_grade({"relevant": [], "attempt": 3}) == "refuse"


@hint(
    fa="امتیاز ≥ آستانه → finish؛ وگرنه rewrite (اگر می‌شود)؛ وگرنه human_review اگر review روشن است و پیش‌نویس جواب داده؛ وگرنه refuse.",
    en="score ≥ threshold → finish; else rewrite if possible; else human_review if review is on and the draft answered; else refuse.",
)
def test_route_after_check(ag: ModuleType, rag: PersianRAG) -> None:
    """the self-check decides: finish, retry, ask a human, or refuse"""
    agent = ag.AgenticRAG(rag, max_retries=1, faithfulness_threshold=0.7)
    answered = {"abstained": False}
    assert agent.route_after_check({"faithfulness": 0.8, "attempt": 1}) == "finish"
    assert agent.route_after_check({"faithfulness": 0.5, "attempt": 1}) == "rewrite"
    assert (
        agent.route_after_check({"faithfulness": None, "attempt": 2, "draft": answered}) == "refuse"
    )
    reviewed = ag.AgenticRAG(rag, max_retries=1, human_review=True)
    assert (
        reviewed.route_after_check({"faithfulness": 0.2, "attempt": 2, "draft": answered})
        == "human_review"
    )
    abstained = {"faithfulness": 0.0, "attempt": 2, "draft": {"abstained": True}}
    assert reviewed.route_after_check(abstained) == "refuse"


@hint(
    fa="زیر آستانهٔ BM25 بدون صدا زدن LLM هیچ متنی مرتبط نیست؛ وگرنه یک فراخوانی با GRADE_TEMPLATE و parse_grades.",
    en="Below the BM25 threshold nothing is relevant and the LLM is not called; else one GRADE_TEMPLATE call + parse_grades.",
)
def test_grade_node(ag: ModuleType, rag: PersianRAG) -> None:
    """grading keeps relevant passages with one LLM call, or none below the threshold"""
    agent = ag.AgenticRAG(rag)
    state: dict[str, Any] = {"question": PIK, "query": PIK, "sub_questions": [PIK]}
    state.update(agent.retrieve(state))
    state.update(agent.rerank(state))
    calls = agent.llm.calls
    update = agent.grade(state)
    assert agent.llm.calls == calls + 1 and update["trace"] == ["grade"]
    assert 1 <= len(update["relevant"]) <= rag.k
    assert "45" in update["relevant"][0]["chunk"]["text"]
    state["query"] = NO_ANSWER
    assert agent.grade(state)["relevant"] == [] and agent.llm.calls == calls + 1


# ------------------------------------------------------------------- runner


@hint(
    fa="حلقه: node را اجرا کنید، trace را اضافه و بقیه را جایگزین کنید، با edges گره بعدی را پیدا کنید تا END.",
    en="Loop: run the node, append trace and replace the rest, find the next node via edges until END.",
)
def test_run_simple_question(ag: ModuleType, rag: PersianRAG) -> None:
    """an easy question: analyze → … → finish with four LLM calls and a citation"""
    result = ag.AgenticRAG(rag).run(PIK)
    assert result.trace == ["analyze", "retrieve", "rerank", "grade", "generate", "check", "finish"]
    assert result.llm_calls == 4 and result.attempts == 1
    assert not result.answer.abstained and "45" in result.answer.text and result.answer.citations


@hint(
    fa="پرسش محاوره‌ای در analyze به فارسی معیار تبدیل می‌شود؛ query در نتیجه باید همان باشد.",
    en="analyze turns the colloquial question into standard Persian; the result's query must be that.",
)
def test_run_colloquial_and_multi_part(ag: ModuleType, rag: PersianRAG) -> None:
    """colloquial questions are rewritten, multi-part questions answered part by part"""
    agent = ag.AgenticRAG(rag)
    r = agent.run("هزینهٔ پیک تو تهران چقدره؟")
    assert r.query == "هزینهٔ پیک در تهران چقدر است؟" and not r.answer.abstained
    multi = agent.run("هزینهٔ پیک در تهران چقدر است و ساعت کاری کتابخانهٔ مهرگان چیست؟")
    assert len(multi.sub_questions) == 2 and multi.llm_calls == 5
    assert len(multi.answer.answer.splitlines()) == 2


@hint(
    fa="«اون چنده؟» → مسیر clarify با یک فراخوانی؛ پرسش بی‌پاسخ → حداکثر max_retries بازنویسی، بعد refuse.",
    en="'اون چنده؟' → clarify after one call; an unanswerable question → at most max_retries rewrites, then refuse.",
)
def test_run_clarify_and_bounded_retries(ag: ModuleType, rag: PersianRAG) -> None:
    """ambiguous questions get a clarifying question; the retry loop is bounded"""
    agent = ag.AgenticRAG(rag, max_retries=2)
    clarify = agent.run("اون چنده؟")
    assert clarify.trace == ["analyze", "clarify"] and clarify.answer.abstain_reason == "clarify"
    refused = agent.run(NO_ANSWER)
    assert refused.answer.abstained and refused.answer.abstain_reason == "not_relevant"
    assert refused.trace.count("rewrite") == 2 and refused.attempts == 3


@hint(
    fa="وقتی بررسی وفاداری ضعیف است و تلاش‌ها تمام شده، پاسخ باید «نمی‌دانم» با دلیل unfaithful باشد.",
    en="When the faithfulness check is weak and retries are used up, answer «نمی‌دانم» with reason unfaithful.",
)
def test_run_refuses_unfaithful_answers(ag: ModuleType, rag: PersianRAG) -> None:
    """a hallucinated answer never reaches the user"""
    result = ag.AgenticRAG(rag, hallucinating(rag), max_retries=1).run(PIK)
    assert result.answer.abstained and result.answer.abstain_reason == "unfaithful"
    assert result.trace.count("check") == 2


# ----------------------------------------------------------------- LangGraph


@hint(
    fa="همهٔ agent.nodes را add_node کنید، START → analyze، و هر یال را ساده یا شرطی اضافه کنید.",
    en="add_node every agent.nodes item, START → analyze, then each edge as plain or conditional.",
)
def test_graph_matches_runner(ag: ModuleType, rag: PersianRAG) -> None:
    """the LangGraph graph gives exactly the plain runner's answers and traces"""
    pytest.importorskip("langgraph", reason=EXTRA)
    agent = ag.AgenticRAG(rag)
    graph = ag.build_graph(agent)
    for q in (PIK, "اون چنده؟", NO_ANSWER):
        via_graph, plain = ag.run_graph(agent, graph, q), agent.run(q)
        assert via_graph.answer == plain.answer and via_graph.trace == plain.trace


@hint(
    fa="compile(checkpointer=checkpointer) را فراموش نکنید؛ interrupt بدون checkpointer کار نمی‌کند.",
    en="Don't forget compile(checkpointer=checkpointer); interrupt() needs a checkpointer.",
)
def test_graph_human_in_the_loop(ag: ModuleType, rag: PersianRAG) -> None:
    """a low-confidence answer pauses for a human, who can edit it"""
    pytest.importorskip("langgraph", reason=EXTRA)
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    agent = ag.AgenticRAG(rag, hallucinating(rag), max_retries=0, human_review=True)
    graph = ag.build_graph(agent, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "lesson11"}}
    paused = ag.run_graph(agent, graph, PIK, config)
    assert "__interrupt__" in paused
    fixed = "هزینهٔ پیک در تهران 45 هزار تومان است [1]."
    final = graph.invoke(Command(resume={"action": "edit", "answer": fixed}), config)
    assert not final["answer"]["abstained"] and final["answer"]["citations"]


# ------------------------------------------------------------- the numbers


@hint(
    fa="این عددها جدول «اندازه‌گیری» درس‌اند؛ اگر پایین‌ترند، آستانه یا مسیرها را با مرجع مقایسه کنید.",
    en="These are the lesson's measured numbers; if yours are lower, compare thresholds and routing with the reference.",
)
def test_your_agent_reproduces_the_lesson_numbers(ag: ModuleType, rag: PersianRAG) -> None:
    """your agent reproduces the lesson's table on both question sets"""
    from parsrag.eval.benchmark import load_fixture_benchmark
    from parsrag.eval.pipeline import evaluate_pipeline

    agent = ag.AgenticRAG(rag)
    original = evaluate_pipeline(agent, load_fixture_benchmark()[1], n_resamples=10)
    para = evaluate_pipeline(
        agent, load_fixture_benchmark(qa_file="qa_paraphrase.jsonl")[1], n_resamples=10
    )
    assert original.metrics["answer_f1"] == pytest.approx(0.314, abs=1e-3)
    assert original.metrics["abstention_f1"] == pytest.approx(0.714, abs=1e-3)
    assert para.metrics["retrieval_r@1"] == pytest.approx(0.478, abs=1e-3)
    assert para.metrics["answered_rate"] == pytest.approx(0.348, abs=1e-3)

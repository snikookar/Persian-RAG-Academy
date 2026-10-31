"""parsrag.agent: rules, prompts/parsers, the plain runner and the LangGraph graph."""

from __future__ import annotations

from typing import Any

import pytest

from parsrag import PersianRAG
from parsrag.agent import (
    AgenticRAG,
    OfflineAgentLLM,
    build_graph,
    colloquial_to_standard,
    is_ambiguous,
    keyword_query,
    parse_analysis,
    parse_grades,
    parse_rewrite,
    run_graph,
    split_questions,
    task_of,
)
from parsrag.agent.prompts import ANALYZE_SYSTEM, GRADE_SYSTEM
from parsrag.llm import FakeLLM, Message

PIK = "هزینهٔ پیک در تهران چقدر است؟"


@pytest.fixture(scope="module")
def rag() -> PersianRAG:
    return PersianRAG.course_default()


@pytest.fixture()
def agent(rag: PersianRAG) -> AgenticRAG:
    return AgenticRAG(rag)


def hallucinating(rag: PersianRAG) -> OfflineAgentLLM:
    """An offline model whose answers are never supported by the context."""
    return OfflineAgentLLM(
        vocabulary=rag.bm25._postings,
        overrides={"generate": lambda msgs: "هزینهٔ پیک در تهران ۹۹ هزار تومان است [1]."},
    )


# ------------------------------------------------------------------ rules


def test_colloquial_rules() -> None:
    assert colloquial_to_standard("اگه کتابو دیر بدم چی میشه؟") == "اگر کتابو دیر بدهم چی می‌شود؟"
    assert colloquial_to_standard("بسته‌م کی می‌رسه؟") == "بسته‌ام کی می‌رسد؟"
    assert colloquial_to_standard("امتیازا کی می‌سوزن؟") == "امتیازا کی می‌سوزند؟"
    assert colloquial_to_standard("سابقه تا کی می‌مونه؟") == "سابقه تا کی می‌ماند؟"
    assert colloquial_to_standard("به آلمان می‌فرستین؟") == "به آلمان می‌فرستید؟"
    assert colloquial_to_standard("خب سفارشمو کنسل کنم؟") == "سفارشم را کنسل کنم؟"
    assert colloquial_to_standard(PIK) == PIK


def test_split_questions() -> None:
    assert split_questions("هزینهٔ پیک چقدر است و مرجوعی تا کی ممکن است؟") == [
        "هزینهٔ پیک چقدر است؟",
        "مرجوعی تا کی ممکن است؟",
    ]
    assert split_questions("ساعت کاری چیست؟ جریمه چقدر است") == [
        "ساعت کاری چیست؟",
        "جریمه چقدر است؟",
    ]
    assert split_questions("قوانین ارسال و مرجوعی چیست؟") == ["قوانین ارسال و مرجوعی چیست؟"]


def test_ambiguity_and_keywords() -> None:
    assert is_ambiguous("اون چنده؟") and is_ambiguous("این چیست؟")
    assert not is_ambiguous(PIK)
    assert keyword_query(PIK) == "هزینه پیک تهران"
    assert keyword_query("کتاب‌های کودک کجاست؟", split_zwnj=True) == "کتاب های کودک"


# ---------------------------------------------------------------- parsers


def test_parse_analysis() -> None:
    a = parse_analysis(
        "مسیر: اسناد\nبازنویسی: هزینهٔ پیک چقدر است؟\nزیرپرسش: الف؟\nزیرپرسش: ب؟", "x"
    )
    assert (a.route, a.query, a.sub_questions) == (
        "documents",
        "هزینهٔ پیک چقدر است؟",
        ["الف؟", "ب؟"],
    )
    b = parse_analysis("مسیر: «ابهام»", "اون چنده؟")
    assert b.route == "clarify" and b.query == "اون چنده؟" and b.clarification
    assert parse_analysis("route: out_of_scope", "q").route == "out_of_scope"
    assert parse_analysis("بی‌ربط", "q").route == "documents"  # safe default


def test_parse_grades_and_rewrite() -> None:
    assert parse_grades("[1] بله\n[۲]: خیر\n[3] yes\n[9] بله", 4) == [True, False, True, False]
    assert parse_rewrite("پرسش: هزینهٔ ارسال\nتوضیح", "fallback") == "هزینهٔ ارسال"
    assert parse_rewrite("  \n", "fallback") == "fallback"


def test_task_of() -> None:
    assert task_of([Message(role="system", content=ANALYZE_SYSTEM)]) == "analyze"
    assert task_of([{"role": "system", "content": GRADE_SYSTEM}]) == "grade"
    assert task_of([{"role": "user", "content": "x"}]) == "generate"


# ------------------------------------------------------------- the runner


def test_answers_with_citations_in_four_calls(agent: AgenticRAG) -> None:
    result = agent.run(PIK)
    assert not result.answer.abstained and "45" in result.answer.text
    assert result.answer.citations
    assert result.trace == ["analyze", "retrieve", "rerank", "grade", "generate", "check", "finish"]
    assert result.llm_calls == 4  # analyze, grade, generate, check
    assert result.faithfulness == 1.0 and result.attempts == 1


def test_colloquial_question_is_rewritten(agent: AgenticRAG) -> None:
    result = agent.run("هزینهٔ پیک تو تهران چقدره؟")
    assert result.query == "هزینهٔ پیک در تهران چقدر است؟"
    assert not result.answer.abstained


def test_multi_part_question(agent: AgenticRAG) -> None:
    result = agent.run("هزینهٔ پیک در تهران چقدر است و ساعت کاری کتابخانهٔ سپیدار چیست؟")
    assert len(result.sub_questions) == 2
    assert len(result.answer.answer.splitlines()) == 2
    assert result.llm_calls == 5  # one generation per sub-question


def test_clarify_route(agent: AgenticRAG) -> None:
    result = agent.run("اون چنده؟")
    assert result.route == "clarify" and result.trace == ["analyze", "clarify"]
    assert result.answer.abstained and result.answer.abstain_reason == "clarify"
    assert "دقیق‌تر" in result.answer.answer and result.llm_calls == 1


def test_out_of_scope_route(rag: PersianRAG) -> None:
    agent = AgenticRAG(rag, OfflineAgentLLM(vocabulary={"پیک"}))
    result = agent.run("بهترین تیم فوتبال جهان؟")
    assert result.route == "out_of_scope" and result.answer.abstain_reason == "out_of_scope"


def test_unanswerable_retries_then_refuses(agent: AgenticRAG) -> None:
    result = agent.run("قیمت بلیت هواپیما به استانبول چقدر است؟")
    assert result.answer.abstained and result.answer.abstain_reason == "not_relevant"
    assert result.attempts == agent.max_retries + 1
    assert result.trace.count("rewrite") == agent.max_retries


def test_unfaithful_answers_are_refused(rag: PersianRAG) -> None:
    agent = AgenticRAG(rag, hallucinating(rag), max_retries=1)
    result = agent.run(PIK)
    assert result.answer.abstained and result.answer.abstain_reason == "unfaithful"
    assert result.faithfulness == 0.0 and result.trace.count("check") == 2


def test_callable_reviewer(rag: PersianRAG) -> None:
    seen: list[dict[str, Any]] = []

    def reviewer(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return {"action": "edit", "answer": "هزینهٔ پیک در تهران 45 هزار تومان است [1]."}

    agent = AgenticRAG(rag, hallucinating(rag), max_retries=0, human_review=reviewer)
    result = agent.run(PIK)
    assert seen and seen[0]["draft"].startswith("هزینهٔ پیک") and seen[0]["sources"]
    assert result.trace[-1] == "human_review" and not result.answer.abstained
    assert "45" in result.answer.text and result.answer.citations


def test_agent_works_with_any_llm(rag: PersianRAG) -> None:
    """A real model is just another LLM: here one that always says «نمی‌دانم»."""
    llm = FakeLLM(handler=lambda msgs: "نمی‌دانم")
    result = AgenticRAG(rag, llm, max_retries=0).run(PIK)
    # The reply to the grading prompt has no «[n] بله» line: no passage counts as relevant.
    assert result.answer.abstained and result.answer.abstain_reason == "not_relevant"
    assert result.llm_calls == 2  # analyze, grade


def test_agent_retriever_and_evaluator(agent: AgenticRAG) -> None:
    from parsrag.eval.benchmark import load_fixture_benchmark
    from parsrag.eval.pipeline import evaluate_pipeline

    assert agent.retriever.search(PIK, 3)[0].chunk.doc_id == "kahroba-shipping"
    assert agent.retriever.search("اون چنده؟", 3) == []
    qa = load_fixture_benchmark()[1][:8]
    report = evaluate_pipeline(agent, qa, n_resamples=50)
    assert report.n_questions == 8 and report.metrics["retrieval_r@1"] > 0.5


def test_validation(rag: PersianRAG) -> None:
    with pytest.raises(ValueError):
        AgenticRAG(rag, max_retries=-1)


# ------------------------------------------------------------- LangGraph


def test_graph_matches_plain_runner(rag: PersianRAG) -> None:
    pytest.importorskip("langgraph", reason="needs the agent extra")
    agent = AgenticRAG(rag)
    graph = build_graph(agent)
    for q in (PIK, "اون چنده؟", "قیمت بلیت هواپیما به استانبول چقدر است؟"):
        via_graph = run_graph(agent, graph, q)
        plain = agent.run(q)
        assert not isinstance(via_graph, dict)
        assert via_graph.answer == plain.answer and via_graph.trace == plain.trace
        assert via_graph.llm_calls == plain.llm_calls


def test_graph_interrupt_and_resume(rag: PersianRAG) -> None:
    pytest.importorskip("langgraph", reason="needs the agent extra")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    agent = AgenticRAG(rag, hallucinating(rag), max_retries=0, human_review=True)
    graph = build_graph(agent, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    paused = run_graph(agent, graph, PIK, config)
    assert isinstance(paused, dict) and "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["question"] == PIK and payload["faithfulness"] == 0.0
    final = graph.invoke(Command(resume={"action": "reject"}), config)
    assert final["answer"]["abstained"] and final["trace"][-1] == "human_review"

    config2 = {"configurable": {"thread_id": "t2"}}
    run_graph(agent, graph, PIK, config2)
    approved = graph.invoke(Command(resume={"action": "approve"}), config2)
    assert not approved["answer"]["abstained"]
    assert graph.get_state(config2).values["answer"] == approved["answer"]


def test_graph_mermaid(rag: PersianRAG) -> None:
    pytest.importorskip("langgraph", reason="needs the agent extra")
    from parsrag.agent import draw_mermaid

    text = draw_mermaid(AgenticRAG(rag))
    for node in ("analyze", "grade", "rewrite", "human_review"):
        assert node in text
    assert "rewrite --> retrieve" in text


def test_build_graph_without_langgraph(rag: PersianRAG, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "langgraph.graph", None)
    with pytest.raises(ImportError, match=r"parsrag\[agent\]"):
        build_graph(AgenticRAG(rag))

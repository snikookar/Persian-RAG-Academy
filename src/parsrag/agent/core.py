"""Agentic RAG: nodes, routers and a plain-Python runner (lesson 11).

The agent is a small state machine::

    analyze ─┬─ documents ──> retrieve → rerank → grade ─┬─ relevant ──> generate → check ─┬─ faithful ─> finish
             ├─ clarify ────> clarify                     ├─ none, retry ─> rewrite ↺       ├─ weak, retry ─> rewrite ↺
             └─ out_of_scope > refuse                     └─ none, done ──> refuse          ├─ weak, review ─> human_review
                                                                                            └─ weak, done ──> refuse

Every node is a method that takes the state (a ``dict``) and returns the keys
it changes, and every branch is a router method that returns the name of
the next node. :meth:`AgenticRAG.run` walks this graph in plain Python (no
extra needed); :func:`parsrag.agent.build_graph` compiles *the same* nodes
and routers into a LangGraph ``StateGraph`` (``pip install 'parsrag[agent]'``)
for checkpoints, human-in-the-loop interrupts and visualisation.

State values are plain JSON types (search results and answers are stored
with ``model_dump()``) so any LangGraph checkpointer can save them.
"""  # noqa: E501

from __future__ import annotations

import operator
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from parsrag.agent.offline import OfflineAgentLLM
from parsrag.agent.prompts import (
    ANALYZE_SYSTEM,
    ANALYZE_TEMPLATE,
    GRADE_SYSTEM,
    GRADE_TEMPLATE,
    REWRITE_SYSTEM,
    REWRITE_TEMPLATE,
    grade_context,
    parse_analysis,
    parse_grades,
    parse_rewrite,
)
from parsrag.eval.generation import LLMJudge
from parsrag.generate import REFUSAL, GroundedAnswer, chunk_context, retriever_confidence
from parsrag.llm import LLM, CountingLLM, Message
from parsrag.rag import PersianRAG
from parsrag.retrieval import reciprocal_rank_fusion
from parsrag.types import SearchResult

END = "__end__"

#: A human reviewer: gets the low-confidence draft, returns
#: ``{"action": "approve"}``, ``{"action": "edit", "answer": "..."}`` or ``{"action": "reject"}``.
Reviewer = Callable[[dict[str, Any]], Mapping[str, Any]]


class AgentState(TypedDict, total=False):
    question: str
    route: str
    query: str
    sub_questions: list[str]
    clarification: str | None
    queries: list[str]  # every query tried, for the rewrite prompt
    attempt: int  # retrieval rounds done
    candidates: list[list[dict[str, Any]]]  # per sub-question, after hybrid retrieval
    results: list[dict[str, Any]]  # after reranking (what gets graded)
    relevant: list[dict[str, Any]]  # after grading (what the generator sees)
    draft: dict[str, Any] | None
    faithfulness: float | None
    answer: dict[str, Any] | None
    trace: Annotated[list[str], operator.add]  # visited nodes (appended, not replaced)


class AgentResult(BaseModel):
    """What :meth:`AgenticRAG.run` returns: the answer plus how the agent got there."""

    answer: GroundedAnswer
    route: str = "documents"
    query: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    attempts: int = 0
    faithfulness: float | None = None
    trace: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    latency_ms: float = 0.0


def _results(dumps: Sequence[dict[str, Any]]) -> list[SearchResult]:
    return [SearchResult.model_validate(d) for d in dumps]


def _dumps(results: Sequence[SearchResult]) -> list[dict[str, Any]]:
    return [r.model_dump() for r in results]


class AgenticRAG:
    """Analyse → route → retrieve → rerank → grade → generate → self-check, with bounded retries.

    ``llm`` answers every step (analyse, grade, rewrite, generate, check);
    the default :class:`~parsrag.agent.OfflineAgentLLM` does so with
    transparent rules. ``max_retries`` bounds the rewrite loop.
    ``use_threshold`` keeps lesson 09's calibrated BM25 abstention (checked
    on the analysed query, before the grading call). ``human_review``:
    ``True`` pauses a LangGraph run with ``interrupt()`` for low-confidence
    answers; a :data:`Reviewer` callable is asked directly (also in :meth:`run`).
    """

    def __init__(
        self,
        rag: PersianRAG,
        llm: LLM | None = None,
        *,
        max_retries: int = 2,
        grade_k: int = 6,
        faithfulness_threshold: float = 0.7,
        use_threshold: bool = True,
        human_review: bool | Reviewer = False,
        scope: str | None = None,
    ) -> None:
        if max_retries < 0 or grade_k <= 0:
            raise ValueError("need max_retries >= 0 and grade_k > 0")
        self.rag = rag
        base = llm if llm is not None else OfflineAgentLLM(vocabulary=rag.bm25._postings)
        self.llm = CountingLLM(base)
        self.max_retries = max_retries
        self.grade_k = grade_k
        self.faithfulness_threshold = faithfulness_threshold
        self.use_threshold = use_threshold
        self.review = human_review
        self.scope = scope or "، ".join(
            dict.fromkeys(d.title or d.id for d in rag.documents.values())
        )
        self.judge = LLMJudge(self.llm)

    def _ask(self, system: str, user: str) -> str:
        return self.llm.generate(
            [Message(role="system", content=system), Message(role="user", content=user)]
        )

    # ------------------------------------------------------------------ nodes

    def analyze(self, state: AgentState) -> dict[str, Any]:
        reply = self._ask(
            ANALYZE_SYSTEM, ANALYZE_TEMPLATE.format(scope=self.scope, question=state["question"])
        )
        a = parse_analysis(reply, state["question"])
        return {
            "route": a.route,
            "query": a.query,
            "sub_questions": a.sub_questions,
            "clarification": a.clarification,
            "queries": [a.query],
            "attempt": 0,
            "trace": ["analyze"],
        }

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        hybrid = self.rag.hybrid
        fetch = max(self.rag.fetch_k, self.grade_k)
        candidates = [_dumps(hybrid.search(q, fetch)) for q in state["sub_questions"]]
        attempt = state.get("attempt", 0) + 1
        return {"candidates": candidates, "attempt": attempt, "trace": ["retrieve"]}

    def rerank(self, state: AgentState) -> dict[str, Any]:
        reranker = self.rag.reranker
        lists = []
        for q, cands in zip(state["sub_questions"], state["candidates"], strict=True):
            results = _results(cands)
            ranked = reranker.rerank(q, results, top_k=self.grade_k) if reranker else results
            lists.append(ranked[: self.grade_k])
        merged = lists[0] if len(lists) == 1 else reciprocal_rank_fusion(lists)
        return {"results": _dumps(merged[: self.grade_k * len(lists)]), "trace": ["rerank"]}

    def grade(self, state: AgentState) -> dict[str, Any]:
        results = _results(state["results"])
        if self.use_threshold and self.rag.min_score is not None:
            confidence = retriever_confidence(self.rag.bm25)(state["query"], results)
            if confidence < self.rag.min_score:
                return {"relevant": [], "trace": ["grade"]}  # no LLM call needed
        if not results:
            return {"relevant": [], "trace": ["grade"]}
        reply = self._ask(
            GRADE_SYSTEM,
            GRADE_TEMPLATE.format(question=state["query"], context=grade_context(results)),
        )
        grades = parse_grades(reply, len(results))
        keep = [r for r, ok in zip(results, grades, strict=True) if ok]
        limit = self.rag.k * len(state["sub_questions"])
        return {"relevant": _dumps(keep[:limit]), "trace": ["grade"]}

    def generate(self, state: AgentState) -> dict[str, Any]:
        gen = self.rag.generator
        gen.llm = self.llm
        sources = _results(state["relevant"])
        parts = [
            gen.finalize(q, sources, self.llm.generate(gen.messages(q, sources)), None)
            for q in state["sub_questions"]
        ]
        answered = [p for p in parts if not p.abstained]
        if not answered:
            draft = parts[0].model_copy(update={"question": state["question"]})
        else:
            citations = {c.number: c for p in answered for c in p.citations}
            draft = GroundedAnswer(
                question=state["question"],
                answer="\n".join(p.answer for p in answered),
                citations=[citations[n] for n in sorted(citations)],
                uncited=any(p.uncited for p in answered),
                invalid_citations=sorted({n for p in answered for n in p.invalid_citations}),
                sources=sources,
                raw="\n".join(p.raw for p in parts),
            )
        return {"draft": draft.model_dump(), "trace": ["generate"]}

    def check(self, state: AgentState) -> dict[str, Any]:
        draft = GroundedAnswer.model_validate(state["draft"])
        if draft.abstained:
            return {"faithfulness": 0.0, "trace": ["check"]}
        contexts = [chunk_context(r.chunk) for r in draft.sources]
        verdict = self.judge.faithfulness(state["query"], draft.text, contexts)
        return {"faithfulness": verdict.score, "trace": ["check"]}

    def rewrite(self, state: AgentState) -> dict[str, Any]:
        previous = state.get("queries", [])
        reply = self._ask(
            REWRITE_SYSTEM,
            REWRITE_TEMPLATE.format(
                question=state["question"], previous="\n".join(f"- {q}" for q in previous)
            ),
        )
        query = parse_rewrite(reply, state["query"])
        return {
            "query": query,
            "sub_questions": [query],
            "queries": [*previous, query],
            "trace": ["rewrite"],
        }

    def finish(self, state: AgentState) -> dict[str, Any]:
        return {"answer": state["draft"], "trace": ["finish"]}

    def clarify(self, state: AgentState) -> dict[str, Any]:
        answer = GroundedAnswer(
            question=state["question"],
            answer=state.get("clarification") or REFUSAL,
            abstained=True,
            abstain_reason="clarify",
        )
        return {"answer": answer.model_dump(), "trace": ["clarify"]}

    def refuse(self, state: AgentState) -> dict[str, Any]:
        if state.get("route") == "out_of_scope":
            reason = "out_of_scope"
        elif (draft := state.get("draft")) and not draft["abstained"]:
            reason = "unfaithful"
        elif draft:
            reason = "model"
        else:
            reason = "not_relevant"
        answer = GroundedAnswer(
            question=state["question"],
            answer=REFUSAL,
            abstained=True,
            abstain_reason=reason,
            sources=_results(state.get("relevant") or state.get("results") or []),
        )
        return {"answer": answer.model_dump(), "trace": ["refuse"]}

    def review_payload(self, state: AgentState) -> dict[str, Any]:
        draft = GroundedAnswer.model_validate(state["draft"])
        return {
            "question": state["question"],
            "draft": draft.answer,
            "faithfulness": state.get("faithfulness"),
            "sources": [
                {"number": i, "doc_id": r.chunk.doc_id, "text": chunk_context(r.chunk)}
                for i, r in enumerate(draft.sources, 1)
            ],
        }

    def apply_review(self, state: AgentState, decision: Mapping[str, Any]) -> dict[str, Any]:
        """Turn a reviewer's decision into the final answer."""
        action = str(decision.get("action", "reject"))
        draft = GroundedAnswer.model_validate(state["draft"])
        if action == "approve":
            return {"answer": draft.model_dump(), "trace": ["human_review"]}
        if action == "edit" and decision.get("answer"):
            edited = self.rag.generator.finalize(
                state["question"], draft.sources, str(decision["answer"]), None
            )
            return {"answer": edited.model_dump(), "trace": ["human_review"]}
        refused = self.refuse(state)
        return {"answer": refused["answer"], "trace": ["human_review"]}

    def human_review(self, state: AgentState) -> dict[str, Any]:
        payload = self.review_payload(state)
        if callable(self.review):
            return self.apply_review(state, self.review(payload))
        from langgraph.types import interrupt  # only reached inside a LangGraph run

        return self.apply_review(state, interrupt(payload))

    # ---------------------------------------------------------------- routers

    def route_after_analyze(self, state: AgentState) -> str:
        return {"clarify": "clarify", "out_of_scope": "refuse"}.get(state["route"], "retrieve")

    def can_retry(self, state: AgentState) -> bool:
        return state.get("attempt", 0) <= self.max_retries

    def route_after_grade(self, state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        return "rewrite" if self.can_retry(state) else "refuse"

    def route_after_check(self, state: AgentState) -> str:
        score = state.get("faithfulness")
        if score is not None and score >= self.faithfulness_threshold:
            return "finish"
        if self.can_retry(state):
            return "rewrite"
        draft = state.get("draft") or {}
        if self.review and not draft.get("abstained", True):
            return "human_review"
        return "refuse"

    # -------------------------------------------------------------- the graph

    @property
    def nodes(self) -> dict[str, Callable[[AgentState], dict[str, Any]]]:
        return {
            "analyze": self.analyze,
            "retrieve": self.retrieve,
            "rerank": self.rerank,
            "grade": self.grade,
            "generate": self.generate,
            "check": self.check,
            "rewrite": self.rewrite,
            "finish": self.finish,
            "clarify": self.clarify,
            "refuse": self.refuse,
            "human_review": self.human_review,
        }

    @property
    def edges(self) -> dict[str, str | tuple[Callable[[AgentState], str], list[str]]]:
        """``node → next node`` or ``node → (router, possible next nodes)``."""
        return {
            "analyze": (self.route_after_analyze, ["retrieve", "clarify", "refuse"]),
            "retrieve": "rerank",
            "rerank": "grade",
            "grade": (self.route_after_grade, ["generate", "rewrite", "refuse"]),
            "generate": "check",
            "check": (self.route_after_check, ["finish", "rewrite", "human_review", "refuse"]),
            "rewrite": "retrieve",
            "finish": END,
            "clarify": END,
            "refuse": END,
            "human_review": END,
        }

    def run(self, question: str, *, max_steps: int = 50) -> AgentResult:
        """Walk the graph in plain Python (no LangGraph needed)."""
        start, calls = time.perf_counter(), self.llm.calls
        state: AgentState = {"question": question, "trace": []}
        node = "analyze"
        for _ in range(max_steps):
            update = self.nodes[node](state)
            for key, value in update.items():
                if key == "trace":
                    state["trace"] = [*state.get("trace", []), *value]
                else:
                    state[key] = value  # type: ignore[literal-required]
            edge = self.edges[node]
            node = edge if isinstance(edge, str) else edge[0](state)
            if node == END:
                break
        else:  # pragma: no cover - the routers always terminate
            raise RuntimeError(f"agent did not finish in {max_steps} steps")
        return self.result(state, self.llm.calls - calls, time.perf_counter() - start)

    def result(self, state: Mapping[str, Any], llm_calls: int, seconds: float) -> AgentResult:
        return AgentResult(
            answer=GroundedAnswer.model_validate(state["answer"]),
            route=state.get("route", "documents"),
            query=state.get("query", ""),
            sub_questions=state.get("sub_questions", []),
            attempts=state.get("attempt", 0),
            faithfulness=state.get("faithfulness"),
            trace=list(state.get("trace", [])),
            llm_calls=llm_calls,
            latency_ms=seconds * 1000,
        )

    # ------------------------------------------- lesson 10 evaluator interface

    def answer(self, question: str) -> GroundedAnswer:
        return self.run(question).answer

    @property
    def retriever(self) -> AgentRetriever:
        return AgentRetriever(self)


class AgentRetriever:
    """Retrieval as the agent sees it on its first try: analyse, hybrid search, rerank.

    Lets :func:`parsrag.eval.pipeline.evaluate_pipeline` measure the effect
    of question analysis on retrieval (R@k, MRR) separately from answers.
    """

    def __init__(self, agent: AgenticRAG) -> None:
        self.agent = agent

    def search(self, query: str, k: int = 4, *, filter: Any = None) -> list[SearchResult]:
        state: AgentState = {"question": query}
        state.update(self.agent.analyze(state))  # type: ignore[typeddict-item]
        if state["route"] != "documents":
            return []
        state.update(self.agent.retrieve(state))  # type: ignore[typeddict-item]
        reranker = self.agent.rag.reranker
        lists = []
        for q, cands in zip(state["sub_questions"], state["candidates"], strict=True):
            results = _results(cands)
            lists.append(reranker.rerank(q, results, top_k=k) if reranker else results[:k])
        merged = lists[0] if len(lists) == 1 else reciprocal_rank_fusion(lists)
        return merged[:k]


__all__ = ["END", "AgentResult", "AgentRetriever", "AgentState", "AgenticRAG", "Reviewer"]

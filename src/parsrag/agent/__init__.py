"""Agentic RAG (lesson 11): question analysis, routing, grading, self-checks and retries.

>>> from parsrag import PersianRAG
>>> from parsrag.agent import AgenticRAG
>>> agent = AgenticRAG(PersianRAG.course_default())      # doctest: +SKIP
>>> result = agent.run("کرایهٔ پیک داخل تهران چنده؟")      # doctest: +SKIP
>>> result.answer.text, result.trace, result.llm_calls    # doctest: +SKIP

The plain-Python runner (:meth:`AgenticRAG.run`) needs nothing beyond the
core install; :func:`build_graph` turns the same nodes into a LangGraph
``StateGraph`` (``pip install 'parsrag[agent]'``).
"""

from parsrag.agent.core import END, AgenticRAG, AgentResult, AgentRetriever, AgentState, Reviewer
from parsrag.agent.graph import build_graph, draw_mermaid, run_graph
from parsrag.agent.offline import OfflineAgentLLM, task_of
from parsrag.agent.prompts import (
    ANALYZE_SYSTEM,
    ANALYZE_TEMPLATE,
    GRADE_SYSTEM,
    GRADE_TEMPLATE,
    REWRITE_SYSTEM,
    REWRITE_TEMPLATE,
    Analysis,
    Route,
    parse_analysis,
    parse_grades,
    parse_rewrite,
)
from parsrag.agent.rewrite import (
    colloquial_to_standard,
    content_words,
    is_ambiguous,
    keyword_query,
    split_questions,
)

__all__ = [
    "ANALYZE_SYSTEM",
    "ANALYZE_TEMPLATE",
    "END",
    "GRADE_SYSTEM",
    "GRADE_TEMPLATE",
    "REWRITE_SYSTEM",
    "REWRITE_TEMPLATE",
    "AgentResult",
    "AgentRetriever",
    "AgentState",
    "AgenticRAG",
    "Analysis",
    "OfflineAgentLLM",
    "Reviewer",
    "Route",
    "build_graph",
    "colloquial_to_standard",
    "content_words",
    "draw_mermaid",
    "is_ambiguous",
    "keyword_query",
    "parse_analysis",
    "parse_grades",
    "parse_rewrite",
    "run_graph",
    "split_questions",
    "task_of",
]

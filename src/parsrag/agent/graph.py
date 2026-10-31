"""The agent as a LangGraph ``StateGraph`` (lesson 11; ``pip install 'parsrag[agent]'``).

:func:`build_graph` adds every node and edge of :class:`~parsrag.agent.AgenticRAG`
to a ``StateGraph`` — the same functions :meth:`AgenticRAG.run` walks in plain
Python — and compiles it, optionally with a checkpointer (needed for
``interrupt()``-based human review and for resuming a conversation)::

    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(agent, checkpointer=InMemorySaver())
    state = graph.invoke({"question": "..."}, {"configurable": {"thread_id": "1"}})

LangGraph is imported inside the functions, so ``import parsrag.agent`` works
on the core install.
"""

from __future__ import annotations

from typing import Any

from parsrag.agent.core import END, AgenticRAG, AgentResult, AgentState


def _import_langgraph() -> Any:
    try:
        import langgraph.graph as lg
    except ImportError as exc:
        raise ImportError(
            "build_graph needs LangGraph. Install it with: pip install 'parsrag[agent]'"
        ) from exc
    return lg


def build_graph(agent: AgenticRAG, *, checkpointer: Any | None = None) -> Any:
    """Compile ``agent``'s nodes and routers into a LangGraph graph."""
    lg = _import_langgraph()
    graph = lg.StateGraph(AgentState)
    for name, fn in agent.nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(lg.START, "analyze")
    for name, edge in agent.edges.items():
        if isinstance(edge, str):
            graph.add_edge(name, lg.END if edge == END else edge)
        else:
            router, targets = edge
            graph.add_conditional_edges(name, router, {t: t for t in targets})
    return graph.compile(checkpointer=checkpointer)


def run_graph(
    agent: AgenticRAG, graph: Any, question: str, config: dict[str, Any] | None = None
) -> AgentResult | dict[str, Any]:
    """Invoke ``graph`` and return an :class:`AgentResult`, like :meth:`AgenticRAG.run`.

    If the run stopped at a human-review ``interrupt()``, the raw state is
    returned instead (it has an ``"__interrupt__"`` key); resume it with
    ``graph.invoke(Command(resume={"action": "approve"}), config)``.
    """
    import time

    start, calls = time.perf_counter(), agent.llm.calls
    state = graph.invoke({"question": question, "trace": []}, config)
    if "__interrupt__" in state:
        return dict(state)
    return agent.result(state, agent.llm.calls - calls, time.perf_counter() - start)


def draw_mermaid(agent: AgenticRAG) -> str:
    """LangGraph's own Mermaid drawing of the compiled graph."""
    return str(build_graph(agent).get_graph().draw_mermaid())


__all__ = ["build_graph", "draw_mermaid", "run_graph"]

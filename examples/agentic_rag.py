"""Agentic RAG vs. the lesson 09 pipeline: quality, LLM calls and latency (lesson 11).

    python examples/agentic_rag.py                       # offline: OfflineAgentLLM + FakeLLM
    python examples/agentic_rag.py --mermaid docs/images/agent_graph.mmd   # needs [agent]

With a real OpenAI-compatible model, on your own machine (never in CI):

    pip install -e ".[openai]"
    python examples/agentic_rag.py --llm qwen2.5:7b --base-url http://localhost:11434/v1

Both systems use the same index, retriever, reranker and abstention
threshold (calibrated on qa.jsonl). The agent adds question analysis,
passage grading, a faithfulness self-check and up to two rewrites.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence
from typing import Any

from parsrag import PersianRAG
from parsrag.agent import AgenticRAG, OfflineAgentLLM
from parsrag.eval.benchmark import load_fixture_benchmark
from parsrag.eval.dataset import QAItem
from parsrag.eval.pipeline import PipelineReport, evaluate_pipeline
from parsrag.eval.stats import paired_bootstrap
from parsrag.llm import LLM, CountingLLM

SETS = {"original": "qa.jsonl", "paraphrase": "qa_paraphrase.jsonl"}
METRICS = (
    ("retrieval_r@1", "retrieval R@1"),
    ("retrieval_mrr", "retrieval MRR@10"),
    ("answer_f1", "answer F1"),
    ("citation_precision", "citation precision"),
    ("abstention_f1", "abstention F1"),
    ("answered_rate", "answered rate"),
)


def cost(system: Any, counter: CountingLLM, qa: Sequence[QAItem]) -> dict[str, float]:
    """Mean LLM calls and median latency per question (one pass over ``qa``)."""
    calls, ms, retries = [], [], 0
    for item in qa:
        before = counter.calls
        start = time.perf_counter()
        if isinstance(system, AgenticRAG):
            result = system.run(item.question)
            retries += result.attempts > 1
        else:
            system.answer(item.question)
        ms.append((time.perf_counter() - start) * 1000)
        calls.append(counter.calls - before)
    return {
        "calls": statistics.mean(calls),
        "max_calls": max(calls),
        "ms": statistics.median(ms),
        "retried": retries / len(qa),
    }


def per_question(report: PipelineReport) -> list[float]:
    return [r.f1 or 0.0 for r in report.records if r.answerable]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", help="OpenAI-compatible model name (default: offline)")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--mermaid", help="write LangGraph's Mermaid drawing of the graph here")
    args = parser.parse_args()

    rag = PersianRAG.course_default()
    vocabulary = rag.bm25._postings
    if args.llm:
        from parsrag.llm import OpenAICompatibleLLM

        real: LLM = OpenAICompatibleLLM(args.llm, base_url=args.base_url)
        rag.llm = real
        systems = {"agent": AgenticRAG(rag, real)}
    else:
        systems = {
            "agent": AgenticRAG(rag, OfflineAgentLLM(vocabulary=vocabulary)),
            "agent, no colloquial rules": AgenticRAG(
                rag, OfflineAgentLLM(vocabulary=vocabulary, colloquial=False)
            ),
        }
    baseline_llm = CountingLLM(rag.llm)
    rag.llm = baseline_llm
    baseline = rag.generator

    for name, file in SETS.items():
        qa = load_fixture_benchmark(qa_file=file)[1]
        base_report = evaluate_pipeline(baseline, qa, name="lesson 09", n_resamples=1000)
        reports = {"lesson 09": base_report}
        costs = {"lesson 09": cost(baseline, baseline_llm, qa)}
        for label, agent in systems.items():
            reports[label] = evaluate_pipeline(agent, qa, name=label, n_resamples=1000)
            costs[label] = cost(agent, agent.llm, qa)

        print(f"\n### {name} ({len(qa)} questions)\n")
        print("| metric | " + " | ".join(reports) + " |")
        print("|---|" + "---|" * len(reports))
        for key, label in METRICS:
            cells = [f"{r.metrics.get(key, 0.0):.3f}" for r in reports.values()]
            print(f"| {label} | " + " | ".join(cells) + " |")
        for key, label, fmt in (
            ("calls", "LLM calls / question (mean)", "{:.2f}"),
            ("max_calls", "LLM calls / question (max)", "{:.0f}"),
            ("retried", "questions with a rewrite", "{:.0%}"),
            ("ms", "latency / question (median, ms)", "{:.1f}"),
        ):
            cells = [fmt.format(c[key]) for c in costs.values()]
            print(f"| {label} | " + " | ".join(cells) + " |")

        agent_report = reports["agent"]
        delta = paired_bootstrap(per_question(base_report), per_question(agent_report))
        print(f"\nagent − lesson 09, answer F1 (paired bootstrap, 95%): {delta}")
        routes: dict[str, int] = {}
        for item in qa:
            route = systems["agent"].run(item.question).route
            routes[route] = routes.get(route, 0) + 1
        print("agent routes:", ", ".join(f"{k}={v}" for k, v in sorted(routes.items())))

    if args.mermaid:
        from pathlib import Path

        from parsrag.agent import draw_mermaid

        Path(args.mermaid).write_text(draw_mermaid(systems["agent"]) + "\n", encoding="utf-8")
        print(f"\nMermaid drawing of the graph written to {args.mermaid}")


if __name__ == "__main__":
    main()

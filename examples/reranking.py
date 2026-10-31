"""Retrieve many, rerank few: quality, latency and cost (lesson 08).

    python examples/reranking.py                       # offline tables
    python examples/reranking.py --chart               # also save docs/images/rerank_r1.png

On your own machine (downloads a model / calls an API — never in CI):

    pip install -e ".[st]"
    python examples/reranking.py --cross-encoder BAAI/bge-reranker-v2-m3
    pip install -e ".[openai]"
    python examples/reranking.py --llm gpt-4o-mini                                    # OPENAI_API_KEY
    python examples/reranking.py --llm qwen2.5:7b --base-url http://localhost:11434/v1  # Ollama

The first stage is the lesson 07 default: RRF over BM25 (light stemming) and
HashingEmbedding. Tuning happens on qa.jsonl only; qa_paraphrase.jsonl is the
held-out check.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

from parsrag import FakeLLM, HashingEmbedding
from parsrag.chunking import estimate_tokens
from parsrag.eval import evaluate_retrieval
from parsrag.eval.benchmark import index_documents, load_fixture_benchmark
from parsrag.eval.dataset import QAItem
from parsrag.rerank import LexicalReranker, LLMReranker, Reranker, RerankingRetriever
from parsrag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, PersianTokenizer

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "docs" / "images" / "rerank_r1.png"
DEPTH = 10  # R@1, R@3 and MRR@10


def first_stage() -> tuple[HybridRetriever, BM25Retriever]:
    docs, _ = load_fixture_benchmark()
    store = index_documents(docs, HashingEmbedding())
    bm25 = BM25Retriever(store.chunks, tokenizer=PersianTokenizer(stem=True))
    return HybridRetriever([bm25, DenseRetriever(store)]), bm25


def scores(qa: list[QAItem], retriever: object) -> tuple[float, float, float, float]:
    """R@1, R@3, MRR@10 and milliseconds per question."""
    start = time.perf_counter()
    rep = evaluate_retrieval(
        qa,
        lambda q, k: retriever.search(q, k),
        (1, 3),
        level="passage",
        depth=DEPTH,  # type: ignore[attr-defined]
    )
    ms = 1000 * (time.perf_counter() - start) / max(rep.n, 1)
    return rep.recall[1], rep.recall[3], rep.mrr, ms


def table(rows: dict[str, object], sets: dict[str, list[QAItem]]) -> dict[str, dict[str, tuple]]:
    out: dict[str, dict[str, tuple]] = {}
    print("| setup | original R@1 | R@3 | MRR@10 | paraphrase R@1 | R@3 | MRR@10 | ms/question |")
    print("|---|---|---|---|---|---|---|---|")
    for name, retriever in rows.items():
        o, p = (scores(qa, retriever) for qa in sets.values())
        out[name] = {"original": o, "paraphrase": p}
        print(
            f"| {name} | {o[0]:.3f} | {o[1]:.3f} | {o[2]:.3f} "
            f"| {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {(o[3] + p[3]) / 2:.1f} |",
            flush=True,
        )
    return out


def llm_cost(qa: list[QAItem], hybrid: HybridRetriever, fetch_k: int) -> tuple[float, float]:
    """LLM calls and estimated prompt tokens per question for pointwise LLM reranking."""
    fake = FakeLLM(["امتیاز: 5"])
    reranker = LLMReranker(fake)
    for item in qa:
        reranker.rerank(item.question, hybrid.search(item.question, fetch_k))
    tokens = sum(estimate_tokens(m.content) for call in fake.calls for m in call)
    return len(fake.calls) / len(qa), tokens / len(qa)


def save_chart(results: dict[str, dict[str, tuple]], path: Path = CHART) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    width = 0.38
    for j, (set_name, color) in enumerate((("original", "#4C78A8"), ("paraphrase", "#F58518"))):
        vals = [results[n][set_name][0] for n in names]
        xs = [i + (j - 0.5) * width for i in range(len(names))]
        bars = ax.bar(xs, vals, width, label=set_name, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("passage-level R@1")
    ax.set_ylim(0, 1.15)
    ax.set_title("Lesson 08: does reranking 20 candidates improve the top answer?")
    ax.legend(title="question set", fontsize=8, ncol=2, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    print(f"\nchart saved to {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chart", action="store_true", help=f"save {CHART.relative_to(ROOT)}")
    parser.add_argument("--cross-encoder", help="sentence-transformers CrossEncoder model name")
    parser.add_argument("--llm", help="model name on an OpenAI-compatible server")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    args = parser.parse_args()

    sets = {name: load_fixture_benchmark(qa_file=f)[1] for name, f in
            (("original", "qa.jsonl"), ("paraphrase", "qa_paraphrase.jsonl"))}  # fmt: skip
    hybrid, bm25 = first_stage()

    print("## Main comparison (first stage = RRF(BM25+stem, hashing), 20 candidates)\n")
    rows: dict[str, object] = {
        "no reranking": hybrid,
        "lexical rerank": RerankingRetriever(hybrid, LexicalReranker()),
        "lexical rerank + idf": RerankingRetriever(hybrid, LexicalReranker(idf=bm25.idf)),
    }
    extra: Reranker | None = None
    if args.cross_encoder:
        from parsrag.rerank import CrossEncoderReranker

        extra = CrossEncoderReranker(args.cross_encoder)
        rows[f"cross-encoder {args.cross_encoder}"] = RerankingRetriever(hybrid, extra)
    if args.llm:
        from parsrag.llm import OpenAICompatibleLLM

        extra = LLMReranker(OpenAICompatibleLLM(args.llm, base_url=args.base_url))
        rows[f"LLM {args.llm}"] = RerankingRetriever(hybrid, extra)
    results = table(rows, sets)

    print("\n## Ablation: proximity and first-stage prior (tuned on original only)\n")
    grid = {
        f"proximity {pw} / prior {prw}": RerankingRetriever(
            hybrid, LexicalReranker(proximity_weight=pw, prior_weight=prw)
        )
        for pw, prw in itertools.product((0.0, 0.3, 0.6), (0.0, 0.2))
    }
    table(grid, sets)

    print("\n## How many candidates to rerank (fetch_k)\n")
    table(
        {
            f"fetch_k={n}": RerankingRetriever(hybrid, LexicalReranker(), fetch_k=n)
            for n in (10, 20, 40)
        },
        sets,
    )

    print("\n## Cost of pointwise LLM reranking (estimated, nothing is called)\n")
    print("| fetch_k | LLM calls / question | prompt tokens / question (≈) |")
    print("|---|---|---|")
    for n in (10, 20, 40):
        calls, tokens = llm_cost(sets["original"], hybrid, n)
        print(f"| {n} | {calls:.0f} | {tokens:,.0f} |")

    if args.chart:
        save_chart(results)


if __name__ == "__main__":
    main()

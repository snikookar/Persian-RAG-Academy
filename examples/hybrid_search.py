"""Lexical vs dense vs hybrid retrieval on both fixture question sets (lesson 07).

    python examples/hybrid_search.py            # tables (offline, a few seconds)
    python examples/hybrid_search.py --chart    # also save docs/images/hybrid_mrr.png

Everything uses the course pipeline: recursive 400-character chunks with
heading context, normalised text. "dense" is HashingEmbedding — offline and
deterministic, but still *lexical* underneath (character n-grams); see the
lesson for how to plug in a neural model with ``--model st:NAME``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from parsrag.embeddings import Embedding, HashingEmbedding, TfidfEmbedding
from parsrag.eval import evaluate_retrieval
from parsrag.eval.benchmark import index_documents, load_fixture_benchmark
from parsrag.retrieval import (
    PERSIAN_STOPWORDS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    PersianTokenizer,
    Retriever,
)

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "docs" / "images" / "hybrid_mrr.png"
SETS = {"original": "qa.jsonl", "paraphrase": "qa_paraphrase.jsonl"}


def make_embedding(spec: str) -> Embedding:
    if spec == "hashing":
        return HashingEmbedding()
    if spec == "tfidf-char":
        return TfidfEmbedding("char")
    if spec.startswith("st:"):
        from parsrag.embeddings import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding(spec[3:])
    raise SystemExit(f"unknown model {spec!r}")


def build(model: str) -> dict[str, Retriever]:
    docs, _ = load_fixture_benchmark()
    store = index_documents(docs, make_embedding(model))
    chunks = store.chunks
    bm25 = BM25Retriever(chunks)
    stem = BM25Retriever(chunks, tokenizer=PersianTokenizer(stem=True))
    dense = DenseRetriever(store)
    return {
        "BM25": bm25,
        "BM25 + stem": stem,
        "BM25 + split ZWNJ": BM25Retriever(chunks, tokenizer=PersianTokenizer(split_zwnj=True)),
        "BM25 + stopwords": BM25Retriever(
            chunks, tokenizer=PersianTokenizer(stopwords=PERSIAN_STOPWORDS)
        ),
        f"dense ({model})": dense,
        "hybrid RRF (BM25 + dense)": HybridRetriever([bm25, dense]),
        "hybrid RRF (BM25+stem + dense)": HybridRetriever([stem, dense]),
        "weighted 0.5/0.5": HybridRetriever([bm25, dense], fusion="weighted", weights=[0.5, 0.5]),
        "weighted 0.3/0.7": HybridRetriever([bm25, dense], fusion="weighted", weights=[0.3, 0.7]),
        "weighted 0.7/0.3": HybridRetriever([bm25, dense], fusion="weighted", weights=[0.7, 0.3]),
    }


def measure(retrievers: dict[str, Retriever]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for set_name, qa_file in SETS.items():
        _, qa = load_fixture_benchmark(qa_file=qa_file)
        for name, r in retrievers.items():
            search: Callable = r.search  # type: ignore[type-arg]
            rep = evaluate_retrieval(qa, lambda q, k, s=search: s(q, k), (1, 3, 5), level="passage")
            out.setdefault(name, {})[set_name] = rep.row()
    return out


def save_chart(results: dict[str, dict[str, dict[str, float]]], path: Path = CHART) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in results if not n.startswith("weighted 0.7") and "stopwords" not in n]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    width = 0.38
    xs = range(len(names))
    for j, (set_name, color) in enumerate((("original", "#4C78A8"), ("paraphrase", "#F58518"))):
        vals = [results[n][set_name]["MRR"] for n in names]
        bars = ax.bar([x + (j - 0.5) * width for x in xs], vals, width, label=set_name, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("passage-level MRR")
    ax.set_ylim(0, 1.15)
    ax.set_title("Lesson 07: BM25 vs dense (HashingEmbedding) vs hybrid")
    ax.legend(title="question set", fontsize=8, ncol=2, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    print(f"\nchart saved to {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="hashing", help="hashing | tfidf-char | st:NAME")
    parser.add_argument("--chart", action="store_true", help=f"save {CHART.relative_to(ROOT)}")
    args = parser.parse_args()
    results = measure(build(args.model))
    print("| retriever | original R@1 | R@3 | MRR | paraphrase R@1 | R@3 | MRR |")
    print("|---|---|---|---|---|---|---|")
    for name, by_set in results.items():
        o, p = by_set["original"], by_set["paraphrase"]
        print(
            f"| {name} | {o['R@1']:.3f} | {o['R@3']:.3f} | {o['MRR']:.3f} "
            f"| {p['R@1']:.3f} | {p['R@3']:.3f} | {p['MRR']:.3f} |"
        )
    if args.chart:
        save_chart(results)


if __name__ == "__main__":
    main()

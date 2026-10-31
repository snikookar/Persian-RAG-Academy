"""Compare chunking strategies on the fixture corpus (lesson 04).

    python examples/chunking_benchmark.py            # print the table
    python examples/chunking_benchmark.py --chart    # also save docs/images/chunking_mrr.png
                                                     # (needs: pip install -e ".[dev]")

Fully offline: HashingEmbedding, no API keys, no downloads. Every document is
chunked on its *original* text (so offsets point to the source), the chunk
text is normalised (lesson 02), and questions are normalised the same way.

Passage-level metrics count a hit only if a retrieved chunk *contains the
answer's evidence span* (``qa.jsonl`` → ``evidence``); doc-level metrics only
ask for the right document. The corpus is small and synthetic (25 documents,
59 answerable questions): one question moves a score by ~1.7 points.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from parsrag import HashingEmbedding, InMemoryVectorStore
from parsrag.chunking import (
    Chunker,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    chunk_document,
)
from parsrag.eval import evaluate_retrieval, load_qa
from parsrag.eval.retrieval import context_text
from parsrag.loaders import load_directory
from parsrag.normalize import normalize
from parsrag.types import Document, SearchResult

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "data" / "fixtures"
CHART = ROOT / "docs" / "images" / "chunking_mrr.png"
SIZES = (200, 400, 800)


@dataclass
class Row:
    name: str
    size: int | None
    chunks: int
    avg_len: float
    metrics: dict[str, float]


def run(
    docs: Sequence[Document], chunker: Chunker, *, heading_context: bool = False
) -> tuple[int, float, dict[str, float]]:
    store = InMemoryVectorStore(HashingEmbedding())
    for doc in docs:
        store.add(
            chunk_document(doc, chunker, preprocess=normalize, heading_context=heading_context)
        )

    def retrieve(question: str, k: int) -> list[SearchResult]:
        return store.search(normalize(question), k)

    qa = load_qa(FIXTURES / "qa.jsonl")
    passage = evaluate_retrieval(qa, retrieve, ks=(1, 3, 5), level="passage")
    doc = evaluate_retrieval(qa, retrieve, ks=(1, 3), level="doc")
    metrics = {
        "P-R@1": passage.recall[1],
        "P-R@3": passage.recall[3],
        "P-R@5": passage.recall[5],
        "P-MRR": passage.mrr,
        "D-R@1": doc.recall[1],
        "D-R@3": doc.recall[3],
    }
    avg_len = sum(len(context_text(c)) for c in store.chunks) / len(store)
    return len(store), avg_len, metrics


def benchmark() -> list[Row]:
    docs = load_directory(FIXTURES / "corpus", "*.md")
    rows: list[Row] = []
    for size in SIZES:
        overlap = size // 8
        for name, chunker, ctx in (
            ("fixed", FixedSizeChunker(size, overlap), False),
            ("recursive", RecursiveChunker(size, overlap), False),
            ("recursive+headings", RecursiveChunker(size, overlap), True),
        ):
            rows.append(Row(name, size, *run(docs, chunker, heading_context=ctx)))
    for window in (0, 1, 2):
        rows.append(
            Row(f"sentence-window w={window}", None, *run(docs, SentenceWindowChunker(window)))
        )
    return rows


def print_table(rows: list[Row]) -> None:
    keys = list(rows[0].metrics)
    head = f"{'strategy':<24} {'size':>5} {'chunks':>6} {'avg len':>7}  "
    print(head + " ".join(f"{k:>6}" for k in keys))
    print("-" * (47 + 7 * len(keys)))
    for r in rows:
        size = str(r.size) if r.size else "-"
        values = " ".join(f"{r.metrics[k]:>6.3f}" for k in keys)
        print(f"{r.name:<24} {size:>5} {r.chunks:>6} {r.avg_len:>7.0f}  {values}")


def save_chart(rows: list[Row], path: Path = CHART) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {  # fixed categorical order: blue, orange, aqua
        "fixed": ("#2a78d6", "o", "fixed-size"),
        "recursive": ("#eb6834", "s", "recursive"),
        "recursive+headings": ("#1baf7a", "D", "recursive + heading context"),
    }
    metric = "P-MRR"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ends: list[tuple[float, str]] = []
    for key, (color, marker, label) in series.items():
        pts = [(r.size, r.metrics[metric]) for r in rows if r.name == key]
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, color=color, marker=marker, markersize=7, linewidth=2, label=label)
        ends.append((ys[-1], f"{ys[-1]:.2f}"))
    # end-of-line value labels, nudged apart so they never overlap
    placed: list[float] = []
    for y, text in sorted(ends):
        y_label = max([y, *[p + 0.03 for p in placed]])
        placed.append(y_label)
        ax.annotate(
            text,
            (SIZES[-1], y_label),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#3d3d3a",
        )
    window = next(r for r in rows if r.name == "sentence-window w=1")
    ax.axhline(window.metrics[metric], color="#8a8a85", linestyle="--", linewidth=1.5)
    ax.annotate(
        f"sentence window (w=1): {window.metrics[metric]:.2f}",
        (SIZES[-1], window.metrics[metric]),
        xytext=(0, -12),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="#3d3d3a",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES, [str(s) for s in SIZES])
    ax.set_xlim(SIZES[0] * 0.85, SIZES[-1] * 1.35)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("chunk size (characters, overlap = size / 8)", color="#3d3d3a")
    ax.set_ylabel("passage-level MRR (higher is better)", color="#3d3d3a")
    ax.set_title(
        "Chunking strategies on the fixture corpus (HashingEmbedding)\n"
        "25 synthetic docs, 59 answerable questions",
        fontsize=10,
        color="#1f1f1d",
    )
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#3d3d3a")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    print(f"\nchart saved to {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chart", action="store_true", help=f"save {CHART.relative_to(ROOT)}")
    args = parser.parse_args()
    rows = benchmark()
    print_table(rows)
    if args.chart:
        save_chart(rows)


if __name__ == "__main__":
    main()

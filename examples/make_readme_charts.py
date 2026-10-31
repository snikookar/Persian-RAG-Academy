"""Regenerate the README / docs-site result charts from the repo's own benchmarks.

    pip install -e ".[dev,loaders]"          # matplotlib (+ pypdf for the PDF chart)
    python examples/make_readme_charts.py    # ~40 s, fully offline

Nothing here is typed in by hand: every number is measured again by calling
the same functions the lesson examples use (``normalization_impact.py``,
``pdf_repair_impact.py``, ``chunking_benchmark.py``, ``hybrid_search.py``,
``reranking.py``, ``agentic_rag.py``). The charts go to ``docs/images/readme_*.png``
and the numbers behind them to ``docs/eval/readme_numbers.json``.

All of it runs on the repository's small synthetic Persian corpus (25 documents,
59 + 46 answerable questions) with offline, deterministic models
(HashingEmbedding, FakeLLM, OfflineAgentLLM). These are teaching numbers, not
a leaderboard: for real embedding models run ``examples/benchmark_embeddings.py``
on your own machine.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
IMAGES = ROOT / "docs" / "images"
NUMBERS = ROOT / "docs" / "eval" / "readme_numbers.json"

# Light-background palette: three validated categorical slots + a neutral gray.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, INK_2, GRID, AXIS = "#1f1f1d", "#3d3d3a", "#e6e5e0", "#b5b4ad"
FOOTNOTE = "Synthetic Persian fixture corpus · offline models · see examples/make_readme_charts.py"


def load_example(name: str) -> ModuleType:
    """Import ``examples/<name>.py`` as a module (the examples are not a package)."""
    spec = importlib.util.spec_from_file_location(f"_example_{name}", EXAMPLES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- measurements


def measure_normalization() -> dict[str, Any]:
    ex = load_example("normalization_impact")
    from parsrag.normalize import Normalizer

    out: dict[str, Any] = {"chunk_size": 300}
    out["raw"] = dict(zip(("hit@1", "hit@3"), ex.evaluate(None, 300), strict=True))
    out["normalized"] = dict(zip(("hit@1", "hit@3"), ex.evaluate(Normalizer(), 300), strict=True))
    without = ex.evaluate(Normalizer(unify_chars=False), 300)
    out["without unify_chars"] = dict(zip(("hit@1", "hit@3"), without, strict=True))
    return out


def measure_pdf() -> dict[str, Any] | None:
    try:
        ex = load_example("pdf_repair_impact")
    except ImportError as exc:  # pypdf is in the `loaders` extra
        print(f"skipping the PDF chart ({exc}); pip install -e '.[loaders]'")
        return None
    out: dict[str, Any] = {}
    for name in ex.PHRASES:
        source = (ex.DOCS / "sources" / f"{name}.txt").read_text(encoding="utf-8")
        reference = "\n\n".join(page.strip() for page in source.split("\f"))
        raw = ex.load_pdf(ex.DOCS / f"{name}.pdf", repair=False).text
        out[name] = {label: ex.word_recall(step(raw), reference) for label, step in ex.STEPS}
    return out


def measure_chunking() -> dict[str, Any]:
    ex = load_example("chunking_benchmark")
    rows = ex.benchmark()
    return {f"{r.name} {r.size or ''}".strip(): r.metrics["P-MRR"] for r in rows}


def measure_hybrid() -> dict[str, Any]:
    ex = load_example("hybrid_search")
    results = ex.measure(ex.build("hashing"))
    keep = {
        "BM25": "BM25",
        "BM25 + stem": "BM25 + stem",
        "dense (hashing)": "dense (hashing)",
        "hybrid RRF (BM25 + dense)": "RRF (BM25 + dense)",
        "hybrid RRF (BM25+stem + dense)": "RRF (BM25+stem + dense)",
    }
    return {
        label: {s: results[name][s]["MRR"] for s in ("original", "paraphrase")}
        for name, label in keep.items()
    }


def measure_rerank() -> dict[str, Any]:
    ex = load_example("reranking")
    sets = {
        name: ex.load_fixture_benchmark(qa_file=f)[1]
        for name, f in (("original", "qa.jsonl"), ("paraphrase", "qa_paraphrase.jsonl"))
    }
    hybrid, _ = ex.first_stage()
    rows = {
        "no reranking": hybrid,
        "lexical rerank of 20": ex.RerankingRetriever(hybrid, ex.LexicalReranker()),
    }
    out: dict[str, Any] = {}
    for label, retriever in rows.items():
        for set_name, qa in sets.items():
            r1, _, mrr, _ = ex.scores(qa, retriever)
            out.setdefault(label, {})[set_name] = {"R@1": r1, "MRR@10": mrr}
    return out


def measure_agent() -> dict[str, Any]:
    ex = load_example("agentic_rag")
    from parsrag import PersianRAG
    from parsrag.agent import AgenticRAG, OfflineAgentLLM
    from parsrag.llm import CountingLLM

    rag = PersianRAG.course_default()
    agent = AgenticRAG(rag, OfflineAgentLLM(vocabulary=rag.bm25._postings))
    counter = CountingLLM(rag.llm)
    rag.llm = counter
    pipeline = rag.generator
    out: dict[str, Any] = {}
    for set_name, file in ex.SETS.items():
        qa = ex.load_fixture_benchmark(qa_file=file)[1]
        for label, system, llm in (("lesson 09", pipeline, counter), ("agent", agent, agent.llm)):
            report = ex.evaluate_pipeline(system, qa, name=label, n_resamples=200)
            cost = ex.cost(system, llm, qa)
            out.setdefault(set_name, {})[label] = {
                "retrieval_r@1": report.metrics["retrieval_r@1"],
                "answer_f1": report.metrics["answer_f1"],
                "answered_rate": report.metrics["answered_rate"],
                "llm_calls": cost["calls"],
            }
    return out


# ---------------------------------------------------------------------- charts


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def _style(ax: Any, ylabel: str, ylim: tuple[float, float] = (0, 1.08)) -> None:
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _bars(ax: Any, labels: list[str], series: dict[str, tuple[list[float], str]]) -> None:
    n = len(series)
    width = 0.8 / n
    for j, (name, (values, color)) in enumerate(series.items()):
        xs = [i + (j - (n - 1) / 2) * width for i in range(len(labels))]
        bars = ax.bar(xs, values, width * 0.92, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2, color=INK_2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)


def _save(fig: Any, name: str) -> Path:
    fig.text(0.01, 0.01, FOOTNOTE, fontsize=6.5, color=GRAY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = IMAGES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    _plt().close(fig)
    print(f"saved {path.relative_to(ROOT)}")
    return path


def chart_normalization(data: dict[str, Any]) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = ["raw text", "without letter\nunification", "full normalisation"]
    keys = ["raw", "without unify_chars", "normalized"]
    _bars(
        ax,
        labels,
        {
            "hit@1": ([data[k]["hit@1"] for k in keys], BLUE),
            "hit@3": ([data[k]["hit@3"] for k in keys], ORANGE),
        },
    )
    _style(ax, "share of questions (59 answerable)", (0, 1.15))
    ax.set_title("Lesson 02 · Persian normalisation (naive RAG, chunk 300)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _save(fig, "readme_normalization.png")


def chart_pdf(data: dict[str, Any]) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    steps = list(next(iter(data.values())))
    labels = ["raw pypdf", "+ NFKC", "+ visual order", "+ ZWNJ\n(= load_pdf)"]
    names = {"kahroba-catalog": "logical-order PDF", "sepidar-newsletter": "visual-order PDF"}
    colors = iter((BLUE, ORANGE))
    _bars(ax, labels, {names[d]: ([data[d][s] for s in steps], next(colors)) for d in data})
    _style(ax, "word recall vs. source text", (0, 1.15))
    ax.set_title("Lesson 03 · repairing text extracted from Persian PDFs")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "readme_pdf_repair.png")


def chart_chunking(data: dict[str, Any]) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    sizes = ["200", "400", "800"]
    _bars(
        ax,
        [f"{s} chars" for s in sizes],
        {
            "fixed-size": ([data[f"fixed {s}"] for s in sizes], BLUE),
            "recursive": ([data[f"recursive {s}"] for s in sizes], ORANGE),
            "recursive + heading context": ([data[f"recursive+headings {s}"] for s in sizes], AQUA),
        },
    )
    window = data["sentence-window w=1"]
    ax.axhline(window, color=GRAY, linestyle="--", linewidth=1.2, zorder=0,
               label=f"sentence window, w=1 ({window:.2f})")  # fmt: skip
    _style(ax, "passage-level MRR", (0, 1.2))
    ax.set_title("Lesson 04 · chunking strategies (HashingEmbedding)")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    _save(fig, "readme_chunking.png")


def chart_hybrid(data: dict[str, Any]) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    labels = list(data)
    _bars(
        ax,
        [label.replace(" (", "\n(") for label in labels],
        {
            "original questions (59)": ([data[n]["original"] for n in labels], BLUE),
            "paraphrased questions (46)": ([data[n]["paraphrase"] for n in labels], ORANGE),
        },
    )
    _style(ax, "passage-level MRR", (0, 1.15))
    ax.set_title("Lesson 07 · BM25 vs dense vs hybrid (RRF)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _save(fig, "readme_hybrid.png")


def chart_rerank(data: dict[str, Any]) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = ["original · R@1", "original · MRR@10", "paraphrase · R@1", "paraphrase · MRR@10"]
    cells = [("original", "R@1"), ("original", "MRR@10"), ("paraphrase", "R@1"),
             ("paraphrase", "MRR@10")]  # fmt: skip
    colors = iter((GRAY, BLUE))
    _bars(
        ax,
        labels,
        {name: ([data[name][s][m] for s, m in cells], next(colors)) for name in data},
    )
    _style(ax, "passage level", (0, 1.15))
    ax.set_title("Lesson 08 · retrieve 20, rerank, keep the best (offline lexical reranker)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _save(fig, "readme_rerank.png")


def chart_agent(data: dict[str, Any]) -> None:
    plt = _plt()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(8.4, 3.7), gridspec_kw={"width_ratios": [3, 1.4]}
    )
    metrics = [("retrieval_r@1", "R@1"), ("answer_f1", "answer F1"), ("answered_rate", "answered")]
    labels = [f"{s}\n{m}" for s in ("original", "paraphrase") for _, m in metrics]
    series = {}
    for system, color in (("lesson 09", GRAY), ("agent", BLUE)):
        series[f"{system} pipeline" if system == "lesson 09" else "agent (lesson 11)"] = (
            [data[s][system][k] for s in ("original", "paraphrase") for k, _ in metrics],
            color,
        )
    _bars(left, labels, series)
    left.tick_params(axis="x", labelsize=7.5)
    _style(left, "score", (0, 1.22))
    left.set_title("Quality")
    left.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)

    calls = {
        "lesson 09 pipeline": ([data[s]["lesson 09"]["llm_calls"] for s in ("original", "paraphrase")], GRAY),
        "agent (lesson 11)": ([data[s]["agent"]["llm_calls"] for s in ("original", "paraphrase")], BLUE),
    }  # fmt: skip
    _bars(right, ["original", "paraphrase"], calls)
    top = max(max(v) for v, _ in calls.values())
    _style(right, "LLM calls per question (mean)", (0, top * 1.25))
    right.set_title("Cost")
    fig.suptitle("Lesson 11 · agentic RAG vs the lesson 09 pipeline (offline rule-based LLM)",
                 fontsize=10.5, color=INK)  # fmt: skip
    _save(fig, "readme_agent.png")


STEPS: dict[str, tuple[Callable[[], Any], Callable[[Any], None]]] = {
    "normalization": (measure_normalization, chart_normalization),
    "pdf_repair": (measure_pdf, chart_pdf),
    "chunking": (measure_chunking, chart_chunking),
    "hybrid": (measure_hybrid, chart_hybrid),
    "rerank": (measure_rerank, chart_rerank),
    "agent": (measure_agent, chart_agent),
}


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=list(STEPS), action="append", help="repeatable")
    parser.add_argument("--no-charts", action="store_true", help="only print / save the numbers")
    args = parser.parse_args()
    numbers: dict[str, Any] = {}
    if NUMBERS.exists():
        numbers = json.loads(NUMBERS.read_text(encoding="utf-8"))
    for name in args.only or STEPS:
        measure, chart = STEPS[name]
        print(f"measuring {name} ...", flush=True)
        data = measure()
        if data is None:
            continue
        numbers[name] = data
        if not args.no_charts:
            chart(data)
    NUMBERS.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_rounded(numbers), ensure_ascii=False, indent=2)
    NUMBERS.write_text(text + "\n", encoding="utf-8")
    print(f"numbers saved to {NUMBERS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

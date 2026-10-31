"""Benchmark embedding models on the Persian fixture corpus (lesson 05).

Offline (default — no downloads, no API keys):

    python examples/benchmark_embeddings.py
    python examples/benchmark_embeddings.py --mismatch      # normalisation experiment

Real models — run these on your own machine; they download weights or call an API:

    pip install -e ".[st]"
    python examples/benchmark_embeddings.py --model st:intfloat/multilingual-e5-small \\
        --query-prefix "query: " --doc-prefix "passage: "
    python examples/benchmark_embeddings.py --model st:BAAI/bge-m3

    pip install -e ".[openai]"
    export OPENAI_API_KEY=...            # not needed for a local server
    python examples/benchmark_embeddings.py --model openai:text-embedding-3-small
    python examples/benchmark_embeddings.py --model openai:nomic-embed-text \\
        --base-url http://localhost:11434/v1          # e.g. Ollama

Each run prints a Markdown table row you can paste into the results table of
lessons/05-embeddings/README.fa.md. Only report numbers you actually measured.
"""

from __future__ import annotations

import argparse
import time

from parsrag.embeddings import Embedding, HashingEmbedding, TfidfEmbedding
from parsrag.eval.benchmark import benchmark_embedding, load_fixture_benchmark
from parsrag.eval.dataset import QAItem
from parsrag.normalize import normalize

OFFLINE = ("hashing", "tfidf-word", "tfidf-char")
HEADER = (
    "| model | dim | passage R@1 | passage R@3 | passage MRR | passage nDCG@3 | doc R@1 | time (s) |\n"
    "|---|---|---|---|---|---|---|---|"
)

# Users typing on an Arabic keyboard layout: Arabic yeh/kaf and Arabic-Indic digits.
ARABIC_KEYBOARD = str.maketrans(
    {"ی": "ي", "ک": "ك", **{p: a for p, a in zip("۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩", strict=True)}}
)


def make_embedding(spec: str, args: argparse.Namespace) -> Embedding:
    if spec == "hashing":
        return HashingEmbedding()
    if spec == "tfidf-word":
        return TfidfEmbedding("word")
    if spec == "tfidf-char":
        return TfidfEmbedding("char", ngram_range=(2, 4))
    if spec.startswith("st:"):
        from parsrag.embeddings import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding(
            spec[3:], query_prefix=args.query_prefix, document_prefix=args.doc_prefix
        )
    if spec.startswith("openai:"):
        from parsrag.embeddings import OpenAICompatibleEmbedding

        return OpenAICompatibleEmbedding(
            spec[7:],
            base_url=args.base_url,
            query_prefix=args.query_prefix,
            document_prefix=args.doc_prefix,
        )
    raise SystemExit(
        f"unknown model spec {spec!r}: use hashing, tfidf-word, tfidf-char, st:NAME or openai:NAME"
    )


def row(spec: str, args: argparse.Namespace) -> str:
    docs, qa = load_fixture_benchmark()
    embedding = make_embedding(spec, args)
    start = time.perf_counter()
    reports = benchmark_embedding(embedding, docs, qa)
    seconds = time.perf_counter() - start
    p, d = reports["passage"], reports["doc"]
    return (
        f"| {spec} | {embedding.dim} | {p.recall[1]:.3f} | {p.recall[3]:.3f} | {p.mrr:.3f} "
        f"| {p.ndcg[3]:.3f} | {d.recall[1]:.3f} | {seconds:.1f} |"
    )


def mismatch() -> None:
    """Index and query normalised differently, with questions typed on an Arabic keyboard."""
    docs, qa = load_fixture_benchmark()
    messy = [
        QAItem(**{**q.model_dump(), "question": q.question.translate(ARABIC_KEYBOARD)}) for q in qa
    ]
    setups = [
        ("clean questions, normalise both", qa, normalize, "same"),
        ("Arabic-keyboard questions, normalise both", messy, normalize, "same"),
        ("Arabic-keyboard questions, normalise index only", messy, normalize, None),
        ("Arabic-keyboard questions, normalise neither", messy, None, "same"),
    ]
    print("| setup | " + " | ".join(f"{m} R@3 / MRR" for m in OFFLINE) + " |")
    print("|---|" + "---|" * len(OFFLINE))
    for label, items, index_prep, query_prep in setups:
        cells = []
        for spec in OFFLINE:
            emb = make_embedding(spec, argparse.Namespace())
            r = benchmark_embedding(
                emb, docs, items, preprocess=index_prep, query_preprocess=query_prep
            )["passage"]
            cells.append(f"{r.recall[3]:.3f} / {r.mrr:.3f}")
        print(f"| {label} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", action="append", help="hashing | tfidf-word | tfidf-char | st:NAME | openai:NAME"
    )
    parser.add_argument("--query-prefix", default="", help='e.g. "query: " for E5 models')
    parser.add_argument("--doc-prefix", default="", help='e.g. "passage: " for E5 models')
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--mismatch", action="store_true", help="run the normalisation experiment")
    args = parser.parse_args()
    if args.mismatch:
        mismatch()
        return
    print(HEADER)
    for spec in args.model or OFFLINE:
        print(row(spec, args), flush=True)


if __name__ == "__main__":
    main()

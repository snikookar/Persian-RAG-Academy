"""Measure how much Persian normalization helps retrieval on the fixtures (lesson 02).

    python examples/normalization_impact.py

Fully offline: HashingEmbedding + FakeLLM, no API keys, no downloads.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from parsrag import FakeLLM, HashingEmbedding
from parsrag.eval import hit_rate, load_qa
from parsrag.naive_rag import NaiveRAG, load_documents
from parsrag.normalize import Normalizer, NormalizerConfig

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def evaluate(preprocess: Normalizer | None, chunk_size: int) -> tuple[float, float]:
    rag = NaiveRAG(HashingEmbedding(), FakeLLM(), chunk_size=chunk_size, preprocess=preprocess)
    rag.index(load_documents(FIXTURES / "corpus"))
    qa = load_qa(FIXTURES / "qa.jsonl")
    return hit_rate(qa, rag.retrieve, k=1), hit_rate(qa, rag.retrieve, k=3)


def main() -> None:
    print("Hit-rate on 24 answerable fixture questions (HashingEmbedding)\n")
    print(f"{'chunk size':>10} | {'raw @1':>7} {'norm @1':>8} | {'raw @3':>7} {'norm @3':>8}")
    print("-" * 52)
    for size in (200, 300, 400):
        raw1, raw3 = evaluate(None, size)
        n1, n3 = evaluate(Normalizer(), size)
        print(f"{size:>10} | {raw1:>7.3f} {n1:>8.3f} | {raw3:>7.3f} {n3:>8.3f}")

    print("\nAblation at chunk size 300: switch off one step at a time\n")
    full1, full3 = evaluate(Normalizer(), 300)
    print(f"{'config':<28} {'@1':>6} {'@3':>6}")
    print(f"{'all steps':<28} {full1:>6.3f} {full3:>6.3f}")
    for f in fields(NormalizerConfig):
        override = {f.name: "keep" if f.name == "digits" else False}
        s1, s3 = evaluate(Normalizer(**override), 300)
        print(f"{'without ' + f.name:<28} {s1:>6.3f} {s3:>6.3f}")


if __name__ == "__main__":
    main()

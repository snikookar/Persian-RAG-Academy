"""Grounded answers with citations and an honest «نمی‌دانم» (lesson 09).

    python examples/grounded_answers.py                  # offline: FakeLLM (extractive)

With a real OpenAI-compatible model, on your own machine (never in CI):

    pip install -e ".[openai]"
    export OPENAI_API_KEY=...
    python examples/grounded_answers.py --llm gpt-4o-mini
    python examples/grounded_answers.py --llm qwen2.5:7b --base-url http://localhost:11434/v1

The retrieval pipeline is the lesson 07–08 default; the abstention threshold
is calibrated on qa.jsonl and then applied unchanged to qa_paraphrase.jsonl.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from parsrag import FakeLLM, HashingEmbedding
from parsrag.eval.benchmark import index_documents, load_fixture_benchmark
from parsrag.eval.dataset import QAItem
from parsrag.generate import (
    Confidence,
    GroundedGenerator,
    abstention_stats,
    calibrate_threshold,
    retriever_confidence,
)
from parsrag.llm import LLM, Message
from parsrag.rerank import LexicalReranker, RerankingRetriever
from parsrag.retrieval import (
    PERSIAN_STOPWORDS,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    PersianTokenizer,
    Retriever,
)

SAMPLES = ("q05", "q32", "q26")


def pipeline() -> tuple[Retriever, dict[str, Confidence]]:
    docs, _ = load_fixture_benchmark()
    store = index_documents(docs, HashingEmbedding())
    bm25 = BM25Retriever(store.chunks, tokenizer=PersianTokenizer(stem=True))
    dense = DenseRetriever(store)
    hybrid = HybridRetriever([bm25, dense])
    reranked = RerankingRetriever(hybrid, LexicalReranker())
    signals = {
        "top BM25 score": retriever_confidence(bm25),
        "top dense cosine": retriever_confidence(dense),
        "top reranker score": retriever_confidence(reranked),
    }
    return reranked, signals


def fake_llm(min_overlap: int) -> FakeLLM:
    return FakeLLM(mode="extractive", cite=True, ignore=PERSIAN_STOPWORDS, min_overlap=min_overlap)


def show_samples(gen: GroundedGenerator, qa: list[QAItem]) -> None:
    by_id = {q.id: q for q in qa}
    for qid in SAMPLES:
        ans = gen.answer(by_id[qid].question)
        print(f"❓ {ans.question}")
        print(f"💬 {ans.answer}")
        if ans.abstained:
            print(f"   (abstained: {ans.abstain_reason}, confidence={ans.top_score:.2f})")
        for c in ans.citations:
            print(f"   [{c.number}] {c.doc_id} › {c.section or '—'}  chars {c.start}–{c.end}")
        print()


def abstention_row(
    label: str, gen: GroundedGenerator, sets: dict[str, list[QAItem]]
) -> tuple[str, int]:
    cells, invalid = [], 0
    for qa in sets.values():
        answers = [gen.answer(q.question) for q in qa]
        invalid += sum(bool(a.invalid_citations) or a.uncited for a in answers)
        s = abstention_stats([a.abstained for a in answers], [q.answerable for q in qa])
        cells.append(f"{s.precision:.2f} | {s.recall:.2f} | {s.f1:.2f} | {s.tn}/{s.tn + s.fp}")
    return f"| {label} | " + " | ".join(cells) + " |", invalid


def sloppy_demo(retriever: Retriever, qa: Sequence[QAItem]) -> None:
    """A model that sometimes cites a block that does not exist, or forgets to cite."""
    replies = iter(["پاسخ این است [1].", "پاسخ این است [7].", "پاسخ بدون ارجاع.", "نمی‌دانم"])

    def handler(messages: list[Message]) -> str:
        return next(replies)

    gen = GroundedGenerator(retriever, FakeLLM(handler=handler), k=4)
    print("| model reply | citations kept | invalid | uncited | abstained |")
    print("|---|---|---|---|---|")
    for item in list(qa)[:4]:
        a = gen.answer(item.question)
        kept = ", ".join(f"[{c.number}] {c.chunk_id}" for c in a.citations) or "—"
        print(f"| {a.raw} | {kept} | {a.invalid_citations or '—'} | {a.uncited} | {a.abstained} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", help="model name on an OpenAI-compatible server")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    args = parser.parse_args()

    sets = {name: load_fixture_benchmark(qa_file=f)[1] for name, f in
            (("original", "qa.jsonl"), ("paraphrase", "qa_paraphrase.jsonl"))}  # fmt: skip
    retriever, signals = pipeline()
    orig = sets["original"]
    labels = [q.answerable for q in orig]

    print("## 1. Which score separates answerable from unanswerable questions?\n")
    print(
        "Threshold calibrated on the original set (max abstention F1), then reused on paraphrases.\n"
    )
    print("| confidence signal | threshold | original P | R | F1 | paraphrase P | R | F1 |")
    print("|---|---|---|---|---|---|---|---|")
    thresholds = {}
    for name, conf in signals.items():
        t, s = calibrate_threshold([conf(q.question, []) for q in orig], labels)
        thresholds[name] = t
        para = sets["paraphrase"]
        p = abstention_stats([conf(q.question, []) < t for q in para], [q.answerable for q in para])
        print(
            f"| {name} | {t:.3f} | {s.precision:.2f} | {s.recall:.2f} | {s.f1:.2f} "
            f"| {p.precision:.2f} | {p.recall:.2f} | {p.f1:.2f} |"
        )

    bm25_conf, t = signals["top BM25 score"], thresholds["top BM25 score"]
    print("\n## 2. Threshold vs instruction-level refusal (FakeLLM stand-in)\n")
    print(
        "Positive class = «should abstain». 'answered' = answerable questions that got an answer.\n"
    )
    print("| setup | orig P | R | F1 | answered | para P | R | F1 | answered |")
    print("|---|---|---|---|---|---|---|---|---|")
    setups = {
        "baseline (FakeLLM min_overlap=1)": GroundedGenerator(retriever, fake_llm(1)),
        "instruction only (min_overlap=3)": GroundedGenerator(retriever, fake_llm(3)),
        "threshold only (BM25 ≥ t)": GroundedGenerator(
            retriever, fake_llm(1), min_score=t, confidence=bm25_conf
        ),
        "threshold + instruction": GroundedGenerator(
            retriever, fake_llm(3), min_score=t, confidence=bm25_conf
        ),
    }
    if args.llm:
        from parsrag.llm import OpenAICompatibleLLM

        real: LLM = OpenAICompatibleLLM(args.llm, base_url=args.base_url)
        setups[f"{args.llm}: instruction only"] = GroundedGenerator(retriever, real)
        setups[f"{args.llm}: threshold + instruction"] = GroundedGenerator(
            retriever, real, min_score=t, confidence=bm25_conf
        )
    bad = {}
    for label, gen in setups.items():
        row, bad[label] = abstention_row(label, gen, sets)
        print(row, flush=True)

    print("\n## 3. Citation checks over both sets\n")
    for label, n in bad.items():
        print(f"- {label}: {n} answers with an invalid or missing citation")

    print("\n## 4. Sample answers (threshold only)\n")
    show_samples(setups["threshold only (BM25 ≥ t)"], orig)

    print("## 5. Citation validation catches a sloppy model\n")
    sloppy_demo(retriever, orig)


if __name__ == "__main__":
    main()

"""How much can we trust our numbers? Bootstrap intervals and judges (lesson 10).

    python examples/evaluation_report.py                 # offline

With a real judge model, on your own machine (never in CI):

    pip install -e ".[openai]"
    python examples/evaluation_report.py --judge gpt-4o-mini
    python examples/evaluation_report.py --judge qwen2.5:7b --base-url http://localhost:11434/v1
"""

from __future__ import annotations

import argparse

from parsrag import FakeLLM, HashingEmbedding
from parsrag.eval import evaluate_retrieval
from parsrag.eval.benchmark import index_documents, load_fixture_benchmark
from parsrag.eval.dataset import QAItem
from parsrag.eval.generation import LLMJudge, faithfulness
from parsrag.eval.pipeline import build_course_pipeline, evaluate_pipeline
from parsrag.eval.stats import bootstrap_ci, paired_bootstrap
from parsrag.llm import Message
from parsrag.rerank import LexicalReranker, RerankingRetriever
from parsrag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, PersianTokenizer

SETS = {"original": "qa.jsonl", "paraphrase": "qa_paraphrase.jsonl"}


def reciprocal_ranks(qa: list[QAItem], retriever: object) -> list[float]:
    rep = evaluate_retrieval(
        qa,
        lambda q, k: retriever.search(q, k),
        (1,),
        level="passage",
        depth=10,  # type: ignore[attr-defined]
    )
    return [1.0 / q.rank if q.rank else 0.0 for q in rep.per_question]


def comparisons() -> None:
    docs, _ = load_fixture_benchmark()
    store = index_documents(docs, HashingEmbedding())
    bm25 = BM25Retriever(store.chunks)
    stem = BM25Retriever(store.chunks, tokenizer=PersianTokenizer(stem=True))
    dense = DenseRetriever(store)
    hybrid = HybridRetriever([stem, dense])
    systems = {
        "BM25": bm25,
        "BM25 + stem": stem,
        "dense": dense,
        "hybrid": hybrid,
        "hybrid + rerank": RerankingRetriever(hybrid, LexicalReranker()),
    }
    claims = [
        ("lesson 07: light stemming helps BM25", "BM25", "BM25 + stem"),
        ("lesson 07: hybrid beats dense alone", "dense", "hybrid"),
        ("lesson 07: hybrid beats BM25 + stem alone", "BM25 + stem", "hybrid"),
        ("lesson 08: reranking helps", "hybrid", "hybrid + rerank"),
    ]
    print("## 1. MRR@10 with 95% bootstrap intervals\n")
    print("| system | original | paraphrase |")
    print("|---|---|---|")
    rr = {
        s: {
            name: reciprocal_ranks(load_fixture_benchmark(qa_file=f)[1], r)
            for name, r in systems.items()
        }
        for s, f in SETS.items()
    }
    for name in systems:
        cells = [str(bootstrap_ci(rr[s][name])) for s in SETS]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n## 2. Paired bootstrap: is the difference real? (ΔMRR, 95% CI)\n")
    print("| claim | original Δ | paraphrase Δ |")
    print("|---|---|---|")
    for label, a, b in claims:
        cells = []
        for s in SETS:
            ci = paired_bootstrap(rr[s][a], rr[s][b])
            verdict = "real" if ci.low > 0 else ("worse" if ci.high < 0 else "not shown")
            cells.append(f"{ci.mean:+.3f} [{ci.low:+.3f}, {ci.high:+.3f}] {verdict}")
        print(f"| {label} | " + " | ".join(cells) + " |")


def ci_width_vs_size() -> None:
    print("\n## 3. Interval width vs. number of questions (R@1 = 0.9 held fixed)\n")
    print("| questions | 95% CI of R@1 | width |")
    print("|---|---|---|")
    for n in (20, 59, 200, 1000):
        hits = [1.0] * round(0.9 * n) + [0.0] * (n - round(0.9 * n))
        ci = bootstrap_ci(hits)
        print(f"| {n} | [{ci.low:.3f}, {ci.high:.3f}] | {ci.high - ci.low:.3f} |")


def verbosity_bias_demo() -> None:
    """A *simulated* judge that likes long answers, next to the heuristic."""
    print("\n## 4. Judge bias, simulated: a judge that rewards length\n")

    def length_loving_judge(messages: list[Message]) -> str:
        answer = messages[-1].content.split("پاسخ سیستم:")[1].split("\n")[0]
        return f"امتیاز: {min(10, len(answer) // 15)}\nدلیل: پاسخ کامل و مفصل است."

    judge = LLMJudge(FakeLLM(handler=length_loving_judge))
    context = ["هزینهٔ پیک در تهران 200 هزار تومان است."]
    answers = {
        "short, correct": "200 هزار تومان.",
        "long, padded with an unsupported claim": (
            "هزینهٔ پیک در تهران 200 هزار تومان است. ارسال با پیک در تمام شهرهای ایران "
            "در کمتر از دو ساعت انجام می‌شود."
        ),
    }
    print("| answer | heuristic faithfulness | length-biased judge |")
    print("|---|---|---|")
    for label, text in answers.items():
        h = faithfulness(text, context).score
        j = judge.faithfulness("هزینهٔ پیک در تهران چقدر است؟", text, context).score
        print(f"| {label} | {h:.2f} | {j:.2f} |")


def real_judge(model: str, base_url: str) -> None:
    from parsrag.llm import OpenAICompatibleLLM

    judge = LLMJudge(OpenAICompatibleLLM(model, base_url=base_url))
    generator = build_course_pipeline()
    print(f"\n## 5. LLM-as-judge: {model}\n")
    print(
        "| set | heuristic faithfulness | judge faithfulness | agreement (both ≥ 0.5 or both < 0.5) |"
    )
    print("|---|---|---|---|")
    for s, f in SETS.items():
        report = evaluate_pipeline(
            generator, load_fixture_benchmark(qa_file=f)[1], name=s, judge=judge
        )
        pairs = [
            (r.faithfulness, r.judge_faithfulness)
            for r in report.records
            if r.faithfulness is not None and r.judge_faithfulness is not None
        ]
        agree = sum((h >= 0.5) == (j >= 0.5) for h, j in pairs) / max(len(pairs), 1)
        print(
            f"| {s} | {report.metrics.get('faithfulness', 0):.3f} "
            f"| {report.metrics.get('judge_faithfulness', 0):.3f} | {agree:.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge", help="judge model name on an OpenAI-compatible server")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    args = parser.parse_args()
    comparisons()
    ci_width_vs_size()
    verbosity_bias_demo()
    if args.judge:
        real_judge(args.judge, args.base_url)


if __name__ == "__main__":
    main()

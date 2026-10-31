"""End-to-end evaluation of a grounded RAG pipeline (lesson 10).

:func:`evaluate_pipeline` runs a :class:`~parsrag.generate.GroundedGenerator`
over a QA set and reports, per question and in aggregate:

* retrieval — passage-level R@1, R@3, MRR@10 of the generator's retriever;
* answers — token F1 against the reference (abstaining on an answerable
  question scores 0), and F1 over the questions that were answered;
* citations — precision (cited chunks that contain the evidence) and hit rate;
* abstention — precision / recall / F1 of «نمی‌دانم» (positive = unanswerable);
* faithfulness — the heuristic of :mod:`parsrag.eval.generation`, and
  optionally an :class:`~parsrag.eval.generation.LLMJudge`;
* 95% bootstrap intervals for the per-question metrics.

:func:`build_course_pipeline` assembles the pipeline chosen in lessons 04–09,
offline and deterministic, so CI can gate on it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parsrag.eval.dataset import QAItem
from parsrag.eval.generation import (
    LLMJudge,
    citation_hit,
    citation_precision,
    faithfulness,
    token_f1,
)
from parsrag.eval.retrieval import context_text, evaluate_retrieval
from parsrag.eval.stats import Interval, bootstrap_ci
from parsrag.generate import GroundedGenerator, abstention_stats

# --------------------------------------------------------------- the report


@dataclass
class QuestionRecord:
    id: str
    question: str
    answerable: bool
    answer: str
    abstained: bool
    f1: float | None  # None for unanswerable questions
    citation_precision: float | None
    citation_hit: bool | None
    faithfulness: float | None  # None when the system abstained
    judge_faithfulness: float | None = None


#: Metrics that get a bootstrap interval: (metric name, record attribute, filter).
_PER_QUESTION = (
    ("answer_f1", "f1", "answerable"),
    ("answered_f1", "f1", "answered"),
    ("citation_precision", "citation_precision", "cited"),
    ("faithfulness", "faithfulness", "answered_any"),
    ("judge_faithfulness", "judge_faithfulness", "judged"),
)


@dataclass
class PipelineReport:
    name: str
    n_questions: int
    metrics: dict[str, float]
    intervals: dict[str, Interval] = field(default_factory=dict)
    records: list[QuestionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_questions": self.n_questions,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "intervals": {
                k: [round(i.low, 4), round(i.high, 4)] for k, i in self.intervals.items()
            },
            "records": [r.__dict__ for r in self.records],
        }

    def to_markdown(self) -> str:
        lines = [
            f"### {self.name} ({self.n_questions} questions)",
            "",
            "| metric | value | 95% CI |",
            "|---|---|---|",
        ]
        for key, value in self.metrics.items():
            ci = self.intervals.get(key)
            cell = f"[{ci.low:.3f}, {ci.high:.3f}]" if ci else ""
            lines.append(f"| {key} | {value:.3f} | {cell} |")
        return "\n".join(lines)


def _subset(records: Sequence[QuestionRecord], which: str) -> list[QuestionRecord]:
    if which == "answerable":
        return [r for r in records if r.answerable]
    if which == "answered":
        return [r for r in records if r.answerable and not r.abstained]
    if which == "cited":
        return [r for r in records if r.citation_precision is not None]
    if which == "judged":
        return [r for r in records if r.judge_faithfulness is not None]
    return [r for r in records if not r.abstained]  # answered_any


def evaluate_pipeline(
    generator: GroundedGenerator,
    items: Sequence[QAItem],
    *,
    name: str = "pipeline",
    judge: LLMJudge | None = None,
    retrieval_depth: int = 10,
    n_resamples: int = 2000,
) -> PipelineReport:
    """Answer every question with ``generator`` and score the answers."""
    records: list[QuestionRecord] = []
    for item in items:
        ans = generator.answer(item.question)
        chunks = {r.chunk.id: r.chunk for r in ans.sources}
        cited = [chunks[c.chunk_id] for c in ans.citations if c.chunk_id in chunks]
        contexts = [context_text(r.chunk) for r in ans.sources]
        faith = None if ans.abstained else faithfulness(ans.text, contexts).score
        judged = None
        if judge is not None and not ans.abstained:
            judged = judge.faithfulness(item.question, ans.text, contexts).score
        records.append(
            QuestionRecord(
                id=item.id,
                question=item.question,
                answerable=item.answerable,
                answer=ans.answer,
                abstained=ans.abstained,
                f1=(0.0 if ans.abstained else token_f1(ans.text, item.answer))
                if item.answerable
                else None,
                citation_precision=citation_precision(cited, item) if item.answerable else None,
                citation_hit=citation_hit(cited, item) if item.answerable and cited else None,
                faithfulness=faith,
                judge_faithfulness=judged,
            )
        )

    retrieval = evaluate_retrieval(
        items,
        lambda q, k: generator.retriever.search(q, k),
        (1, 3),
        level="passage",
        depth=retrieval_depth,
    )
    abst = abstention_stats([r.abstained for r in records], [r.answerable for r in records])
    metrics: dict[str, float] = {
        "retrieval_r@1": retrieval.recall[1],
        "retrieval_r@3": retrieval.recall[3],
        "retrieval_mrr": retrieval.mrr,
    }
    intervals: dict[str, Interval] = {}
    ranks = [q.rank for q in retrieval.per_question]
    if ranks:
        intervals["retrieval_r@1"] = bootstrap_ci(
            [1.0 if r == 1 else 0.0 for r in ranks], n_resamples=n_resamples
        )
        intervals["retrieval_mrr"] = bootstrap_ci(
            [1.0 / r if r else 0.0 for r in ranks], n_resamples=n_resamples
        )
    for key, attr, which in _PER_QUESTION:
        values = [getattr(r, attr) for r in _subset(records, which)]
        values = [float(v) for v in values if v is not None]
        if values:
            ci = bootstrap_ci(values, n_resamples=n_resamples)
            metrics[key], intervals[key] = ci.mean, ci
    hits = [r.citation_hit for r in _subset(records, "answered")]
    if hits:
        metrics["citation_hit"] = sum(bool(h) for h in hits) / len(hits)
    metrics.update(
        {
            "abstention_precision": abst.precision,
            "abstention_recall": abst.recall,
            "abstention_f1": abst.f1,
            "answered_rate": abst.tn / (abst.tn + abst.fp) if abst.tn + abst.fp else 0.0,
        }
    )
    return PipelineReport(name, len(items), metrics, intervals, records)


# --------------------------------------------------------- the course pipeline


def build_course_pipeline(
    calibration: Sequence[QAItem] | None = None, fixtures: str | Path | None = None
) -> GroundedGenerator:
    """The pipeline chosen in lessons 04–09, fully offline.

    recursive chunks (400, heading context, normalised) → RRF(BM25 + light
    stemming, HashingEmbedding) → lexical rerank of 20 → GroundedGenerator(k=4)
    with an extractive, citing ``FakeLLM`` and a BM25 abstention threshold
    calibrated on ``calibration`` (default: ``qa.jsonl``).
    """
    from parsrag import FakeLLM, HashingEmbedding
    from parsrag.eval.benchmark import FIXTURES, index_documents, load_fixture_benchmark
    from parsrag.generate import calibrate_threshold, retriever_confidence
    from parsrag.rerank import LexicalReranker, RerankingRetriever
    from parsrag.retrieval import (
        PERSIAN_STOPWORDS,
        BM25Retriever,
        DenseRetriever,
        HybridRetriever,
        PersianTokenizer,
    )

    docs, qa = load_fixture_benchmark(fixtures or FIXTURES)
    calibration = qa if calibration is None else calibration
    store = index_documents(docs, HashingEmbedding())
    bm25 = BM25Retriever(store.chunks, tokenizer=PersianTokenizer(stem=True))
    retriever = RerankingRetriever(
        HybridRetriever([bm25, DenseRetriever(store)]), LexicalReranker(), fetch_k=20
    )
    confidence = retriever_confidence(bm25)
    threshold, _ = calibrate_threshold(
        [confidence(q.question, []) for q in calibration], [q.answerable for q in calibration]
    )
    llm = FakeLLM(mode="extractive", cite=True, ignore=PERSIAN_STOPWORDS)
    return GroundedGenerator(retriever, llm, k=4, min_score=threshold, confidence=confidence)


def save_report(
    reports: Sequence[PipelineReport], json_path: str | Path | None, md_path: str | Path | None
) -> None:
    """Write the reports as JSON (machine-readable) and/or Markdown (for humans)."""
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(
            json.dumps([r.to_dict() for r in reports], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if md_path:
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        body = "\n\n".join(r.to_markdown() for r in reports)
        Path(md_path).write_text(
            "# Evaluation report — course pipeline\n\n"
            "Generated by `python -m parsrag.eval run`. Offline: FakeLLM (extractive) + "
            "HashingEmbedding.\n\n" + body + "\n",
            encoding="utf-8",
        )


__all__ = [
    "PipelineReport",
    "QuestionRecord",
    "build_course_pipeline",
    "evaluate_pipeline",
    "save_report",
]

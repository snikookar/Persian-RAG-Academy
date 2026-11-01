# 10 — Evaluation

**Persian lesson:** [ارزیابی](../lessons/10-evaluation.md) · **Module:** `parsrag.eval`

## What you build

An evaluation suite for the whole pipeline: token F1 with Persian normalisation, citation
precision, abstention precision/recall, a claim-level faithfulness heuristic (with a rule for
numbers) and an LLM judge with Persian prompts; bootstrap and paired-bootstrap confidence
intervals; `evaluate_pipeline` with JSON and Markdown reports; and an evaluation gate,
`python -m parsrag.eval gate`, that fails CI when a metric drops below the committed baseline.

## Key ideas

- **Normalise before comparing answers.** «۲۰۰ هزار تومان» and «200 هزار تومان است [1].» are the
  same answer: citations, digit systems, thousand separators, ZWNJ and a few function words are
  removed before computing SQuAD-style token F1.
- **Faithful is not correct.** The extractive stand-in copies sentences from the context, so it
  is perfectly faithful, yet its answers only partly match the references.
- **Report intervals, compare in pairs.** With tens of questions, confidence intervals are wide.
  To compare two systems, resample *questions* (paired bootstrap) so that "hard for both" is not
  counted as noise. "Not shown" means "we don't know", not "no effect".
- **Keep a held-out set untouched**, and gate CI on deterministic offline runs.

## Measured results

Course pipeline (`python -m parsrag.eval run`; 95% bootstrap intervals; original set 66
questions, paraphrase set 51):

| Metric | original | paraphrase |
|---|---|---|
| retrieval R@1 | 0.881 [0.797, 0.949] | 0.370 [0.239, 0.500] |
| retrieval MRR@10 | 0.941 [0.898, 0.975] | 0.487 [0.373, 0.600] |
| answer token F1 (abstain = 0) | 0.276 [0.210, 0.344] | 0.016 [0.005, 0.031] |
| citation precision | 0.842 [0.737, 0.930] | 0.800 [0.500, 1.000] |
| faithfulness (heuristic) | 1.000 | 1.000 |
| abstention F1 | 0.714 | 0.178 |
| answered rate (answerable questions) | 0.966 | 0.217 |

Earlier claims under a paired bootstrap (ΔMRR [95%]):

| Claim | original | paraphrase |
|---|---|---|
| lesson 07: light stemming helps BM25 | +0.021 [-0.012, +0.059] not shown | +0.017 [-0.040, +0.082] not shown |
| lesson 07: hybrid beats dense alone | +0.084 [+0.028, +0.148] real | +0.109 [+0.031, +0.196] real |
| lesson 07: hybrid beats BM25 + stemming | +0.032 [-0.005, +0.074] not shown | +0.007 [-0.056, +0.067] not shown |
| lesson 08: reranking helps | +0.016 [-0.014, +0.047] not shown | +0.040 [-0.028, +0.114] not shown |

Only one of four "improvements" is defensible with this many questions. How the interval
narrows for a fixed R@1 of 0.9: 20 questions [0.750, 1.000] (width 0.250), 59 [0.814, 0.966]
(0.153), 200 [0.855, 0.940] (0.085), 1000 [0.880, 0.918] (0.038) — roughly `1/√n`, so halving
uncertainty takes four times the questions.

The gate compares every metric with `docs/eval/baseline.json` (tolerance 0.02). LLM-judge
numbers on a real model are left for you (`examples/evaluation_report.py --judge ...`).

## Takeaway

Measure answers, citations and abstention, not just retrieval; always report intervals; and
let a deterministic gate protect quality in CI.

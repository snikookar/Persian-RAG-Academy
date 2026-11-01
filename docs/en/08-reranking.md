# 08 — Reranking

**Persian lesson:** [بازرتبه‌بندی](../lessons/08-reranking.md) · **Module:** `parsrag.rerank`

## What you build

A `Reranker` protocol and three rerankers — an offline `LexicalReranker` (idf-weighted query
coverage with typo and prefix matching, optional proximity and first-stage prior), a lazy
`CrossEncoderReranker` (sentence-transformers) and an `LLMReranker` that asks a model for a 0–10
relevance score with a Persian prompt and a forgiving parser — plus `RerankingRetriever`:
retrieve many, rerank, keep few.

## Key ideas

- **Two stages.** The first stage must be fast and recall-oriented (fetch 20 candidates); the
  second can be slow and precise because it only scores those candidates.
- **Cost grows linearly with candidates.** A pointwise LLM reranker makes one call per
  candidate. A small cross-encoder is usually the right trade-off in production.
- **Tune on one set, check on another.** Several reranker features looked good on paraphrases
  but not on the tuning set; choosing them would have been overfitting to the test set.

## Measured results

Passage level, on top of the lesson 07 default (`python examples/reranking.py`), MRR@10:

| Setting | orig R@1 | orig R@3 | orig MRR@10 | para R@1 | para R@3 | para MRR@10 |
|---|---|---|---|---|---|---|
| no reranking | 0.864 | 0.983 | 0.925 | 0.326 | 0.500 | 0.447 |
| lexical reranker (default) | 0.881 | 1.000 | 0.941 | 0.370 | 0.543 | 0.487 |
| lexical reranker + idf weighting | 0.847 | 0.966 | 0.908 | 0.261 | 0.478 | 0.414 |

Proximity and prior weights (proximity / prior): 0.0 / 0.0 is the default (orig MRR 0.941);
0.3 / 0.2 reaches the best paraphrase MRR (0.504) but was not chosen, because it was not better
on the original set (0.936); proximity alone hurts (0.3 / 0.0: orig R@1 0.797; 0.6 / 0.0:
0.678). Heading context puts common query words side by side, which fools proximity.

Candidates and cost: with `fetch_k` 10, 20 and 40 the original R@1 stays 0.881 while time per
question grows roughly linearly (about 5, 9 and 17 ms; about 2 ms without reranking). For an
LLM reranker the script estimates cost without calling any model:

| fetch_k | LLM calls per question | input tokens per question (≈) |
|---|---|---|
| 10 | 10 | 2,865 |
| 20 | 20 | 5,713 |
| 40 | 40 | 11,381 |

Lesson 10 later tests the reranking gain with a paired bootstrap: +0.016 [-0.014, +0.047] on
the original set, so it is not shown statistically.

Cross-encoder and LLM reranking numbers on real models are left for you to measure with the
script's flags; none were invented.

## Takeaway

Retrieve many, rerank few. A cheap reranker gives a small, plausible gain; measure the cost per
question before reaching for an LLM.

# 07 — Hybrid search

**Persian lesson:** [جست‌وجوی ترکیبی](../lessons/07-hybrid-search.md) · **Module:** `parsrag.retrieval`

## What you build

A Persian tokenizer (normalisation, ZWNJ-aware, optional stop words, optional splitting at the
ZWNJ, light suffix stemming), Okapi BM25 written from scratch, a dense retriever, and two ways to
combine them: Reciprocal Rank Fusion (RRF) and weighted score fusion, behind one `Retriever`
protocol shared with the vector stores.

## Key ideas

- **BM25 and dense retrieval make different mistakes.** BM25 wants the exact word; a dense
  model wants similar meaning (or, for the hashing stand-in, similar characters). Fusing them
  recovers the strengths of both.
- **RRF needs no tuning.** It combines ranks, not scores (`1 / (60 + rank)` per list), so the
  scales of BM25 and cosine never have to be matched. Weighted fusion needs normalised scores
  and weights, and is sensitive to them.
- **A second question set.** `qa_paraphrase.jsonl` asks the same facts with synonyms, colloquial
  Persian, loanwords and a few typos (46 answerable questions). It shows what lexical methods
  cannot do.

## Measured results

Passage level, course chunking (recursive 400 + headings), dense = `HashingEmbedding`
(`python examples/hybrid_search.py`):

| Retriever | orig R@1 | orig R@3 | orig MRR | para R@1 | para R@3 | para MRR |
|---|---|---|---|---|---|---|
| BM25 | 0.780 | 0.949 | 0.872 | 0.370 | 0.435 | 0.434 |
| BM25 + light stemming | 0.814 | 0.983 | 0.893 | 0.370 | 0.478 | 0.448 |
| BM25 + ZWNJ splitting | 0.763 | 0.949 | 0.858 | 0.391 | 0.500 | 0.474 |
| BM25 + stop words removed | 0.814 | 0.949 | 0.892 | 0.370 | 0.457 | 0.430 |
| dense (hashing) | 0.729 | 0.983 | 0.841 | 0.196 | 0.391 | 0.349 |
| RRF (BM25 + dense) | 0.831 | 0.983 | 0.905 | 0.370 | 0.478 | 0.467 |
| RRF (BM25 + stemming + dense) | 0.864 | 0.983 | 0.925 | 0.326 | 0.500 | 0.453 |
| weighted 0.5 / 0.5 | 0.881 | 1.000 | 0.932 | 0.370 | 0.457 | 0.455 |
| weighted 0.3 / 0.7 (dense heavier) | 0.864 | 1.000 | 0.924 | 0.304 | 0.478 | 0.421 |
| weighted 0.7 / 0.3 (BM25 heavier) | 0.847 | 0.966 | 0.910 | 0.413 | 0.435 | 0.465 |

- **Fusion works:** on the original questions RRF beats both single methods (MRR 0.905 vs 0.872
  and 0.841).
- **Light stemming is a small help, not a proven one:** 0.872 → 0.893 MRR. Lesson 10's paired
  bootstrap puts the improvement at +0.021 [-0.012, +0.059]: the interval contains zero.
- **Weights are sensitive:** 0.5/0.5 edges RRF on the original set within noise (0.932 vs 0.925),
  but 0.3/0.7 drops paraphrase R@1 to 0.304. RRF is close to the best without any tuning.
- **The lexical gap:** every method roughly halves on paraphrases (MRR ≈ 0.45). No tokenizer
  trick connects «کرایه» to «هزینه». The hashing "dense" model is lexical underneath and is the
  worst on paraphrases (0.349).

The course default from here on is RRF over BM25 with light stemming and dense retrieval, 20
candidates from each.

## Takeaway

Use hybrid retrieval with RRF as a robust default, test on paraphrased questions, and be
suspicious of small gains: on sets this size, most of them are not statistically shown.

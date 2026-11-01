# 09 — Grounded generation

**Persian lesson:** [تولید مستند](../lessons/09-grounded-generation.md) · **Module:** `parsrag.generate`

## What you build

A `GroundedGenerator` that shows the model numbered context blocks, asks for `[n]` citations,
maps each citation back to chunk metadata (document, section, page and exact character span),
validates citations (invalid numbers, uncited answers), recognises refusals («نمی‌دانم») and
abstains *before* calling the model when retrieval confidence is below a calibrated threshold.

## Key ideas

- **Citations are data.** `[2]` becomes a `Citation` with `doc_id`, `section`, `page`, `start`
  and `end`, so a UI can show the exact source passage, and a citation to a block that does not
  exist (`[7]` when only four were given) is caught.
- **Two ways to say "I don't know".** A *retrieval threshold* skips the model when even the best
  passage is weak (cheap and deterministic); an *instruction-level refusal* lets the model decide
  (smarter, but model-dependent).
- **Which score?** RRF scores come from ranks only (always about 0.03) and are useless as a
  threshold. `retriever_confidence` uses the best score of another retriever, and
  `calibrate_threshold` picks the threshold with the best abstention F1 on a development set.

## Measured results

Positive class = "should say I don't know" (7 unanswerable original questions, 5 paraphrases).
Threshold chosen on the original set, applied unchanged to paraphrases:

| Confidence signal | threshold | orig P | orig R | orig F1 | para P | para R | para F1 |
|---|---|---|---|---|---|---|---|
| top BM25 score | 8.423 | 0.71 | 0.71 | 0.71 | 0.10 | 0.80 | 0.18 |
| top cosine (hashing) | 0.404 | 0.67 | 0.29 | 0.40 | 0.07 | 0.60 | 0.13 |
| top reranker score | 0.904 | 0.21 | 1.00 | 0.35 | 0.10 | 1.00 | 0.18 |

Threshold vs (simulated) instruction-level refusal; "answered" = answerable questions that got
an answer:

| Setting | orig F1 | orig answered | para F1 | para answered |
|---|---|---|---|---|
| base (`min_overlap=1`) | 0.00 | 59/59 | 0.25 | 44/46 |
| instruction only (`min_overlap=3`) | 0.47 | 53/59 | 0.20 | 6/46 |
| threshold only (BM25) | 0.71 | 57/59 | 0.18 | 10/46 |
| threshold + instruction | 0.53 | 52/59 | 0.19 | 3/46 |

- The raw top BM25 score separates answerable from unanswerable questions best; reranker
  coverage is worst, because the unanswerable questions reuse corpus words.
- **The threshold does not transfer:** on paraphrases the system answers only 10 of 46
  answerable questions. A lexical score measures "worded like the documents", not "the answer is
  in the documents".
- These numbers describe an extractive stand-in model, not a real LLM, which may refuse far
  better at the instruction level — measure it.

## Takeaway

Make citations verifiable, calibrate abstention on a development set, and expect a lexical
threshold to over-abstain on colloquial questions.

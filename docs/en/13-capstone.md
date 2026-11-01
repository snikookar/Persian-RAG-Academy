# 13 — Capstone

**Persian lesson:** [پروژهٔ پایانی](../lessons/13-capstone.md) · **Modules:** `parsrag.rag`, `parsrag.cli`

## What you build

`PersianRAG`, one object for the whole course pipeline with every part replaceable (embedding,
LLM, chunker, store, reranker, `k`, guard), the `parsrag` command line (`init`, `ingest`, `ask`,
`eval`, `serve`), a right-to-left web UI with clickable citations, and your own question sets
checked by `validate_qa`, an evaluation gate and a production-readiness checklist.

## Key ideas

- **Defaults from measurements.** Normalisation (02), recursive 400-character chunks with
  heading context (04), RRF over BM25 with light stemming and dense retrieval (07), lexical
  reranking of 20 candidates (08), four passages with citations and a calibrated BM25 threshold
  (09).
- **Configuration, not code.** `parsrag.toml` describes documents, index folder, chunking,
  retrieval, embedding, LLM and security; API keys are referenced by environment variable name
  only. From 0.2.0, `parsrag init --preset ollama` or `--preset openai-compatible` writes a
  ready-to-edit file, and the index is saved, loaded and updated incrementally.
- **Calibrate, or it never says "I don't know".** An uncalibrated server answered an
  out-of-scope question (a flight price) with an unrelated passage. Calibrate on a development
  set, evaluate on a separate test set.
- **Your own questions.** `validate_qa` warns about questions whose evidence is not found in
  the named document, before you trust any number.

## Measured results

The offline course pipeline on the 13-question example file
(`lessons/13-capstone/my_questions.example.jsonl`):

| Metric | value | 95% interval |
|---|---|---|
| retrieval R@1 (10 answerable questions) | 0.900 | [0.700, 1.000] |
| answer F1 | 0.336 | [0.184, 0.493] |
| citation precision | 1.000 | |
| abstention precision / recall / F1 | 0.500 / 0.667 / 0.571 | |
| answered rate (answerable questions) | 0.800 | |

The intervals are wide: with 13 questions, R@1 is "somewhere between 0.7 and 1". Two colloquial
questions were not answered (the gap of lessons 10–11), and one unanswerable question got a
wrong answer because its words appear in an installation document — exactly the kind of failure
to look for on your own documents.

Gate in CI? Yes with the offline models (deterministic and free), as this repository does. With
a real model, not on every pull request (paid, non-deterministic, keys in CI): run it nightly
with repetitions and intervals instead.

## Takeaway

Ship the pipeline as one configurable object with a CLI, an API and a UI — and treat
calibration and evaluation on your own questions as part of the product.

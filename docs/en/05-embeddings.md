# 05 — Embeddings

**Persian lesson:** [برداری‌سازی](../lessons/05-embeddings.md) · **Modules:** `parsrag.embeddings`, `parsrag.eval`

## What you build

A retrieval benchmark — Recall@k, MRR and nDCG@k at document and passage level
(`parsrag.eval.retrieval`) and `benchmark_embedding` for any `Embedding` — plus a TF-IDF baseline
(word and character n-grams), a cosine helper and an OpenAI-compatible embedding client. A
script and a Colab notebook score real models with exactly the same code.

## Key ideas

- **One `Embedding` protocol.** Hashing, TF-IDF, sentence-transformers and any
  OpenAI-compatible endpoint (including Ollama) are interchangeable, so models are compared on
  the same chunks, normalisation and questions.
- **Persian morphology hurts whole-word matching.** «کتاب» / «کتاب‌ها» or «امانت» / «امانتی» are
  different words to word-level TF-IDF; character n-grams partly fix this.
- **Lexical is not semantic.** All offline methods here are lexical. The questions reuse
  document wording, which favours them; users say «کرایهٔ پیک» where the document says
  «هزینهٔ ارسال». Neural embeddings should close that gap, and you must measure it yourself.
- **Normalise queries the same way as documents.** Same function, both sides.

## Measured results

Offline models, 59 answerable questions (`python examples/benchmark_embeddings.py`):

| model | dim | passage R@1 | passage R@3 | passage MRR | passage nDCG@3 | doc R@1 |
|---|---|---|---|---|---|---|
| hashing | 1024 | 0.729 | 0.983 | 0.841 | 0.876 | 0.847 |
| tfidf-word | 1195 | 0.729 | 0.932 | 0.836 | 0.850 | 0.797 |
| tfidf-char | 5957 | 0.797 | 0.983 | 0.883 | 0.908 | 0.831 |

Questions typed on an Arabic keyboard (Arabic YEH/KAF and digits), passage R@3 / MRR:

| setting | hashing | tfidf-word | tfidf-char |
|---|---|---|---|
| clean questions, both sides normalised | 0.983 / 0.841 | 0.932 / 0.836 | 0.983 / 0.883 |
| Arabic-keyboard questions, both sides normalised | 0.983 / 0.841 | 0.932 / 0.836 | 0.983 / 0.883 |
| Arabic-keyboard questions, documents only | 0.847 / 0.787 | 0.797 / 0.680 | 0.932 / 0.860 |
| Arabic-keyboard questions, no normalisation | 0.814 / 0.761 | 0.746 / 0.611 | 0.847 / 0.759 |

Normalising both sides makes messy questions score exactly like clean ones; normalising only
the index costs word TF-IDF 0.156 MRR.

On the paraphrase set (46 answerable questions, different wording) passage MRR drops to 0.349
(hashing), 0.435 (word TF-IDF) and 0.472 (char TF-IDF): lexical methods roughly halve.

The lesson's real-model table (multilingual E5, BGE-M3, MiniLM, ...) is **intentionally empty**
until someone contributes an actual run of the notebook; no real-model number was invented.

## Takeaway

Benchmark embeddings on passage-level metrics, with your own questions, including paraphrased
ones. Character n-grams are a strong offline baseline; semantic matching needs a real model.

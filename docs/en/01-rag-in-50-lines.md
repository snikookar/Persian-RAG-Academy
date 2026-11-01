# 01 — RAG in 50 lines

**Persian lesson:** [RAG در ۵۰ خط](../lessons/01-rag-in-50-lines.md) · **Module:** `parsrag.naive_rag`

## What you build

The smallest complete Retrieval-Augmented Generation pipeline, in about fifty lines: load the
Markdown files of the course corpus, cut them into fixed windows, embed the windows, retrieve
the most similar ones for a question and let a model answer from them.

## Key ideas

- **The five steps.** Every RAG system does the same five things; the rest of the course
  improves one step at a time:

  | Step | In this lesson | Improved in |
  |---|---|---|
  | Load | read `.md` files | lesson 03 (PDF, DOCX, HTML, right-to-left problems) |
  | Chunk | fixed 400-character windows, 50 overlap | lesson 04 (sentence- and structure-aware) |
  | Embed | `HashingEmbedding` | lesson 05 (benchmarking real models) |
  | Retrieve | cosine similarity in memory | lessons 06–08 (FAISS, hybrid search, reranking) |
  | Generate | `FakeLLM` in extractive mode | lesson 09 (citations, honest "I don't know") |

- **A baseline is a tool.** The naive pipeline is deliberately bad, and the lesson measures
  *how* bad, so that every later lesson can show its improvement with a number instead of a
  claim.
- **Read the failures.** The lesson walks through two broken answers: one cut in the middle of
  a word (fixed windows ignore word and sentence boundaries) and one where the right document is
  not even in the top three. The second happens because that document was typed with Arabic
  YEH and KAF: for a computer «كتاب» and «کتاب» are different words. That is lesson 02.

## Data

The corpus is 25 original Persian documents about a fictional shop («کهربا») and a fictional
library («کتابخانهٔ مهرگان»), deliberately messy, with 66 questions in `qa.jsonl` (59 answerable,
each with a verbatim evidence span, and 7 unanswerable).

## Measured results

On the 59 answerable questions, document-level hit rate with 400-character chunks:

| Pipeline | hit@1 | hit@3 |
|---|---|---|
| Lesson 01 naive (chunk 400) | 0.763 | 0.864 |

So the right document is ranked first for 76% of the questions and is in the top three for
86%. With 59 questions, one question moves these numbers by about 1.7 points.

## Takeaway

Start with the simplest pipeline that runs end to end, measure it, and look at individual
failures. They tell you what to fix next far better than intuition.

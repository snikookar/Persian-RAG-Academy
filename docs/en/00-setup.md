# 00 — Setup

**Persian lesson:** [آماده‌سازی محیط](../lessons/00-setup.md) · **Module:** `parsrag.doctor`

## What you build

A working development environment and a small health check, `parsrag.doctor`, that answers
three questions: is the Python version recent enough (3.10+), which optional extras are
installed, and does a tiny end-to-end RAG run succeed on this machine? You also meet the
grader that checks every later lesson.

## Key ideas

- **Offline first.** The core package depends only on `pydantic` and `numpy`. Everything else
  (PDF/DOCX loaders, FAISS, pgvector, LangGraph, FastAPI, sentence-transformers, an HTTP client
  for OpenAI-compatible servers) is an *optional extra* that is imported lazily. Tests, the
  grader and CI never download a model, never call a paid API and never need a key.
- **Two stand-ins that make this possible.** `FakeLLM` is a deterministic language model with
  canned, echo, extractive and handler modes; in extractive mode it copies the sentence of the
  context that overlaps most with the question, which is enough to test a pipeline end to end.
  `HashingEmbedding` turns text into a fixed-size vector by hashing words and character 2–4-grams
  (CRC32), so "dense" retrieval works without any model. Both are honest about what they are:
  lexical tools, useful for tests and demos, not for real users.
- **The grader.** `python -m grader run 00` runs the lesson's tests against your starter code
  (loaded through `PARSRAG_LEARNER_PATH`) and prints a bilingual report with a hint for each
  failing check. `python -m grader check-solutions` runs the same tests against the reference
  solution, which is the library itself.
- **Real models are one setting away.** Any OpenAI-compatible server works, including Ollama on
  your own machine (free, no key). Keys are read from environment variables, never from code.

## The exercise

Implement three functions in the starter: `python_version_ok`, `extras_status` (which extras
are importable) and `smoke_test` (three sentences in an `InMemoryVectorStore`, find the nearest
one to a question, answer with `FakeLLM` in extractive mode).

## Measured results

This lesson has no benchmark. Its "result" is that `pytest`, the grader and
`python -m parsrag.doctor` pass on your machine, which every later lesson relies on.

## Takeaway

Make the whole pipeline testable without a network before plugging in any model. Every number
reported in the course is produced by this offline setup, and real-model numbers are left for
you to measure with the same tools.

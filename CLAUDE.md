# CLAUDE.md — Persian RAG Academy

## What this repo is
- A Persian, project-based course on RAG (`lessons/`) + the `parsrag` Python library (`src/parsrag/`).
- Every lesson builds one module of `parsrag`. The library is the reference solution.

## Hard rules (never break these)
1. **No paid API calls, ever.** Tests, CI, graders and examples must run with zero API keys and no network.
   Use `FakeLLM` and `HashingEmbedding` (deterministic, offline) everywhere in tests.
2. Real providers (OpenAI-compatible APIs, Ollama, sentence-transformers) are optional extras, never imported at package top level, never required by tests.
3. Do not download models or datasets in tests or CI.
4. All Persian sample text in `data/fixtures/` must be written originally for this repo. Never copy copyrighted text (news, books, websites).
5. Keep the core install lightweight: core deps only `pydantic` and `numpy`. Everything else goes in extras.

## Language
- Code, identifiers, docstrings, comments, commit messages, PR descriptions: **English**.
- Lesson explanations (`lessons/*/README.fa.md`): **Persian**, clear and friendly, with code snippets and Mermaid diagrams where helpful. Technical terms: Persian + English in parentheses on first use, e.g. «بازیابی (Retrieval)».

## Layout
```
src/parsrag/          library (llm.py, embeddings.py, normalize.py, loaders/, chunking.py, stores/, retrieval.py, rerank.py, generate.py, eval/, agent/, api/)
lessons/NN-slug/      README.fa.md, starter/, tests/   (solution = the parsrag module)
grader/               `python -m grader run NN --path <dir>` and `python -m grader check-solutions`
data/fixtures/        small original Persian corpus + QA eval set (JSONL)
tests/                library unit tests
examples/             runnable examples
docs/                 extra docs, images, charts
ROADMAP.md            lesson checklist + session log (update at the end of every session)
```

## Lesson plan (13 lessons)
00 setup · 01 RAG in 50 lines · 02 Persian normalization · 03 loading Persian documents (PDF/DOCX/HTML, RTL issues) ·
04 chunking Persian text · 05 embeddings & comparing models · 06 vector stores (in-memory → FAISS → pgvector) ·
07 hybrid search (Persian BM25 + dense + RRF) · 08 reranking · 09 grounded generation with citations & honest "I don't know" ·
10 evaluation (Recall@k, MRR, faithfulness, LLM-as-judge) · 11 agentic RAG with LangGraph · 12 production API (FastAPI, streaming, cache, Docker) ·
13 capstone: RAG over your own documents + UI

## Quality bar
- Python 3.10+, full type hints, `ruff` clean, `pytest` green, src layout, `hatchling` build.
- Every decision in a lesson is justified with a measurement on the fixtures when possible (show the numbers).
- Small, focused commits using Conventional Commits (`feat:`, `docs:`, `test:`...).
- Before ending a session: run `ruff check .`, `pytest`, `python -m grader check-solutions`; update ROADMAP.md session log.

## Extras (pyproject optional-dependencies)
openai, ollama, st (sentence-transformers), faiss, pgvector, agent (langgraph, langchain-core), api (fastapi, uvicorn), loaders (pypdf, python-docx, beautifulsoup4), all.

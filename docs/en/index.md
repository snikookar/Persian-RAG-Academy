# English lesson summaries

**Persian RAG Academy** is a free, project-based course on Retrieval-Augmented Generation for
Persian text, written in Persian. Along the way learners rebuild `parsrag`, an offline-first
Python library. These pages are short English summaries for international readers: what each
lesson builds, the key ideas, and the measured results with the same numbers as the Persian
lesson. They are not translations; every page links to the full Persian lesson.

!!! note "About the numbers"
    Every number comes from the course's own scripts on a small, synthetic Persian corpus (25
    documents, 66 original and 51 paraphrased questions) with offline stand-in models
    (`HashingEmbedding`, an extractive `FakeLLM`). They show how to measure, not how good a real
    model is. Real-model numbers are left for you to produce with the same tools; none were
    invented.

| # | Lesson | Module | What it is about |
|---|---|---|---|
| 00 | [Setup](00-setup.md) | `parsrag.doctor` | offline tooling, extras, grader |
| 01 | [RAG in 50 lines](01-rag-in-50-lines.md) | `parsrag.naive_rag` | the five steps and a baseline |
| 02 | [Persian normalization](02-normalization.md) | `parsrag.normalize` | Arabic letters, digits, ZWNJ |
| 03 | [Loading documents](03-loading-documents.md) | `parsrag.loaders` | broken PDF text and its repair |
| 04 | [Chunking](04-chunking.md) | `parsrag.chunking` | Persian sentences, heading context, tokens |
| 05 | [Embeddings](05-embeddings.md) | `parsrag.embeddings` | passage-level benchmarks |
| 06 | [Vector stores](06-vector-stores.md) | `parsrag.stores` | in-memory, FAISS, pgvector |
| 07 | [Hybrid search](07-hybrid-search.md) | `parsrag.retrieval` | BM25 + dense with RRF |
| 08 | [Reranking](08-reranking.md) | `parsrag.rerank` | retrieve many, rerank few |
| 09 | [Grounded generation](09-grounded-generation.md) | `parsrag.generate` | citations and "I don't know" |
| 10 | [Evaluation](10-evaluation.md) | `parsrag.eval` | answer metrics, bootstrap, CI gate |
| 11 | [Agentic RAG](11-agentic-rag.md) | `parsrag.agent` | LangGraph agent and its cost |
| 12 | [Production API](12-production-api.md) | `parsrag.api` | FastAPI, streaming, caching |
| 13 | [Capstone](13-capstone.md) | `parsrag.rag`, `parsrag.cli` | `PersianRAG`, CLI, web UI |
| 14 | [RAG security](14-rag-security.md) | `parsrag.security` | indirect prompt injection |

## Try the library

```bash
pip install "parsrag[loaders]"
parsrag init                       # offline preset; --preset ollama for local models
parsrag ingest docs/               # index a folder (saved; re-runs only embed what changed)
parsrag ask "هزینهٔ ارسال چقدر است؟"
```

The library reference, the CLI and the HTTP API are documented (in Persian) in the
[reference section](../reference/library.md); the code itself has English docstrings.

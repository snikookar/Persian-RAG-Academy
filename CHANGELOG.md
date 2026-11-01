# Changelog

All notable changes to `parsrag` and the Persian RAG Academy course are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **14 RAG security** (lesson): indirect prompt injection hidden in retrieved documents.
  `parsrag.security` — Persian/English injection detection on normalised text, sanitising
  of links, images, HTML, fake citations and delimiter look-alikes, sandboxed context with
  a random nonce, instruction-hierarchy prompt and reminder, post-generation checks
  (links removed; refuse uncited, unsupported, echoed or instruction-following answers),
  `InjectionGuard` and `GuardedRetriever`. Opt-in via `PersianRAG(guard=True)`,
  `[security]` in `parsrag.toml` or `PARSRAG_SECURITY` for the API; answers carry the
  problems found and refusals use `abstain_reason="unsafe"`.
- `FakeLLM(obey=...)`: simulated models that obey instructions found in their context
  (naive) or only outside the data markers (hierarchy-respecting), for offline measurement.
- `data/fixtures/attacks/`: 8 original attack documents, 3 benign hard negatives and
  `attacks.jsonl`, kept apart from the main corpus so its numbers are unchanged.
- `examples/injection_attacks.py` (offline attack-success, detection and cost tables) and
  `examples/injection_test.py` (the same attacks against any OpenAI-compatible model).
- `notebooks/benchmark_real_embeddings.ipynb`: a Colab notebook that benchmarks multilingual
  Hugging Face embedding models (and optionally any OpenAI-compatible endpoint) next to the
  offline baselines on both question sets; lesson 05 has an empty real-model results table
  for contributed runs. CI checks the notebook offline (`scripts/check_notebook.py`).
- The web UI bundles the Vazirmatn font (SIL OFL 1.1), served by the API at
  `GET /static/fonts/{name}`, so it renders the same offline.

### Changed

- Web UI: blue theme with consistent hover/focus/disabled states (WCAG AA), title-only
  header, Persian-only error messages (API error codes are translated in the page) and a
  Persian, keyboard-accessible file picker.
- Fixtures: the fictional library is renamed to «کتابخانهٔ مهرگان» (`mehregan-*`) and all
  prices are set to plausible late-2026 levels; every benchmark number, chart and the eval
  baseline are recomputed.

### Fixed

- Docs workflow: a running GitHub Pages deployment from `main` is no longer cancelled by the
  next push.

## [0.1.0] - not yet released

The first version: 14 lessons and the complete `parsrag` library. Prepared for PyPI; the
release date is set when it is published (see `docs/RELEASING.md`).

### Added — course

- **00 Setup:** environment check (`parsrag.doctor`), the offline tools `FakeLLM` and
  `HashingEmbedding`, the grader with bilingual hints.
- **01 RAG in 50 lines:** load → chunk → embed → retrieve → generate (`parsrag.naive_rag`).
- **02 Persian normalisation:** Arabic → Persian letters, digits, ZWNJ, diacritics, tatweel,
  presentation forms, with a measured ablation (`parsrag.normalize`).
- **03 Loading Persian documents:** TXT (cp1256), Markdown, HTML, DOCX and PDF loaders with
  section and page offsets; PDF repair for presentation forms, visual order, lost ZWNJ and bidi
  controls (`parsrag.loaders`).
- **04 Chunking:** Persian sentence splitter, fixed-size, recursive and sentence-window
  chunkers, heading context, token estimate (`parsrag.chunking`).
- **05 Embeddings:** Recall@k / MRR / nDCG at document and passage level, TF-IDF baseline,
  a benchmark script for real models (`parsrag.embeddings`, `parsrag.eval`).
- **06 Vector stores:** one protocol with filters and save/load; in-memory, FAISS
  (flat / HNSW / IVF) and pgvector (`parsrag.stores`).
- **07 Hybrid search:** Persian BM25 with light stemming, dense retrieval, RRF and weighted
  fusion; a second, paraphrased question set (`parsrag.retrieval`).
- **08 Reranking:** offline lexical reranker, cross-encoder and LLM rerankers,
  retrieve-many → rerank-few (`parsrag.rerank`).
- **09 Grounded generation:** numbered context, citations mapped to document / section / page /
  offsets, citation validation, calibrated «نمی‌دانم» (`parsrag.generate`).
- **10 Evaluation:** token F1, citation precision, faithfulness (heuristic and LLM judge),
  bootstrap confidence intervals, `python -m parsrag.eval gate` in CI (`parsrag.eval`).
- **11 Agentic RAG:** analyse / rewrite → route → retrieve → rerank → grade → generate →
  self-check → bounded retry; plain-Python runner and LangGraph `StateGraph` with human review
  (`parsrag.agent`).
- **12 Production API:** FastAPI with SSE streaming, a normalised-question cache, token-bucket
  rate limiting, JSON logs with per-stage latency, Docker and Compose with pgvector
  (`parsrag.api`).
- **13 Capstone:** the `PersianRAG` facade, the `parsrag` command line, an RTL web UI, own
  question sets with `validate_qa`, and a production-readiness checklist (`parsrag.rag`,
  `parsrag.cli`).
- Original Persian fixtures: 25 documents, 66 questions (`qa.jsonl`) and 51 paraphrases
  (`qa_paraphrase.jsonl`), plus PDF / DOCX / HTML / cp1256 documents for the loaders.

### Added — project

- English `README.md` with result charts (`examples/make_readme_charts.py`) and a Persian
  `README.fa.md`.
- Persian documentation site (MkDocs Material, RTL) on GitHub Pages, with library, CLI and HTTP
  API reference pages and a Persian/English glossary.
- Packaging for PyPI: full metadata, an explicit file list for the sdist and wheel
  (`scripts/check_dist.py`), and a Trusted Publishing release workflow (TestPyPI on demand,
  PyPI on a GitHub release).
- `python -m grader check-starters`; an offline Markdown link checker (`scripts/check_links.py`).
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, issue and PR
  templates, and a social preview image (`scripts/make_social_preview.py`).

[Unreleased]: https://github.com/snikookar/Persian-RAG-Academy/commits/main
[0.1.0]: https://github.com/snikookar/Persian-RAG-Academy/commits/main

# Roadmap

Each lesson builds one `parsrag` module, ships a Persian `README.fa.md`, a
`starter/` with TODOs and `tests/` used by the grader.

## Lessons

- [x] **00 — Setup**: environment, tests, grader, FakeLLM/HashingEmbedding, optional real API / Ollama → `parsrag.doctor`
- [x] **01 — RAG in 50 lines**: load → chunk → embed → retrieve → generate → `parsrag.naive_rag`
- [x] **02 — Persian normalization**: Arabic→Persian letters, digits, ZWNJ, diacritics; measured impact → `parsrag.normalize`
- [x] **03 — Loading Persian documents**: TXT/MD/HTML/DOCX/PDF, presentation forms, visual order, lost ZWNJ, bidi controls → `parsrag.loaders`
- [x] **04 — Chunking Persian text**: Persian sentence splitter, fixed / recursive / sentence-window, token estimate → `parsrag.chunking`
- [x] **05 — Embeddings**: Recall@k / MRR / nDCG benchmark, TF-IDF baseline, real-model script → `parsrag.embeddings`, `parsrag.eval`
- [x] **06 — Vector stores**: protocol, filters, save/load, FAISS flat/HNSW/IVF, pgvector → `parsrag.stores`
- [ ] **07 — Hybrid search**: Persian BM25 + dense + Reciprocal Rank Fusion → `parsrag.retrieval`
- [ ] **08 — Reranking** → `parsrag.rerank`
- [ ] **09 — Grounded generation**: citations and honest "I don't know" → `parsrag.generate`
- [ ] **10 — Evaluation**: Recall@k, MRR, faithfulness, LLM-as-judge → `parsrag.eval`
- [ ] **11 — Agentic RAG with LangGraph** → `parsrag.agent`
- [ ] **12 — Production API**: FastAPI, streaming, cache, Docker → `parsrag.api`
- [ ] **13 — Capstone**: RAG over your own documents + UI

## Baseline numbers (fixtures, HashingEmbedding)

Fixtures since Day 2: 25 documents, 66 questions (59 answerable, each with an evidence span).

| Pipeline | hit@1 | hit@3 |
|---|---|---|
| Lesson 01 naive (chunk 400) | 0.763 | 0.847 |
| Lesson 01 naive (chunk 300) | 0.814 | 0.864 |
| + Lesson 02 normalization (chunk 300) | 0.898 | 0.949 |

Passage-level (a retrieved chunk contains the evidence), normalised:

| Pipeline | R@1 | R@3 | MRR |
|---|---|---|---|
| Lesson 04 fixed-size 200 chars | 0.525 | 0.780 | 0.670 |
| Lesson 04 recursive 400 + heading context (course default) | 0.763 | 0.983 | 0.858 |
| Lesson 05 same pipeline, TF-IDF (word) | 0.729 | 0.932 | 0.836 |
| Lesson 05 same pipeline, TF-IDF (char 2–4) | 0.797 | 0.983 | 0.886 |

## Session log

### Day 1 — 2026-10-31: foundation + lessons 00–02

Built:
- `CLAUDE.md` project rules; `pyproject.toml` (hatchling, src layout, extras, ruff, pytest); `.gitignore`.
- `parsrag` core: `LLM` protocol, `FakeLLM` (canned / echo / extractive / handler),
  `OpenAICompatibleLLM` (lazy `httpx`, HTTP layer mocked in tests), `Embedding` protocol,
  `HashingEmbedding` (CRC32-hashed word + char 2–4-grams), lazy `SentenceTransformerEmbedding`,
  `InMemoryVectorStore`, pydantic types, `eval.load_qa` / `eval.hit_rate`.
- Lesson modules: `parsrag.doctor` (00), `parsrag.naive_rag` (01), `parsrag.normalize` (02).
- Grader: `python -m grader {list,run,check-solutions}` with bilingual report and per-test hints
  (`@pytest.mark.hint(fa=..., en=...)`), learner code loaded via `PARSRAG_LEARNER_PATH`.
- Fixtures: 10 original, deliberately messy Persian docs (fictional shop «کهربا» and library «سپیدار»)
  + 28 QA pairs (4 unanswerable).
- Lessons 00, 01, 02: Persian READMEs with Mermaid diagrams, starters, tests (10 / 17 / 74 checks).
- Examples: `rag_in_50_lines.py`, `normalization_impact.py` (with per-step ablation).
- CI: Python 3.10–3.12, ruff, pytest, grader check-solutions, examples, package build.

Results: `ruff check .` clean · `pytest` 160 passed · `python -m grader check-solutions` all pass
(also verified on Python 3.10 and 3.13 locally).

Findings worth remembering:
- Character unification is by far the most valuable normalization step on the fixtures
  (ablation: hit@1 0.958 → 0.833 without it). Diacritics/tatweel/ZWNJ/whitespace show no
  measurable effect with HashingEmbedding on this small set — re-measure with BM25 (lesson 07)
  and neural embeddings (lesson 05).
- With 24 answerable questions, one question = ~4 points; treat small differences with care.
  Growing the QA set would make later comparisons more reliable.

### Day 2 — 2026-10-31: fixtures + lessons 03–06

Built:
- Fixtures: corpus 10 → 25 original docs (4 long multi-section docs of 819–1089 words);
  `qa.jsonl` 28 → 66 questions (7 unanswerable) with a verbatim `evidence` span per
  answerable question (`QAItem.evidence`). Lesson 01/02 numbers, thresholds and locked
  tests updated for the larger set.
- `data/fixtures/documents/`: cp1256 TXT, boilerplate-heavy HTML, DOCX (headings, list,
  table, RLM marks), two PDFs (presentation forms in logical order; presentation forms in
  visual order), generated from `documents/sources/` by `data/fixtures/build_fixtures.py`
  (DOCX/TXT byte-reproducible with the stdlib, checked in CI; PDFs need the new `dev` extra).
- `parsrag.loaders` (03): TXT/MD/HTML/DOCX/PDF loaders with source/section/page offsets,
  `locate()`; `parsrag.loaders.bidi`: toy shaper, visual-order detection (glyph-shape and
  common-word votes) and repair, ZWNJ recovery from glyph shapes, bidi-control stripping.
- `parsrag.chunking` (04): Persian sentence splitter, fixed-size (word-snapped), recursive
  (heading → paragraph → line → sentence → word), sentence-window, `chunk_document` with
  page/section metadata and optional heading context, byte-based `estimate_tokens`.
- `parsrag.eval.retrieval` / `parsrag.eval.benchmark` (05): Recall@k, MRR, nDCG@k at doc and
  passage level; `benchmark_embedding` for any `Embedding`. `TfidfEmbedding`,
  `cosine_similarity`, `OpenAICompatibleEmbedding` (mocked in tests).
- `parsrag.stores` (06): `VectorStore` protocol, dict/callable filters, precomputed vectors,
  save/load; `FaissVectorStore` (flat/HNSW/IVF, post-filter + exact fallback);
  `PgVectorStore` (dict filters → SQL; integration test only with `PARSRAG_TEST_DATABASE_URL`).
- Grader: skipped checks (missing optional extras) are shown with their install hint and do
  not fail a lesson.
- Examples: `pdf_repair_impact.py`, `chunking_benchmark.py` (+ `docs/images/chunking_mrr.png`),
  `benchmark_embeddings.py` (offline by default; `st:`/`openai:` models on demand),
  `vector_store_latency.py`.
- CI: main matrix installs `.[loaders,faiss]`; new core-only job; fixtures check; new examples.

Results: `ruff check .` clean · `pytest` 377 passed, 1 skipped (pgvector without DB) ·
`python -m grader check-solutions` all pass (36 / 27 / 22 / 15 checks for lessons 03–06).
Also verified: Python 3.10 with extras, Python 3.12 core-only (16 extras checks skipped),
completed starters pass every lesson, and `PgVectorStore` against a real PostgreSQL 16 +
pgvector (local, not in CI).

Findings worth remembering:
- PDF repair (on our own simulated PDFs): word recall 0.07 → 1.00; NFKC alone recovers 94%
  for logical-order PDFs but only 8% for visual-order ones.
- Chunking: structure beats fixed windows at every size; heading context is the cheapest win
  (fixed 200: MRR 0.670 → recursive+headings 200: 0.825). Doc-level recall hides all of this.
- Lexical embeddings: char n-grams > hashing > word TF-IDF on this corpus. Normalising only
  the index (not queries) costs TF-IDF-word 0.15 MRR on Arabic-keyboard queries.
- HNSW + a very selective post-filter can miss matches even when fetching the whole index;
  `FaissVectorStore` falls back to exact search over the matching chunks.
- Questions reuse document wording, which favours lexical methods; real-model numbers must
  be produced by learners (`examples/benchmark_embeddings.py`), never invented.

### Next session (Day 3) — lessons 07–10

- Lesson 07 — hybrid search: Persian BM25 (normalisation + simple tokenisation; consider
  light stemming of ها/های/ترین, measure it), dense + BM25 with Reciprocal Rank Fusion,
  `parsrag.retrieval`; compare on passage-level MRR with `evaluate_retrieval`.
- Lesson 08 — reranking: offline cross-encoder stand-in (e.g. term-overlap / BM25 rescoring)
  for tests, optional real cross-encoder behind `st`; measure R@1 gains.
- Lesson 09 — grounded generation with citations (use chunk `start`/`end`/`page`/`section`
  metadata) and honest "I don't know" on the 7 unanswerable questions; `FakeLLM` in tests.
- Lesson 10 — evaluation: reuse `parsrag.eval.retrieval`; add faithfulness and
  LLM-as-judge with `FakeLLM`; consider paraphrased/synonym questions (not reusing document
  wording) so lexical vs. semantic differences become visible.

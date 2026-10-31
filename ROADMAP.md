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
- [x] **07 — Hybrid search**: Persian BM25 (+ light stemming) + dense + RRF / weighted fusion → `parsrag.retrieval`
- [x] **08 — Reranking**: lexical (offline), cross-encoder (`st`), LLM pointwise; retrieve-many → rerank-few → `parsrag.rerank`
- [x] **09 — Grounded generation**: numbered context, citations → chunk metadata, citation validation, calibrated «نمی‌دانم» → `parsrag.generate`
- [x] **10 — Evaluation**: token F1, citation precision, abstention, faithfulness (heuristic + LLM judge), bootstrap CIs, `python -m parsrag.eval gate` in CI → `parsrag.eval`
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

Lesson 07–08, passage level, both question sets (`qa.jsonl`: 59 answerable;
`qa_paraphrase.jsonl`: 46 answerable). Course chunking (recursive 400 + headings),
dense = HashingEmbedding. Lesson 07 rows use depth 12 (`evaluate_retrieval` default
for ks 1, 3), lesson 08 rows MRR@10:

| Pipeline | orig R@1 | orig R@3 | orig MRR | para R@1 | para R@3 | para MRR |
|---|---|---|---|---|---|---|
| BM25 | 0.780 | 0.949 | 0.872 | 0.370 | 0.435 | 0.434 |
| BM25 + light stemming | 0.814 | 0.983 | 0.893 | 0.370 | 0.478 | 0.448 |
| dense (hashing) | 0.763 | 0.983 | 0.858 | 0.196 | 0.457 | 0.355 |
| RRF (BM25 + dense) | 0.831 | 0.983 | 0.903 | 0.391 | 0.478 | 0.475 |
| RRF (BM25 + stem + dense) — course default | 0.881 | 0.966 | 0.929 | 0.348 | 0.522 | 0.468 |
| + lexical rerank of 20 (MRR@10) | 0.898 | 1.000 | 0.949 | 0.370 | 0.522 | 0.488 |

End-to-end course pipeline (`python -m parsrag.eval run`; FakeLLM extractive;
committed in `docs/eval/baseline.json`, gated in CI with tolerance 0.02):

| Metric | original | paraphrase |
|---|---|---|
| retrieval R@1 [95% CI] | 0.898 [0.814, 0.966] | 0.370 [0.239, 0.500] |
| retrieval MRR@10 | 0.949 [0.907, 0.983] | 0.488 [0.372, 0.600] |
| answer token F1 (abstain = 0) | 0.289 [0.221, 0.360] | 0.016 [0.005, 0.031] |
| citation precision | 0.860 | 0.800 |
| faithfulness (heuristic) | 1.000 | 1.000 |
| abstention F1 (BM25 threshold) | 0.714 | 0.178 |
| answered rate (answerable questions) | 0.966 | 0.217 |

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

### Day 3 — 2026-10-31: paraphrase set + lessons 07–10

Built:
- Fixtures: `qa_paraphrase.jsonl` — 51 originally written questions (46 answerable, 5
  unanswerable) about the same facts as `qa.jsonl`, with synonyms, colloquial Persian,
  loanwords and three deliberate typos; each row links `source_id` and reuses the original's
  `doc_id`/`answer`/`evidence`. `QAItem.source_id`; `load_fixture_benchmark(qa_file=...)`;
  consistency tests.
- `parsrag.retrieval` (07): Persian `tokenize` (normalise, ZWNJ-aware, optional stop words,
  ZWNJ splitting, `light_stem`), `BM25Retriever` from scratch, `DenseRetriever`,
  `reciprocal_rank_fusion`, `weighted_score_fusion`, `HybridRetriever`; one `Retriever`
  protocol shared with vector stores.
- `parsrag.rerank` (08): `Reranker` protocol, `LexicalReranker` (idf-weighted coverage with
  typo/prefix matching, proximity, first-stage prior), lazy `CrossEncoderReranker` (`st`),
  `LLMReranker` (Persian 0–10 prompt, `parse_score`), `RerankingRetriever`.
- `parsrag.generate` (09): Persian system prompt, numbered context blocks, `Citation` mapped to
  chunk metadata (doc, section, page, start/end), `validate_citations` (invalid / uncited),
  `is_refusal`, `GroundedGenerator` with a pluggable confidence + threshold,
  `calibrate_threshold`, `abstention_stats`. `FakeLLM` extractive mode gained opt-in
  `cite`, `ignore` and `min_overlap` (and skips `[n]` header lines).
- `parsrag.eval` (10): `generation` (token F1 with Persian normalisation, citation
  precision/hit, claim-level faithfulness heuristic with a numbers rule, `LLMJudge` with
  Persian faithfulness/correctness prompts), `stats` (bootstrap + paired bootstrap),
  `pipeline` (`evaluate_pipeline`, `build_course_pipeline`, JSON/Markdown reports), `gate`
  and `python -m parsrag.eval {run,gate}`; baseline + report in `docs/eval/`.
- Lessons 07–10: Persian READMEs with Mermaid diagrams and measured tables, starters with
  TODOs, grader tests with bilingual hints (20 / 21 / 20 / 14 checks).
- Examples: `hybrid_search.py` (+ `docs/images/hybrid_mrr.png`), `reranking.py`
  (+ `docs/images/rerank_r1.png`, latency and LLM-cost estimates), `grounded_answers.py`
  (`--llm` for any OpenAI-compatible model), `evaluation_report.py` (`--judge`).
- CI: eval gate step in the main matrix and in the core-only job; new examples.

Results: `ruff check .` clean · `pytest` 480 passed, 1 skipped (pgvector without DB) ·
`python -m grader check-solutions` all pass · `python -m parsrag.eval gate` passes.
Also verified: Python 3.10 core-only install (455 passed, 16 skipped; all lessons pass;
gate reproduces the baseline exactly), starters fail with hints, package builds.

Findings worth remembering:
- Paraphrases halve every lexical method: passage MRR ≈ 0.87–0.93 → 0.36–0.49. HashingEmbedding
  is the worst on paraphrases (0.355): it is "dense" but lexical underneath.
- Light stemming, stop words, ZWNJ splitting: small or mixed effects. Paired bootstrap
  (lesson 10) shows only "hybrid beats dense alone" is a defensible claim on our sets;
  stemming (+0.021 MRR), hybrid vs BM25+stem (+0.036) and lexical reranking (+0.020) all
  have 95% intervals that contain 0.
- Weighted fusion is sensitive to weights (0.3/0.7 drops paraphrase R@1 to 0.304); RRF needs
  no tuning and is the course default.
- Reranker proximity hurt on the tuning set in every setting (heading context puts common
  query words side by side); idf weighting in the reranker also hurt. Both are off by default.
  The setting that looked best on paraphrases was not the one chosen on the original set —
  a live example of test-set overfitting.
- Abstention: raw top BM25 score separates answerable/unanswerable best (F1 0.71 on
  original); reranker coverage is the worst signal (unanswerable questions reuse corpus
  words). The calibrated threshold does **not** transfer to paraphrases: the pipeline then
  answers only 10/46 answerable paraphrases. Lexical confidence measures "worded like the
  docs", not "answer is in the docs".
- Extractive FakeLLM is perfectly faithful (1.0) but answer F1 is only 0.29 — faithful ≠
  correct. Citation precision 0.86: the cited block often is not the one with the evidence.
- Real-model numbers (neural embeddings, cross-encoders, LLM refusal, LLM judge) are left
  for learners to produce with the `--model`/`--cross-encoder`/`--llm`/`--judge` flags;
  none were invented.

### Next session (Day 4) — lessons 11–13

- **Facade first: `parsrag.PersianRAG`.** Today a full pipeline takes ~15 lines of wiring
  (see `build_course_pipeline`). Proposal, a thin layer over existing modules, no new deps:

  ```python
  from parsrag import PersianRAG

  rag = PersianRAG.from_directory("my_docs/", embedding=HashingEmbedding(), llm=FakeLLM(...))
  # or PersianRAG(documents, chunker=..., embedding=..., store=..., reranker=..., llm=...)
  ans = rag.ask("هزینهٔ پیک چقدر است؟")            # -> GroundedAnswer (citations, abstained)
  rag.search("...", k=5)                            # -> list[SearchResult] (hybrid + rerank)
  rag.add_documents(docs); rag.save("index/"); PersianRAG.load("index/", ...)
  rag.calibrate(qa_items)                           # sets the abstention threshold
  rag.evaluate(qa_items)                            # -> PipelineReport
  ```

  Defaults = the course choices (normalise, recursive 400 + headings, RRF(BM25+stem, dense),
  lexical rerank of 20, k=4, BM25 threshold). Every part overridable; `build_course_pipeline`
  becomes `PersianRAG.course_default()`. Keep it importable from the core install.
- **Lesson 11 — agentic RAG with LangGraph** (`parsrag.agent`, `agent` extra, lazy import):
  a small graph — route (answer directly / retrieve / say «نمی‌دانم») → retrieve → grade
  relevance → rewrite the query and retry (max 2) → generate with citations. The query
  rewriter is the natural fix for today's paraphrase gap; measure it on
  `qa_paraphrase.jsonl` with the lesson 10 report (FakeLLM rewriter with a small synonym
  table offline; real LLM on demand). Provide a plain-Python fallback loop so the lesson's
  tests run without LangGraph; LangGraph checks are skipped when the extra is missing.
- **Lesson 12 — production API** (`parsrag.api`, `api` extra): FastAPI app around
  `PersianRAG` — `POST /ask` (JSON with citations), `POST /ask/stream` (SSE; FakeLLM streams
  word by word), `POST /documents`, `GET /health`; answer cache keyed by normalised question
  + index version; request timeouts; Dockerfile (core + api, non-root, healthcheck);
  tests with FastAPI's TestClient (skipped without the extra). Measure cache hit latency.
- **Lesson 13 — capstone**: RAG over the learner's own folder via `PersianRAG.from_directory`
  (loaders from lesson 03), a minimal UI (single static HTML page talking to the API, RTL,
  clickable citations using `start`/`end`), a template for writing 20+ own questions
  (answerable, paraphrased, unanswerable) and running `python -m parsrag.eval` on them.
- Consider: growing both QA sets (+50 questions) so lesson 07–08 differences can be tested
  with tighter intervals; a relative BM25 confidence (score / max possible for the query)
  for abstention that might transfer better to paraphrases.

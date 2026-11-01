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
- [x] **11 — Agentic RAG with LangGraph**: analyse/rewrite → route → retrieve → rerank → grade → generate → self-check → bounded retry; plain-Python runner + LangGraph `StateGraph`, `interrupt()` review, checkpointers → `parsrag.agent`
- [x] **12 — Production API**: FastAPI, SSE streaming, normalised-question cache, token-bucket rate limiting, JSON logs with per-stage latency, Docker + compose with pgvector → `parsrag.api`
- [x] **13 — Capstone**: `PersianRAG` facade, `parsrag` CLI, RTL web UI, own question sets (`validate_qa`), eval gate, production checklist → `parsrag.rag`, `parsrag.cli`
- [x] **14 — RAG security**: indirect prompt injection hidden in documents; detection (Persian + English, normalised), sanitising, sandbox with a random nonce, instruction hierarchy, post-generation checks, opt-in guard in `PersianRAG` and the API; measured with simulated models → `parsrag.security`

## Baseline numbers (fixtures, HashingEmbedding)

Fixtures since Day 2: 25 documents, 66 questions (59 answerable, each with an evidence span).

| Pipeline | hit@1 | hit@3 |
|---|---|---|
| Lesson 01 naive (chunk 400) | 0.763 | 0.864 |
| Lesson 01 naive (chunk 300) | 0.814 | 0.864 |
| + Lesson 02 normalization (chunk 300) | 0.898 | 0.949 |

Passage-level (a retrieved chunk contains the evidence), normalised:

| Pipeline | R@1 | R@3 | MRR |
|---|---|---|---|
| Lesson 04 fixed-size 200 chars | 0.525 | 0.763 | 0.660 |
| Lesson 04 recursive 400 + heading context (course default) | 0.729 | 0.983 | 0.841 |
| Lesson 05 same pipeline, TF-IDF (word) | 0.729 | 0.932 | 0.836 |
| Lesson 05 same pipeline, TF-IDF (char 2–4) | 0.797 | 0.983 | 0.883 |

Lesson 07–08, passage level, both question sets (`qa.jsonl`: 59 answerable;
`qa_paraphrase.jsonl`: 46 answerable). Course chunking (recursive 400 + headings),
dense = HashingEmbedding. Lesson 07 rows use depth 12 (`evaluate_retrieval` default
for ks 1, 3), lesson 08 rows MRR@10:

| Pipeline | orig R@1 | orig R@3 | orig MRR | para R@1 | para R@3 | para MRR |
|---|---|---|---|---|---|---|
| BM25 | 0.780 | 0.949 | 0.872 | 0.370 | 0.435 | 0.434 |
| BM25 + light stemming | 0.814 | 0.983 | 0.893 | 0.370 | 0.478 | 0.448 |
| dense (hashing) | 0.729 | 0.983 | 0.841 | 0.196 | 0.391 | 0.349 |
| RRF (BM25 + dense) | 0.831 | 0.983 | 0.905 | 0.370 | 0.478 | 0.467 |
| RRF (BM25 + stem + dense) — course default | 0.864 | 0.983 | 0.925 | 0.326 | 0.500 | 0.453 |
| + lexical rerank of 20 (MRR@10) | 0.881 | 1.000 | 0.941 | 0.370 | 0.543 | 0.487 |

End-to-end course pipeline (`python -m parsrag.eval run`; FakeLLM extractive;
committed in `docs/eval/baseline.json`, gated in CI with tolerance 0.02):

| Metric | original | paraphrase |
|---|---|---|
| retrieval R@1 [95% CI] | 0.881 [0.797, 0.949] | 0.370 [0.239, 0.500] |
| retrieval MRR@10 | 0.941 [0.898, 0.975] | 0.487 [0.373, 0.600] |
| answer token F1 (abstain = 0) | 0.276 [0.210, 0.344] | 0.016 [0.005, 0.031] |
| citation precision | 0.842 | 0.800 |
| faithfulness (heuristic) | 1.000 | 1.000 |
| abstention F1 (BM25 threshold) | 0.714 | 0.178 |
| answered rate (answerable questions) | 0.966 | 0.217 |

Lesson 11 — agent vs. lesson 09 pipeline (same index, retriever, reranker and BM25
threshold; offline `OfflineAgentLLM`, grade threshold 0.4 chosen on `qa.jsonl` only;
`python examples/agentic_rag.py`):

| Metric | orig: lesson 09 | orig: agent | para: lesson 09 | para: agent | para: agent, no colloquial rules |
|---|---|---|---|---|---|
| retrieval R@1 | 0.881 | 0.881 | 0.370 | 0.478 | 0.370 |
| retrieval MRR@10 | 0.941 | 0.941 | 0.487 | 0.557 | 0.469 |
| answer F1 | 0.276 | 0.301 | 0.016 | 0.021 | 0.007 |
| citation precision | 0.842 | 0.877 | 0.800 | 0.562 | 0.700 |
| abstention F1 | 0.714 | 0.714 | 0.178 | 0.205 | 0.178 |
| answered rate | 0.966 | 0.966 | 0.217 | 0.348 | 0.217 |
| LLM calls / question (mean / max) | 0.89 / 1 | 3.89 / 4 | 0.22 / 1 | 3.49 / 6 | 3.27 / 6 |

Paired bootstrap, answer F1 agent − lesson 09: original +0.025 [0.000, +0.063],
paraphrase +0.005 [-0.010, +0.019] — neither shown.

Lesson 12 — API latency, one run on the dev container, in-process TestClient, 117 questions
(`python examples/api_latency.py`; the second table uses a *simulated* 30 ms/word model):

| Request | median ms | p95 ms |
|---|---|---|
| /ask, cache miss (FakeLLM) | 9.08 | 13.23 |
| /ask, cache hit | 1.90 | 2.56 |
| /ask, cache miss (simulated model) | 251.80 | 880.83 |
| /ask/stream, first token (simulated model, server side) | 37.21 | 42.92 |
| /ask/stream, whole answer (simulated model, server side) | 345.86 | 1130.20 |

Spelling variants (Arabic ي/ك, no «؟», extra spaces) hitting the cache: raw key 0/117,
normalised key 117/117.

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
- Fixtures: 10 original, deliberately messy Persian docs (fictional shop «کهربا» and library «مهرگان»)
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

### Day 4 — 2026-10-31: lessons 11–13

Built:
- `parsrag.rag.PersianRAG` (+ `parsrag.config`): one object for the course pipeline with
  every part replaceable (`embedding`, `llm`, `chunker`, `store`, `reranker`, `k`, ...),
  `from_directory` / `from_config` (TOML or JSON) / `course_default`, `add_documents`
  (documents, files, folders), `search`, `ask` (optional per-stage `Timings`), `stream`,
  `passage` (exact original span of a citation), `calibrate`, `evaluate`, `save` / `load`.
  `build_course_pipeline` now delegates to it and reproduces the gate baseline exactly.
- `GroundedGenerator.answer` split into `precheck` / `messages` / `finalize` (`answer_from`);
  `FakeLLM.stream`, `OpenAICompatibleLLM.stream` (SSE), `stream_text`, `CountingLLM`.
- `parsrag.agent` (11): `AgenticRAG` nodes + routers shared by a plain-Python runner (core
  install) and `build_graph` (LangGraph `StateGraph`, checkpointer, `interrupt()` human
  review, `draw_mermaid`); Persian prompts with forgiving parsers; `OfflineAgentLLM`
  answering every step with rules (generic colloquial → standard Persian, question
  splitting, lexical grading, keyword rewrites, heuristic faithfulness).
- `parsrag.api` (12): FastAPI app (`/health`, `/ingest`, `/ingest/files`, `/documents`,
  `/ask`, `/ask/stream` SSE, `/` UI), `Settings` from `PARSRAG_*` env vars, `AnswerCache`
  (LRU + TTL, key = normalised question + index version), `RateLimiter` (token bucket per
  API key/IP), JSON logs with per-stage latency (question hashed, not logged), clear errors
  (409 empty index/duplicates, 413/415 uploads, 429, 502, 504 provider timeout).
  `Dockerfile` (multi-stage, slim, uid 10001, healthcheck), `docker-compose.yml` with
  `pgvector/pgvector:pg16`. `python-multipart` added to the `api` extra.
- `parsrag.cli` (13): `parsrag init | ingest | ask | eval | serve`; `eval` calibrates
  (`--calibrate`, `--calibrate-on`), gates against a baseline and warns about QA problems
  found by the new `parsrag.eval.validate_qa`.
- Web UI (13): one RTL HTML page, no build step, no external requests: upload, streamed
  answers, clickable citations with source passage and highlighted span in the full document.
- Lessons 11–13: Persian READMEs with Mermaid diagrams (and LangGraph's own drawing in
  `docs/images/agent_graph.mmd`), starters with TODOs, grader checks (14 / 10 / 11); the
  capstone has an example question file and a production-readiness checklist.
- Examples: `agentic_rag.py` (`--llm`, `--mermaid`), `api_latency.py` (`--token-delay`).
- CI: main matrix installs `.[loaders,faiss,agent,api,openai]`, runs the new examples and a
  CLI sequence (ingest → ask → eval gate); new `docker` job builds the image, smoke-tests
  `/health`, `/ask` and the non-root user, validates the compose file (skipped without Docker).

Results: `ruff check .` clean · `pytest` 586 passed, 1 skipped (pgvector without DB) ·
`python -m grader check-solutions` all pass · `python -m parsrag.eval gate` passes.
Also verified: Python 3.10 with extras (585 passed, 2 skipped: pgvector, TOML test needs
3.11), Python 3.12 core-only (537 passed, 26 skipped; all lessons pass; gate passes;
`parsrag serve` prints the install hint), completed starters pass lessons 11–13 and the
blank starters fail with hints, `PgVectorStore` tests and the API's database mode against a
real PostgreSQL 16 + pgvector container, the Docker image (420 MB, uid 10001, `healthy`)
and the compose stack (109 chunks in `parsrag_chunks`, `/ask` answered), the web UI in
headless Chromium (ask, citation click, document highlight, upload), and the wheel
contains the UI and the `parsrag` entry point. Image builds here needed a local proxy CA;
the committed Dockerfile is unchanged by that.

Findings worth remembering:
- The agent costs ~4.4× the LLM calls of the lesson 09 pipeline on `qa.jsonl` (3.89 vs
  0.89 per question) for an answer-F1 gain of +0.025 whose interval touches 0. On
  paraphrases the whole retrieval gain (R@1 0.370 → 0.478) comes from the colloquial rules;
  the ablation without them equals lesson 09 exactly. The offline retry loop (keyword
  queries) rescues almost nothing: 71% of paraphrases were rewritten, answered rate only
  0.22 → 0.35, and citation precision fell 0.80 → 0.56 on the newly answered questions.
- Test-set leakage, admitted in lesson 11: some colloquial rules were added after looking at
  rewritten `qa_paraphrase.jsonl` questions. One-off measurement: paraphrase R@1 with no
  rules 0.370, first-pass rules 0.435, all rules 0.478 — treat ~0.043 of the gain as
  optimistic. Future rules must be written on a separate development set.
- The offline faithfulness self-check never fires (extractive answers are always faithful);
  its value only shows with a model that hallucinates (simulated in tests).
- The synonym gap («کرایه ↔ هزینه», «پایتخت ↔ تهران») cannot be closed by rules; it is the
  job of a real model or neural embeddings, which must be measured by learners.
- Without calibration the system never abstains on score: an uncalibrated `serve --docs`
  answered an out-of-scope question with an unrelated heading. The capstone makes
  `--calibrate` on a dev set part of the workflow.
- On the 13-question capstone example the course pipeline gets R@1 0.900 [0.700, 1.000] —
  a reminder of how wide intervals are on small own sets.
- Normalising the cache key matters for Persian: 0/117 spelling variants hit a raw-text
  cache, 117/117 hit the normalised key. Cache hits cost ~2 ms vs ~9 ms (FakeLLM) or
  hundreds of ms with a (simulated) model; streaming cut time to first token from ~346 ms
  to ~37 ms with the simulated model.
- TestClient buffers streamed responses; time-to-first-token has to be measured on the
  server's generator (done in `api_latency.py`).

Deviations from the Day 4 brief:
- Agent order is retrieve → **rerank → grade** (not grade → rerank): grading is the expensive
  LLM step, so the cheap reranker narrows 20 candidates to 6 first and all 6 are graded in
  one call.
- After a rewrite, a multi-part question is retried as one query (one generation).
- The PostgreSQL checkpointer is shown in lesson 11 but not tested (it needs a database and
  `langgraph-checkpoint-postgres`, which is not in the `agent` extra); tests use
  `InMemorySaver`.
- In database mode the API rebuilds the pgvector table from the documents folder at startup
  and keeps BM25 in memory.
- The UI does not load Vazirmatn from a CDN (no external requests); it uses it when installed.
- `OpenAICompatibleLLM.stream` is tested with a mocked SSE stream only (no network).

### Day 5 — 2026-10-31: README, docs site, release preparation and polish

Built:
- `README.md` rewritten as the project's front page: pitch, badges (CI, Docs, PyPI, Python,
  license), "why Persian breaks naive RAG" with three before/after examples from lessons 02, 03
  and 11, Mermaid architecture and course-map diagrams, six result charts with an explicit
  caveat (synthetic corpus, offline models; real models via `examples/benchmark_embeddings.py`),
  quick start for learners / library users / service, lesson table with GitHub and docs-site
  links, FAQ, contributing, license, citation. `README.fa.md`: a full Persian version.
- `examples/make_readme_charts.py`: re-measures every chart number by calling the lesson
  examples' own functions and writes `docs/images/readme_*.png` and
  `docs/eval/readme_numbers.json`; CI fails if the committed numbers no longer reproduce.
- Persian documentation site: MkDocs Material (`language: fa`, RTL, Vazirmatn, Mermaid, code
  copy, search, light/dark). Lesson pages are generated from `lessons/*/README.fa.md` by
  `scripts/mkdocs_hooks.py` (no copies; lesson / site / repository links rewritten, "edit"
  points at the README). Pages for the library, CLI and HTTP API, results, and
  `docs/glossary.fa.md`. `docs` extra; `.github/workflows/docs.yml` builds with
  `mkdocs build --strict` on every PR and deploys from `main` with the official Pages actions.
- Packaging: version `0.1.0` (not tagged), full PyPI metadata (keywords, classifiers, project
  URLs), PyPI readme with absolute links via `hatch-fancy-pypi-readme`, explicit sdist file list
  (`src/parsrag`, `README.md`, `LICENSE`, `pyproject.toml`, `CHANGELOG.md`);
  `scripts/check_dist.py` + `twine check --strict` in CI.
- `.github/workflows/release.yml`: published GitHub release → PyPI (environment `pypi`), manual
  run → TestPyPI (environment `testpypi`), Trusted Publishing with `id-token: write`; tag/version
  check, dist check and a wheel smoke test before upload. `docs/RELEASING.md` lists the one-time
  setup on pypi.org, test.pypi.org and GitHub. Nothing was published, tagged or released.
- Community files: `CONTRIBUTING.md`, `CHANGELOG.md` (Keep a Changelog, 0.1.0),
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`, `CITATION.cff`, issue forms
  (bug, lesson feedback, new lesson proposal), PR template, social preview image
  (`scripts/make_social_preview.py`, 1280×640, libraqm or the lesson 03 shaper).
- `scripts/check_links.py`: offline check of relative links and heading anchors in every
  tracked Markdown file (new CI job). `python -m grader check-starters` (in CI) and the starter
  test now cover all 14 lessons (it stopped at lesson 10).
- Consistency pass over the 14 lessons: same skeleton (title with English term, info box with
  module / starter / grader command and check count / prerequisite, … exercise → curious →
  further reading → next lesson with a link); new further-reading sections for lessons 00–04
  and 06; lesson 02's check count fixed (74 → 89); terms aligned with the glossary (توهم،
  آزمایش جداسازی، فهرست وارونه، انبارهٔ برداری، سرتیتر); every number table in lessons 02–12
  re-checked against its example script and the baselines above (all match).

Results: `ruff check .` clean · `pytest` 590 passed, 1 skipped (pgvector without DB) ·
`python -m grader check-solutions` all pass · `python -m grader check-starters` all fail as they
should · `python -m parsrag.eval gate` passes · `mkdocs build --strict` passes · sdist/wheel
contain only the allowed files and pass `twine check --strict` · link checker clean ·
fixtures and README numbers reproduce. Also verified: core-only install (541 passed, 26
skipped), the site in headless Chromium (RTL layout, lesson pages, rewritten links), the social
preview with and without libraqm.

Findings worth remembering:
- hatchling force-includes `.gitignore` in every sdist and has no switch to turn it off
  (`ignore-vcs` does not affect it); `check_dist.py` allows exactly that file and `PKG-INFO`
  besides the five listed files.
- Relative links break on PyPI; rewriting them at build time keeps the README relative (and
  link-checked) on GitHub and absolute on PyPI.
- Generating the lesson pages from the READMEs in a hook keeps one source of truth; the link
  checker and `mkdocs build --strict` together catch broken links on both GitHub and the site.

Deviations from the Day 5 brief:
- The web UI screenshot is the existing `docs/images/capstone_ui.png` (captured offline in
  headless Chromium on Day 4); it was not regenerated.
- The Python badge is static (3.10–3.12); the PyPI badge starts working after the first release.
- The site loads Vazirmatn (Google Fonts) and Mermaid (CDN) in the reader's browser; the build
  itself needs no network beyond installing the `docs` extra.
- The library reference page is hand-written in Persian (signatures from the code), not
  generated from docstrings, to keep the `docs` extra small.

### Day 6 — 2026-10-31: UI refresh and realistic fixture content

Built / changed:
- Web UI: Vazirmatn 33.0.3 bundled as woff2 (Regular/Medium/Bold, official npm package, SIL
  OFL 1.1 licence alongside) and served by the API at `/static/fonts/`; navy/blue palette
  (primary `#1e3a8a`, hover `#1e40af`) with consistent hover/focus/disabled states, WCAG AA in
  light and dark mode; header shows only the title; every API error code and network failure
  shown in Persian; Persian, keyboard-accessible file picker. `capstone_ui.png` recaptured in
  headless Chromium.
- Fixtures: the fictional library is renamed to «کتابخانهٔ مهرگان» (slugs `mehregan-*`, newsletter PDF
  regenerated); every price set to a plausible late-2026 level (Tehran courier 200,000 Toman,
  the rest scaled consistently), only amounts changed.
- All benchmarks, charts, `baseline.json`, `readme_numbers.json` and `report.md` recomputed;
  lessons, READMEs and the baselines above updated from the new output. Main shifts (original
  set): course pipeline R@1 0.898 → 0.881, MRR 0.949 → 0.941, answer F1 0.289 → 0.276; dense
  hashing MRR 0.858 → 0.841. Changing a few tokens of a 25-document corpus moves single
  questions, which is the lesson-10 point about small evaluation sets.

Decisions:
- Locked test expectations were updated only from recomputed values; the lesson-07 sanity
  floor (`MRR > 0.85` for BM25 and dense) became `> 0.83` because dense now measures 0.841.
- Where a claim no longer held exactly, the text says so: weighted 0.5/0.5 fusion now edges RRF
  on the original questions within noise (0.932 vs 0.925), and switching off diacritic removal
  now moves one question.
- Historical session-log entries above keep the numbers measured at the time.

### Day 7 — 2026-11-01: Colab embedding benchmark and lesson 14 (RAG security)

Docs deployment check:
- The failed `github-pages` deployment on `main` (run of 2026-11-01 04:08) was a 404 from
  `actions/deploy-pages` ("Ensure GitHub Pages has been enabled"): Pages was not enabled yet.
  The manual re-run after enabling Pages succeeded. Permissions (`pages: write`,
  `id-token: write`), `environment: github-pages`, `upload-pages-artifact@v3` /
  `deploy-pages@v4` and `mkdocs build --strict` were already fine. Only change: a running
  deployment from `main` is no longer cancelled by the next push (`cancel-in-progress` only
  for PRs). The runner warns that `deploy-pages@v4` targets Node 20 (forced onto Node 24);
  it still works.

Built:
- `notebooks/benchmark_real_embeddings.ipynb` (Open in Colab badge; linked from lesson 05,
  both READMEs, the docs home and results pages): installs `parsrag` from the repository,
  loads the corpus with both question sets, scores multilingual-e5 small/base, bge-m3 and
  paraphrase-multilingual-MiniLM next to hashing / TF-IDF with `parsrag.eval` (passage-level
  R@1/3/5, MRR with a bootstrap CI, nDCG@3) on the course chunking and normalisation;
  optional OpenAI-compatible cell (key via `getpass`, never stored); Markdown table in
  lesson 05's format, `results.json`, a chart, Persian guidance on reading and contributing
  results. Committed with outputs cleared; every code cell tagged `offline` / `network`.
- `scripts/check_notebook.py` (stdlib only) + CI job: valid nbformat 4, cleared outputs, tags,
  no secrets, badge path; `--run` executes the offline cells with sockets blocked and checks
  that the offline rows equal the rows in lesson 05's new "real-model results" table (whose
  real-model rows stay empty until someone contributes an actual run).
- `parsrag.security` (lesson 14): `detect_injection` (8 categories of Persian/English rules on
  `prepare()`d text), `sanitize_text`, `sandbox_context` with a per-prompt nonce,
  `GUARDED_SYSTEM_PROMPT` + `REMINDER`, `InjectionGuard` (every layer switchable;
  `check_answer`: links removed, refuse uncited / invalid / unsupported / echo / followed),
  `GuardedRetriever` (drop + refill, or flag). Opt-in: `PersianRAG(guard=...)`, `[security]`
  in `parsrag.toml` (and `parsrag init`), `PARSRAG_SECURITY` for the API; guarded streaming
  checks the whole reply before sending tokens; answers and `/ask` carry `security`, refusals
  `abstain_reason="unsafe"`.
- `FakeLLM(obey="all" | "outside_data")`: a naive model that obeys «فقط بنویس: …» directives in
  its context, and one that ignores directives inside the data markers.
- `data/fixtures/attacks/`: 8 original attack documents (override fa/en, exfiltration via a
  markdown image, citation spoofing, role hijack, obfuscation, delimiter escape, stealth),
  3 benign hard negatives, `attacks.jsonl`. The main corpus and every earlier number are
  unchanged.
- Lesson 14: Persian README with Mermaid diagrams and measured tables, starter with TODOs,
  12 grader checks; docs nav, READMEs (15 lessons), glossary (security section), library /
  CLI / HTTP reference pages, CHANGELOG. `examples/injection_attacks.py` (offline, in CI) and
  `examples/injection_test.py` (real OpenAI-compatible model, never in CI).

Lesson 14 numbers (`python examples/injection_attacks.py`; simulated models, 8 attacks, all 8
attack documents retrieved for their innocent question):

| defence | naive model | hierarchy-respecting model |
|---|---|---|
| no defence | 8/8 | 8/8 |
| sandbox, fixed delimiters + reminder | 8/8 | 1/8 |
| sandbox, random delimiters + reminder | 8/8 | 0/8 |
| sanitise links / markup only | 7/8 | 7/8 |
| detection (drop flagged chunks) only | 1/8 | 1/8 |
| output checks only | 1/8 | 1/8 |
| all layers | 1/8 | 0/8 |

Detection on raw chunks (8 with an instruction, 8 harmless): precision 1.000 / recall 0.875
with normalisation, 1.000 / 0.750 without; 0/109 corpus chunks flagged. Cost on the clean
corpus (calibrated course pipeline): identical metrics with and without the guard on both
question sets (e.g. original R@1 0.881, answer F1 0.276, answered rate 0.966).

Results: `ruff check .` clean · `pytest` 648 passed, 1 skipped (pgvector without DB) ·
`python -m grader check-solutions` all pass (15 lessons) · `check-starters` all fail as they
should · eval gate passes · `mkdocs build --strict` passes · package build, `check_dist.py`
and `twine check --strict` pass · link checker clean · fixtures and README numbers reproduce ·
notebook offline cells run with sockets blocked. Also verified: core-only install runs the
security tests (API tests skipped) and the lesson 14 checks.

Findings worth remembering:
- Sandboxing and reminders do nothing against a model that does not respect them (naive model
  8/8 either way); fixed delimiters can be closed by the document itself (a07), a random
  nonce cannot.
- Detection and output checks each stop 7/8 but share the same rules, so they fail together on
  the stealth attack (a08). A supported, faithful answer can still be an attack: the injected
  text *is* in the source, which is why `check_answer` refuses answers that cite a block with
  an instruction (`followed`).
- Disabling the detector's own normalisation changes nothing end to end because the index is
  already normalised (lesson 02); on raw chunks it costs one attack (a06).
- The rules and the attack documents were written together: detection numbers are optimistic
  until attacks written by someone else are added.
- No real-model numbers were produced: embedding results and real-model injection results are
  left to `notebooks/benchmark_real_embeddings.ipynb` and `examples/injection_test.py`.

Deviations from the Day 7 brief:
- The notebook installs `parsrag` from a shallow clone of the repository (the fixtures are not
  in the wheel), not with `pip install git+…`.
- The notebook check runs the offline cells with a small stdlib executor instead of `nbclient`
  (no new dependency); matplotlib comes from the existing `dev` extra.

### Next session (Day 8) — optional ideas

- Fill lesson 05's real-model table from actual notebook runs (contributors' PRs).
- Attacks written by someone who has not seen the rules, and a held-out split of the attack
  fixtures; an LLM-judge detector compared with the rules.
- **A Telegram bot example on top of `parsrag`** (`PersianRAG` + streaming answers with
  citations), optional dependency, never run in CI.
- **English summaries of each lesson** (one short page per lesson on the docs site or in
  `lessons/*/README.en.md`), so non-Persian readers can follow the library design.
- Still open from earlier days: a development split for `qa_paraphrase.jsonl` so rule-based or
  tuned components can be fitted without leakage; growing both QA sets; a relative BM25
  confidence for abstention; Redis-backed cache / rate limit as an optional extra; self-hosting
  Vazirmatn and Mermaid on the docs site; the first real release (follow `docs/RELEASING.md`).

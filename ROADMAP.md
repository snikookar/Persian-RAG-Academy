# Roadmap

Each lesson builds one `parsrag` module, ships a Persian `README.fa.md`, a
`starter/` with TODOs and `tests/` used by the grader.

## Lessons

- [x] **00 — Setup**: environment, tests, grader, FakeLLM/HashingEmbedding, optional real API / Ollama → `parsrag.doctor`
- [x] **01 — RAG in 50 lines**: load → chunk → embed → retrieve → generate → `parsrag.naive_rag`
- [x] **02 — Persian normalization**: Arabic→Persian letters, digits, ZWNJ, diacritics; measured impact → `parsrag.normalize`
- [ ] **03 — Loading Persian documents**: PDF / DOCX / HTML, RTL and presentation-form issues → `parsrag.loaders`
- [ ] **04 — Chunking Persian text**: sentence/paragraph-aware chunking → `parsrag.chunking`
- [ ] **05 — Embeddings**: comparing HashingEmbedding vs. real multilingual models → `parsrag.embeddings`
- [ ] **06 — Vector stores**: in-memory → FAISS → pgvector → `parsrag.stores`
- [ ] **07 — Hybrid search**: Persian BM25 + dense + Reciprocal Rank Fusion → `parsrag.retrieval`
- [ ] **08 — Reranking** → `parsrag.rerank`
- [ ] **09 — Grounded generation**: citations and honest "I don't know" → `parsrag.generate`
- [ ] **10 — Evaluation**: Recall@k, MRR, faithfulness, LLM-as-judge → `parsrag.eval`
- [ ] **11 — Agentic RAG with LangGraph** → `parsrag.agent`
- [ ] **12 — Production API**: FastAPI, streaming, cache, Docker → `parsrag.api`
- [ ] **13 — Capstone**: RAG over your own documents + UI

## Baseline numbers (fixtures, HashingEmbedding)

| Pipeline | hit@1 | hit@3 |
|---|---|---|
| Lesson 01 naive (chunk 400) | 0.750 | 0.958 |
| Lesson 01 naive (chunk 300) | 0.792 | 0.917 |
| + Lesson 02 normalization (chunk 300) | 0.958 | 1.000 |

## Session log

### Day 1 — 2026-10-31: foundation + lessons 00–02

Built:
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

### Next session (Day 2)

- Lesson 03 — loaders: `parsrag.loaders` for TXT/MD/HTML/PDF/DOCX behind the `loaders` extra;
  create small original PDF/DOCX/HTML fixtures (generated in-repo, no downloads); show
  presentation-form and RTL/bidi problems and how `normalize` fixes them.
- Lesson 04 — chunking: sentence splitter for Persian punctuation (`.`، `؟`، `!`، `؛`),
  paragraph/recursive chunkers, measure hit@k vs. the naive fixed-window baseline.
- Consider growing `qa.jsonl` (more questions per document) before lessons 05–07 comparisons.

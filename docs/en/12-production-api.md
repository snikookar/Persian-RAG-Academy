# 12 — Production API

**Persian lesson:** [API تولیدی](../lessons/12-production-api.md) · **Module:** `parsrag.api`

## What you build

A FastAPI service around `PersianRAG`: `/health`, `/ingest` (JSON) and `/ingest/files`
(uploads), `/documents`, `/ask` and `/ask/stream` (Server-Sent Events), an answer cache keyed by
the *normalised* question and the index version, token-bucket rate limiting per API key or IP,
JSON logs with per-stage latency (the question is hashed, never logged), clear error codes, a
multi-stage non-root Docker image and a compose file with PostgreSQL + pgvector.

## Key ideas

- **Normalise the cache key.** Persian users type the same question with Arabic letters,
  without «؟» or with extra spaces; a raw-text cache misses all of them.
- **Stream for perceived latency.** Send sources first, then tokens, then a final `done` event
  whose citations are validated on the full text. Since the question is in a POST body, the
  browser uses `fetch` with a `ReadableStream`, not `EventSource`.
- **Lock retrieval, not generation.** Indexing mutates BM25 and the store, so retrieval runs
  under a lock; slow model calls do not block each other.
- **Fail clearly.** 409 for an empty index or duplicates, 413/415 for uploads, 429 with
  `Retry-After`, 502 for a model error and 504 for a model timeout.

## Measured results

One run on the course machine, in-process (no network), 117 questions from both sets
(`python examples/api_latency.py`):

| Request | median ms | p95 ms |
|---|---|---|
| cache miss (full HTTP round trip, FakeLLM) | 9.08 | 13.23 |
| … of which retrieval (server timing) | 6.15 | 9.67 |
| cache hit (same question again) | 1.90 | 2.56 |

Spelling variants (Arabic YEH/KAF, no «؟», extra spaces) hitting the cache: raw-text key 0/117,
normalised key 117/117.

With a *simulated* model of 30 ms per word (a number chosen for the demo, not a measurement of
a real model): cache miss 251.80 / 880.83 ms, cache hit 2.47 / 3.71 ms (median / p95). Streaming
with the same simulated model, server side, 19 answerable questions: first token 37.21 / 42.92
ms, whole answer 345.86 / 1130.20 ms.

`TestClient` buffers streamed responses, so time to first token is measured on the server's
generator.

## Takeaway

For Persian, cache on the normalised question; stream to cut perceived latency; and make every
failure a stable, documented error code.

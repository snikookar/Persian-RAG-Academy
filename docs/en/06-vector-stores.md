# 06 — Vector stores

**Persian lesson:** [انباره‌های برداری](../lessons/06-vector-stores.md) · **Module:** `parsrag.stores`

## What you build

One `VectorStore` protocol (`add`, `search`, `search_by_vector`, `len`) with three
implementations: an exact in-memory NumPy store with metadata filters and save/load,
`FaissVectorStore` (flat, HNSW and IVF indexes) and `PgVectorStore` for PostgreSQL with the
pgvector extension.

## Key ideas

- **Same interface everywhere.** The rest of the pipeline never knows where vectors live, so you
  can start in memory and move to FAISS or a database without touching retrieval code.
- **Metadata filters.** Every chunk carries `doc_id`, `section` and `page` from lessons 03–04.
  A filter is a dict (equality, or a list meaning "any of") or, for in-memory and FAISS stores,
  any predicate. pgvector translates dict filters to SQL.
- **Exact vs approximate.** Flat search compares the query with every vector. HNSW builds a
  layered small-world graph (`m` neighbours per node, `ef_search` candidates per step): no
  training, but slow to build and memory-hungry. IVF clusters vectors into `nlist` cells and
  searches `nprobe` of them; it must be trained on enough vectors first.
- **Approximate search plus a filter can miss results.** With HNSW and a very selective
  post-filter, even fetching the whole index can come up short; `FaissVectorStore` then falls
  back to exact search over the matching chunks.

## Measured results

Synthetic 384-dimensional unit vectors around 50 random cluster centres, 200 queries, k = 10,
single thread, one run on the course's CPU (`python examples/vector_store_latency.py`; recall is
against exact search):

| n = 2,000 | build (s) | query (ms) | recall@10 |
|---|---|---|---|
| in-memory (NumPy, exact) | 0.006 | 0.359 | 1.000 |
| FAISS flat (exact) | 0.003 | 0.235 | 1.000 |
| FAISS HNSW (m=32, ef=64) | 0.317 | 0.306 | 0.995 |
| FAISS IVF (nlist=20, nprobe=8) | 0.017 | 0.167 | 0.934 |

| n = 10,000 | build (s) | query (ms) | recall@10 |
|---|---|---|---|
| in-memory (NumPy, exact) | 0.027 | 2.998 | 1.000 |
| FAISS flat (exact) | 0.039 | 1.336 | 1.000 |
| FAISS HNSW (m=32, ef=64) | 3.173 | 0.433 | 0.993 |
| FAISS IVF (nlist=100, nprobe=8) | 0.163 | 0.216 | 0.993 |

Exact search is perfectly fine for small collections; approximate indexes pay off as the
collection grows, at the cost of build time and a little recall. Timings will differ on your
machine; the ratios are what matter.

## Takeaway

Choose the store by collection size and operations, not by fashion: in memory up to tens of
thousands of chunks, FAISS for speed at scale, a database when you need persistence,
concurrency and SQL filters.

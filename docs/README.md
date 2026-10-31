# docs

Extra documentation, images and benchmark charts for the course live here.

- `images/chunking_mrr.png` — passage-level MRR of the chunking strategies
  (lesson 04), produced by `python examples/chunking_benchmark.py --chart`.
- `images/hybrid_mrr.png` — BM25 vs dense vs hybrid on both question sets
  (lesson 07), `python examples/hybrid_search.py --chart`.
- `images/rerank_r1.png` — R@1 with and without reranking (lesson 08),
  `python examples/reranking.py --chart`.
- `eval/baseline.json` — the metrics the CI eval gate protects (lesson 10);
  update with `python -m parsrag.eval gate --baseline docs/eval/baseline.json --update`.
- `eval/report.md` — the full report of the course pipeline,
  `python -m parsrag.eval run --markdown docs/eval/report.md`.

(The full project overview arrives with the Day 5 README.)

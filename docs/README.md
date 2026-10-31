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
- `images/agent_graph.mmd` — LangGraph's own Mermaid drawing of the lesson 11 agent,
  `python examples/agentic_rag.py --mermaid docs/images/agent_graph.mmd`.
- `images/capstone_ui.png` — the web UI of lesson 13 (`parsrag serve`).

- `images/readme_*.png` and `eval/readme_numbers.json` — the result charts in the
  README and on the docs site, and the numbers behind them,
  `python examples/make_readme_charts.py`.
- `images/social-preview.png` — the GitHub social preview (1280×640),
  `python scripts/make_social_preview.py`.
- `RELEASING.md` — how a release is built and published (maintainers).
- Everything else (`index.md`, `*.fa.md`, `api/`, `stylesheets/`, `hooks.py`) is the
  Persian documentation site; build it with `pip install -e ".[docs]"` and
  `mkdocs serve`. The lesson pages are generated from `lessons/*/README.fa.md`.

The project overview is the [README](../README.md) ([فارسی](../README.fa.md)).

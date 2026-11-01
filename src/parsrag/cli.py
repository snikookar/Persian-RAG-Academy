"""The ``parsrag`` command line (lesson 13).

    parsrag init                                  # write a commented parsrag.toml
    parsrag ingest docs/                          # index a folder, save to .parsrag/index
    parsrag ask "شرایط مرجوعی کالا چیست؟"           # answer with sources
    parsrag eval my_questions.jsonl --calibrate   # report (+ set the «نمی‌دانم» threshold)
    parsrag eval my_questions.jsonl --baseline eval/baseline.json   # the eval gate
    parsrag serve --port 8000                     # API + web UI (needs the api extra)

Every command reads ``--config`` (default ``parsrag.toml`` if it exists) and
uses the index folder from ``--index`` or the config's ``[index] path``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from parsrag.config import RAGConfig, load_config, make_embedding, make_llm
from parsrag.rag import PersianRAG, answer_payload

DEFAULT_CONFIG = "parsrag.toml"

CONFIG_TEMPLATE = """# parsrag configuration — see `parsrag.config` and lesson 13.

[documents]
paths = ["docs"]               # folders or files, relative to this file
pattern = "*"
recursive = true

[index]
path = ".parsrag/index"        # written by `parsrag ingest`

[chunking]
size = 400                     # lesson 04
overlap = 50
heading_context = true

[retrieval]
k = 4                          # passages shown to the model (lesson 09)
fetch_k = 20                   # candidates for the reranker (lesson 08)
rerank = "lexical"             # "lexical" | "cross-encoder" | "none"
stem = true                    # light stemming in BM25 (lesson 07)
# min_score = 4.0              # set by `parsrag eval --calibrate` (lesson 09)

[embedding]
provider = "hashing"           # "hashing" (offline) | "st" | "openai" | "tfidf"
# model = "intfloat/multilingual-e5-base"

[llm]
provider = "fake"              # "fake" (offline, extractive) | "ollama" | "openai"
# model = "qwen2.5:7b"
# base_url = "http://localhost:11434/v1"
# api_key_env = "OPENAI_API_KEY"
timeout = 60

[security]
enabled = false                # prompt-injection guard (lesson 14)
# action = "drop"              # "drop" | "flag" suspicious chunks
"""


def _config(args: argparse.Namespace) -> RAGConfig:
    path = args.config or (DEFAULT_CONFIG if Path(DEFAULT_CONFIG).is_file() else None)
    return load_config(path)


def _index_dir(args: argparse.Namespace, cfg: RAGConfig) -> Path:
    return Path(args.index) if args.index else cfg.resolve(cfg.index.path)


def _load(args: argparse.Namespace) -> tuple[PersianRAG, RAGConfig, Path]:
    cfg = _config(args)
    index = _index_dir(args, cfg)
    if not (index / "rag.json").is_file():
        raise SystemExit(
            f"No index in {index}. Run `parsrag ingest <folder>` first. / "
            "هنوز نمایه‌ای ساخته نشده؛ اول `parsrag ingest` را اجرا کنید."
        )
    from parsrag.chunking import RecursiveChunker
    from parsrag.rag import _make_reranker

    rag = PersianRAG.load(
        index,
        embedding=make_embedding(cfg.embedding),
        llm=make_llm(cfg.llm),
        chunker=RecursiveChunker(cfg.chunking.size, cfg.chunking.overlap),
        reranker=_make_reranker(cfg.retrieval.rerank, cfg.retrieval.cross_encoder),
        stem=cfg.retrieval.stem,
        config=cfg,
    )
    if cfg.retrieval.min_score is not None:
        rag.min_score = cfg.retrieval.min_score
    return rag, cfg, index


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _config(args)
    rag = PersianRAG.from_config(cfg, load_documents=not args.paths)
    for path in args.paths:
        rag.add_path(path, cfg.documents.pattern, cfg.documents.recursive)
    if not rag.documents:
        print("no supported documents found (TXT, MD, HTML, DOCX, PDF)", file=sys.stderr)
        return 1
    if args.calibrate:
        threshold = rag.calibrate(args.calibrate)
        print(f"«نمی‌دانم» threshold calibrated on {args.calibrate}: {threshold:.3f}")
    index = _index_dir(args, cfg)
    rag.save(index)
    print(f"indexed {len(rag.documents)} documents, {len(rag)} chunks → {index}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    rag, _, _ = _load(args)
    if args.k:
        rag.k = args.k
    payload = answer_payload(rag, rag.ask(args.question))
    if args.json:
        print(json.dumps(payload.model_dump(), ensure_ascii=False, indent=2))
        return 0
    print(payload.answer)
    for c in payload.citations:
        where = " › ".join(str(x) for x in (c["doc_id"], c.get("section")) if x)
        passage = " ".join(c["passage"].split())
        print(f"  [{c['number']}] {where}: {passage[:160]}{'…' if len(passage) > 160 else ''}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from parsrag.eval.dataset import load_qa, validate_qa
    from parsrag.eval.gate import (
        all_changes,
        compare_metrics,
        format_changes,
        load_baseline,
        save_baseline,
    )
    from parsrag.eval.pipeline import save_report

    rag, _, index = _load(args)
    problems = validate_qa(load_qa(args.qa), rag.documents)
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    if problems:
        print(
            f"{len(problems)} problem(s) in {args.qa} — fix them before trusting the numbers",
            file=sys.stderr,
        )
    if args.calibrate or args.calibrate_on:
        source = args.calibrate_on or args.qa
        threshold = rag.calibrate(source)
        note = "" if args.calibrate_on else " (same set: optimistic, see lesson 10)"
        print(f"«نمی‌دانم» threshold calibrated on {source}: {threshold:.3f}{note}")
        if args.save:
            rag.save(index)
    report = rag.evaluate(args.qa, name=args.name)
    print(report.to_markdown())
    save_report([report], args.json, args.markdown)
    if not args.baseline:
        return 0
    current = {args.name: report.metrics}
    if args.update_baseline:
        save_baseline(args.baseline, current, args.tolerance or 0.02)
        print(f"\nbaseline written to {args.baseline}")
        return 0
    baseline, tolerance = load_baseline(args.baseline)
    tolerance = args.tolerance if args.tolerance is not None else tolerance
    failures = compare_metrics(baseline, current, tolerance)
    print("\n" + format_changes(all_changes(baseline, current), failures))
    if failures:
        print(f"\n✘ {len(failures)} metric(s) dropped by more than {tolerance} — gate failed.")
        return 1
    print(f"\n✔ no metric dropped by more than {tolerance} — gate passed.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    try:
        import uvicorn

        from parsrag.api import Settings, create_app
    except ImportError:
        print("`parsrag serve` needs: pip install 'parsrag[api]'", file=sys.stderr)
        return 1
    settings = Settings.from_env()
    overrides: dict[str, Any] = {"config": args.config, "index": args.index, "docs": args.docs}
    settings = settings.model_copy(update={k: v for k, v in overrides.items() if v})
    if not settings.config and Path(DEFAULT_CONFIG).is_file():
        settings.config = DEFAULT_CONFIG
    if not settings.index and not settings.docs:
        cfg = load_config(settings.config) if settings.config else RAGConfig()
        index = cfg.resolve(cfg.index.path)
        if (index / "rag.json").is_file():
            settings.index = str(index)
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parsrag", description="Persian RAG from the command line"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--config", help=f"TOML/JSON config (default: ./{DEFAULT_CONFIG} if present)"
        )
        p.add_argument("--index", help="index folder (default: the config's [index] path)")
        return p

    p = sub.add_parser("init", help="write a commented parsrag.toml")
    p.add_argument("path", nargs="?", default=DEFAULT_CONFIG)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = common(sub.add_parser("ingest", help="index folders/files and save the index"))
    p.add_argument("paths", nargs="*", help="folders or files (default: the config's paths)")
    p.add_argument("--calibrate", help="QA JSONL to calibrate the «نمی‌دانم» threshold on")
    p.set_defaults(func=cmd_ingest)

    p = common(sub.add_parser("ask", help="answer one question with sources"))
    p.add_argument("question")
    p.add_argument("--k", type=int, help="passages shown to the model")
    p.add_argument("--json", action="store_true", help="print the full answer as JSON")
    p.set_defaults(func=cmd_ask)

    p = common(sub.add_parser("eval", help="evaluate on a QA JSONL file (lesson 10 report)"))
    p.add_argument("qa")
    p.add_argument("--name", default="custom", help="report/baseline name")
    p.add_argument("--calibrate", action="store_true", help="calibrate on the same file")
    p.add_argument("--calibrate-on", help="calibrate on another QA file (recommended)")
    p.add_argument("--save", action="store_true", help="save the calibrated threshold")
    p.add_argument("--json", help="write the report with per-question records here")
    p.add_argument("--markdown", help="write a Markdown summary here")
    p.add_argument("--baseline", help="gate against this baseline JSON")
    p.add_argument("--update-baseline", action="store_true", help="write the baseline instead")
    p.add_argument("--tolerance", type=float, help="allowed drop (default: the baseline's)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="run the API and web UI (needs the api extra)")
    p.add_argument("--config")
    p.add_argument("--index")
    p.add_argument("--docs", help="index this folder at startup instead of loading an index")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""The ``parsrag`` command line (lesson 13).

    parsrag init [--preset ollama]                # write a commented parsrag.toml
    parsrag ingest docs/                          # index a folder, save to .parsrag/index
    parsrag ingest docs/ --prune                  # again: only new/changed files are embedded
    parsrag remove some-doc-id                    # delete a document from the index
    parsrag ask "شرایط مرجوعی کالا چیست؟"           # answer with sources
    parsrag ask "..." --filter format=pdf         # only search matching documents
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

from parsrag.config import (
    PRESETS,
    RAGConfig,
    config_template,
    load_config,
    make_embedding,
    make_llm,
)
from parsrag.rag import IndexUpdate, PersianRAG, answer_payload

DEFAULT_CONFIG = "parsrag.toml"

#: The offline preset (kept under its 0.1.0 name).
CONFIG_TEMPLATE = config_template("offline")


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
    path.write_text(config_template(args.preset), encoding="utf-8")
    print(f"wrote {path} ({args.preset} preset)")
    if args.preset != "offline":
        print("edit the <...> placeholders (model names) before `parsrag ingest`")
    if args.preset == "openai-compatible":
        print("the API key is read from the environment variable in api_key_env, never the file")
    return 0


def _read_documents(paths: list[Path], cfg: RAGConfig) -> list[Any]:
    from parsrag.loaders import load_directory, load_document

    docs: list[Any] = []
    for p in paths:
        if p.is_dir():
            docs.extend(load_directory(p, cfg.documents.pattern, recursive=cfg.documents.recursive))
        elif p.is_file():
            docs.append(load_document(p))
        else:
            raise SystemExit(f"no such file or folder: {p}")
    return docs


def _compatible(index: Path, cfg: RAGConfig) -> str | None:
    """``None`` if the saved index can be updated in place, else why it must be rebuilt."""
    info = PersianRAG.index_info(index)
    if info["format_version"] < 2:
        return "the index was saved by parsrag 0.1.0"
    saved = info.get("config") or {}
    for section in ("chunking", "embedding"):
        if saved.get(section) != getattr(cfg, section).model_dump(mode="json"):
            return f"[{section}] changed since the index was built"
    if info.get("stem") != cfg.retrieval.stem:
        return "[retrieval] stem changed since the index was built"
    return None


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _config(args)
    index = _index_dir(args, cfg)
    paths = [Path(p) for p in args.paths] or [cfg.resolve(p) for p in cfg.documents.paths]
    docs = _read_documents(paths, cfg)
    if not docs:
        print("no supported documents found (TXT, MD, HTML, DOCX, PDF)", file=sys.stderr)
        return 1
    reason = "--rebuild" if args.rebuild else None
    if not reason and (index / "rag.json").is_file():
        reason = _compatible(index, cfg)
        if reason:
            print(f"rebuilding the index: {reason}")
    if (index / "rag.json").is_file() and not reason:
        rag, _, _ = _load(args)
        report = rag.upsert_documents(docs, prune=args.prune)
    else:
        rag = PersianRAG.from_config(cfg, load_documents=False)
        report = rag.upsert_documents(docs)
    if args.calibrate:
        threshold = rag.calibrate(args.calibrate)
        print(f"«نمی‌دانم» threshold calibrated on {args.calibrate}: {threshold:.3f}")
    if args.save:
        rag.save(index)
    print(report.summary())
    where = f"→ {index}" if args.save else "(not saved: --no-save)"
    print(f"indexed {len(rag.documents)} documents, {len(rag)} chunks {where}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    rag, _, index = _load(args)
    missing = [i for i in args.ids if i not in rag.documents]
    for doc_id in missing:
        print(f"not in the index: {doc_id}", file=sys.stderr)
    removed = rag.remove_documents(args.ids, missing_ok=True)
    report = IndexUpdate(removed=[i for i in args.ids if i not in missing], chunks_removed=removed)
    rag.save(index)
    print(report.summary())
    return 1 if missing and not report.removed else 0


def parse_filters(items: list[str] | None) -> dict[str, Any] | None:
    """``["lang=fa", "page=[1,2]"]`` → ``{"lang": "fa", "page": [1, 2]}`` (JSON values allowed)."""
    if not items:
        return None
    filters: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--filter expects key=value, got {item!r}")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        filters[key.strip()] = value
    return filters


def cmd_ask(args: argparse.Namespace) -> int:
    rag, _, _ = _load(args)
    if args.k:
        rag.k = args.k
    payload = answer_payload(rag, rag.ask(args.question, filters=parse_filters(args.filter)))
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
    p.add_argument(
        "--preset",
        choices=PRESETS,
        default="offline",
        help="offline (default, no models) | ollama (local models) | openai-compatible",
    )
    p.set_defaults(func=cmd_init)

    p = common(sub.add_parser("ingest", help="index folders/files and save the index"))
    p.add_argument("paths", nargs="*", help="folders or files (default: the config's paths)")
    p.add_argument("--calibrate", help="QA JSONL to calibrate the «نمی‌دانم» threshold on")
    p.add_argument(
        "--rebuild", action="store_true", help="ignore the saved index and index everything again"
    )
    p.add_argument(
        "--prune", action="store_true", help="remove indexed documents not found in these paths"
    )
    p.add_argument(
        "--no-save", dest="save", action="store_false", help="index without writing the index"
    )
    p.set_defaults(func=cmd_ingest)

    p = common(sub.add_parser("remove", help="delete documents from the saved index by id"))
    p.add_argument("ids", nargs="+", help="document ids (see `parsrag ask --json` or /documents)")
    p.set_defaults(func=cmd_remove)

    p = common(sub.add_parser("ask", help="answer one question with sources"))
    p.add_argument("question")
    p.add_argument("--k", type=int, help="passages shown to the model")
    p.add_argument("--json", action="store_true", help="print the full answer as JSON")
    p.add_argument(
        "--filter",
        action="append",
        metavar="KEY=VALUE",
        help="only use matching documents/chunks, e.g. doc_id=faq or format=pdf (repeatable)",
    )
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

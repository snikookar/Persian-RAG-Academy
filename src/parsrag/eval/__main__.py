"""``python -m parsrag.eval`` — evaluation report and regression gate (lesson 10).

    python -m parsrag.eval run  --json docs/eval/report.json --markdown docs/eval/report.md
    python -m parsrag.eval gate --baseline docs/eval/baseline.json
    python -m parsrag.eval gate --baseline docs/eval/baseline.json --update   # accept new numbers

Both commands evaluate the offline course pipeline
(:func:`parsrag.eval.pipeline.build_course_pipeline`) on ``qa.jsonl``
("original") and ``qa_paraphrase.jsonl`` ("paraphrase"). ``gate --current
report.json`` compares an existing report instead of running the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parsrag.eval.gate import (
    all_changes,
    compare_metrics,
    format_changes,
    load_baseline,
    save_baseline,
)
from parsrag.eval.pipeline import PipelineReport

SETS = {"original": "qa.jsonl", "paraphrase": "qa_paraphrase.jsonl"}

#: The metrics the gate watches by default (all "higher is better").
GATED = (
    "retrieval_r@1",
    "retrieval_mrr",
    "answer_f1",
    "citation_precision",
    "faithfulness",
    "abstention_f1",
)


def run_course_pipeline() -> list[PipelineReport]:
    from parsrag.eval.benchmark import load_fixture_benchmark
    from parsrag.eval.pipeline import build_course_pipeline, evaluate_pipeline

    generator = build_course_pipeline()
    return [
        evaluate_pipeline(generator, load_fixture_benchmark(qa_file=f)[1], name=name)
        for name, f in SETS.items()
    ]


def metrics_of(reports: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    return {str(r["name"]): dict(r["metrics"]) for r in reports}  # type: ignore[call-overload]


def cmd_run(args: argparse.Namespace) -> int:
    from parsrag.eval.pipeline import save_report

    reports = run_course_pipeline()
    save_report(reports, args.json, args.markdown)
    for r in reports:
        print(r.to_markdown(), end="\n\n")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    if args.current:
        current = metrics_of(json.loads(Path(args.current).read_text(encoding="utf-8")))
    else:
        current = metrics_of([r.to_dict() for r in run_course_pipeline()])
    if args.update:
        gated = {
            g: {k: round(v, 4) for k, v in m.items() if k in GATED} for g, m in current.items()
        }
        save_baseline(args.baseline, gated, args.tolerance if args.tolerance is not None else 0.02)
        print(f"baseline written to {args.baseline}")
        return 0
    baseline, tolerance = load_baseline(args.baseline)
    if args.tolerance is not None:
        tolerance = args.tolerance
    failures = compare_metrics(baseline, current, tolerance)
    print(format_changes(all_changes(baseline, current), failures))
    print()
    if failures:
        print(f"✘ {len(failures)} metric(s) dropped by more than {tolerance} — gate failed.")
        return 1
    print(f"✔ no metric dropped by more than {tolerance} — gate passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="python -m parsrag.eval", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="evaluate the course pipeline and write a report")
    p_run.add_argument("--json", help="write the full report (with per-question records) here")
    p_run.add_argument("--markdown", help="write a Markdown summary here")
    p_run.set_defaults(func=cmd_run)
    p_gate = sub.add_parser("gate", help="fail if metrics dropped below the baseline")
    p_gate.add_argument("--baseline", required=True, help="baseline JSON file")
    p_gate.add_argument("--current", help="a report JSON from `run --json` (default: run now)")
    p_gate.add_argument(
        "--tolerance", type=float, help="allowed absolute drop (default: baseline's)"
    )
    p_gate.add_argument("--update", action="store_true", help="overwrite the baseline instead")
    p_gate.set_defaults(func=cmd_gate)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

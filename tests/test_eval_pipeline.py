from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsrag import FakeLLM
from parsrag.eval import QAItem
from parsrag.eval.__main__ import main as eval_main
from parsrag.eval.gate import all_changes, compare_metrics, format_changes, load_baseline
from parsrag.eval.generation import LLMJudge
from parsrag.eval.pipeline import evaluate_pipeline, save_report
from parsrag.generate import GroundedGenerator
from parsrag.retrieval import BM25Retriever
from parsrag.types import Chunk

BASELINE = Path(__file__).resolve().parents[1] / "docs" / "eval" / "baseline.json"


def tiny_generator() -> GroundedGenerator:
    chunks = [
        Chunk(id="ship#0", doc_id="ship", text="هزینهٔ پیک در تهران 45 هزار تومان است."),
        Chunk(id="lib#0", doc_id="lib", text="کتابخانه تا ۸ شب باز است."),
    ]
    bm25 = BM25Retriever(chunks)
    return GroundedGenerator(bm25, FakeLLM(mode="extractive", cite=True), k=2, min_score=0.5)


def tiny_qa() -> list[QAItem]:
    return [
        QAItem(id="a", question="هزینهٔ پیک در تهران چقدر است؟", answer="۴۵ هزار تومان",
               doc_id="ship", evidence="پیک در تهران 45 هزار"),
        QAItem(id="b", question="کتابخانه تا کی باز است؟", answer="تا ۸ شب",
               doc_id="lib", evidence="تا ۸ شب باز"),
        QAItem(id="c", question="پارکینگ دارد؟", answer="-"),
    ]  # fmt: skip


def test_evaluate_pipeline_on_a_tiny_corpus(tmp_path: Path) -> None:
    judge = LLMJudge(FakeLLM(["امتیاز: 10\nدلیل: درست"]))
    report = evaluate_pipeline(
        tiny_generator(), tiny_qa(), name="tiny", judge=judge, n_resamples=200
    )
    m = report.metrics
    assert m["retrieval_r@1"] == 1.0 and m["citation_precision"] == 1.0
    assert m["faithfulness"] == 1.0 and m["judge_faithfulness"] == 1.0
    assert m["abstention_f1"] == 1.0 and m["answered_rate"] == 1.0
    assert 0 < m["answer_f1"] < 1
    assert [r.abstained for r in report.records] == [False, False, True]
    assert "| answer_f1 |" in report.to_markdown()
    save_report([report], tmp_path / "r.json", tmp_path / "r.md")
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data[0]["name"] == "tiny" and len(data[0]["records"]) == 3
    assert "# Evaluation report" in (tmp_path / "r.md").read_text(encoding="utf-8")


def test_gate_formatting() -> None:
    base = {"s": {"m": 0.5}}
    cur = {"s": {"m": 0.4}}
    table = format_changes(all_changes(base, cur), compare_metrics(base, cur))
    assert "| s | m | 0.500 | 0.400 | -0.100 | ✘ |" in table
    assert "missing" in format_changes(all_changes(base, {}), [])


def test_gate_cli_passes_on_the_committed_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    assert eval_main(["gate", "--baseline", str(BASELINE)]) == 0
    assert "gate passed" in capsys.readouterr().out


def test_gate_cli_fails_on_a_regression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    metrics, _ = load_baseline(BASELINE)
    worse = {g: {k: v - 0.1 for k, v in m.items()} for g, m in metrics.items()}
    current = tmp_path / "current.json"
    current.write_text(
        json.dumps([{"name": g, "metrics": m} for g, m in worse.items()]), encoding="utf-8"
    )
    assert eval_main(["gate", "--baseline", str(BASELINE), "--current", str(current)]) == 1
    assert "gate failed" in capsys.readouterr().out
    loose = ["gate", "--baseline", str(BASELINE), "--current", str(current), "--tolerance", "0.2"]
    assert eval_main(loose) == 0


def test_gate_cli_update_and_run(tmp_path: Path) -> None:
    out = tmp_path / "baseline.json"
    current = tmp_path / "report.json"
    assert eval_main(["run", "--json", str(current), "--markdown", str(tmp_path / "r.md")]) == 0
    assert eval_main(["gate", "--baseline", str(out), "--current", str(current), "--update"]) == 0
    assert load_baseline(out) == load_baseline(BASELINE)

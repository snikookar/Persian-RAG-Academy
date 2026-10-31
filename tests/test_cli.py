"""The `parsrag` command line (lesson 13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsrag.cli import main
from parsrag.eval.benchmark import FIXTURES

QA = str(FIXTURES / "qa.jsonl")


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_init_writes_a_readable_config(workdir: Path) -> None:
    assert main(["init"]) == 0
    assert "[retrieval]" in (workdir / "parsrag.toml").read_text(encoding="utf-8")
    assert main(["init"]) == 1  # no silent overwrite
    assert main(["init", "--force"]) == 0


def test_ingest_ask_eval(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest", str(FIXTURES / "corpus"), "--calibrate", QA]) == 0
    out = capsys.readouterr().out
    assert "indexed 25 documents" in out and (workdir / ".parsrag/index/rag.json").is_file()

    assert main(["ask", "هزینهٔ پیک در تهران چقدر است؟"]) == 0
    out = capsys.readouterr().out
    assert "45" in out and "[1] kahroba-shipping" in out

    assert main(["ask", "هزینهٔ پیک در تهران چقدر است؟", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["citations"][0]["passage"] and not data["abstained"]

    # the saved threshold is the one calibrated on qa.jsonl: same numbers as the lesson 10 gate
    assert main(["eval", QA, "--name", "original", "--json", "r.json"]) == 0
    report = json.loads((workdir / "r.json").read_text(encoding="utf-8"))[0]
    baseline = json.loads((Path(FIXTURES).parents[1] / "docs/eval/baseline.json").read_text())
    assert report["metrics"]["answer_f1"] == pytest.approx(
        baseline["metrics"]["original"]["answer_f1"], abs=1e-4
    )


def test_eval_gate(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["ingest", str(FIXTURES / "corpus")])
    gate = ["eval", QA, "--calibrate-on", QA, "--baseline", "b.json"]
    assert main([*gate, "--update-baseline"]) == 0
    assert main(gate) == 0
    assert "gate passed" in capsys.readouterr().out
    # no calibration → the system never abstains → abstention F1 drops → the gate fails
    assert main(["eval", QA, "--baseline", "b.json"]) == 1
    assert "gate failed" in capsys.readouterr().out


def test_config_file_and_errors(workdir: Path) -> None:
    docs = workdir / "docs"
    docs.mkdir()
    (docs / "cafe.md").write_text("# کافه\nکافه جمعه‌ها تعطیل است.", encoding="utf-8")
    (workdir / "parsrag.json").write_text(
        json.dumps({"documents": {"paths": ["docs"]}, "index": {"path": "idx"}}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="parsrag ingest"):
        main(["ask", "?", "--config", "parsrag.json"])
    assert main(["ingest", "--config", "parsrag.json"]) == 0
    assert (workdir / "idx" / "documents.jsonl").is_file()
    assert main(["ask", "کافه کی تعطیل است؟", "--config", "parsrag.json"]) == 0
    (workdir / "empty").mkdir()
    assert main(["ingest", "empty", "--config", "parsrag.json"]) == 1

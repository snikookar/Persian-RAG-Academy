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
    assert "200" in out and "[1] kahroba-shipping" in out

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


def test_eval_warns_about_bad_questions(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["ingest", str(FIXTURES / "corpus")])
    bad = workdir / "bad.jsonl"
    rows = [
        {"id": "a", "question": "هزینهٔ پیک؟", "answer": "۲۰۰", "doc_id": "kahroba-shipping",
         "evidence": "هزینهٔ پیک در تهران ۹۹ هزار"},
        {"id": "a", "question": "؟", "answer": "-", "doc_id": "nope", "evidence": "x"},
    ]  # fmt: skip
    bad.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    assert main(["eval", str(bad)]) == 0
    err = capsys.readouterr().err
    assert "evidence not found" in err and "duplicate id" in err and "unknown doc_id" in err


# ------------------------------------------------- 0.2.0: presets, incremental ingest


@pytest.mark.parametrize("preset", ["offline", "ollama", "openai-compatible"])
def test_init_presets(workdir: Path, preset: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init", "--preset", preset]) == 0
    text = (workdir / "parsrag.toml").read_text(encoding="utf-8")
    assert f"({preset} preset)" in text and "api_key =" not in text
    out = capsys.readouterr().out
    if preset == "offline":
        assert 'provider = "fake"' in text
    else:
        assert "<chat-model>" in text and "placeholders" in out


def test_ingest_is_incremental_and_remove(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = workdir / "docs"
    docs.mkdir()
    (docs / "bakery.md").write_text("# ساعت کار\nنانوایی سپیده ساعت ۶ صبح باز می‌شود.", "utf-8")
    (docs / "clinic.md").write_text("# نوبت\nنوبت درمانگاه نارون تلفنی است.", "utf-8")
    assert main(["ingest", "docs", "--index", "idx"]) == 0
    assert "added 2, updated 0, unchanged 0" in capsys.readouterr().out

    assert main(["ingest", "docs", "--index", "idx"]) == 0
    assert "added 0, updated 0, unchanged 2" in capsys.readouterr().out

    (docs / "clinic.md").write_text("# نوبت\nنوبت درمانگاه نارون اینترنتی است.", "utf-8")
    (docs / "note.txt").write_text("کتابخانهٔ محله ساعت ۹ باز است.", "utf-8")
    (docs / "bakery.md").unlink()
    assert main(["ingest", "docs", "--index", "idx"]) == 0
    assert "added 1, updated 1, unchanged 0, removed 0" in capsys.readouterr().out
    assert main(["ingest", "docs", "--index", "idx", "--prune"]) == 0
    assert "removed 1" in capsys.readouterr().out

    assert main(["ask", "نوبت درمانگاه چطور است؟", "--index", "idx", "--json"]) == 0
    assert "اینترنتی" in json.loads(capsys.readouterr().out)["text"]

    ask = ["ask", "ساعت باز شدن؟", "--index", "idx", "--json", "--filter", "doc_id=note"]
    assert main(ask) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {c["doc_id"] for c in payload["citations"]} == {"note"}

    assert main(["remove", "note", "--index", "idx"]) == 0
    assert "removed 1" in capsys.readouterr().out
    assert main(["remove", "note", "--index", "idx"]) == 1


def test_ingest_rebuilds_when_config_changes(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workdir / "docs").mkdir()
    (workdir / "docs" / "a.md").write_text("# الف\nباغ گیاه‌شناسی شنبه‌ها تعطیل است.", "utf-8")
    cfg = {"documents": {"paths": ["docs"]}, "index": {"path": "idx"}}
    (workdir / "c.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert main(["ingest", "--config", "c.json"]) == 0
    cfg["chunking"] = {"size": 200, "overlap": 20}
    (workdir / "c.json").write_text(json.dumps(cfg), encoding="utf-8")
    capsys.readouterr()
    assert main(["ingest", "--config", "c.json"]) == 0
    out = capsys.readouterr().out
    assert "rebuilding the index: [chunking] changed" in out and "added 1" in out
    assert main(["ingest", "--config", "c.json", "--rebuild"]) == 0
    assert "rebuilding the index: --rebuild" not in capsys.readouterr().out  # silent rebuild


def test_parse_filters() -> None:
    from parsrag.cli import parse_filters

    assert parse_filters(None) is None
    assert parse_filters(["lang=fa", "page=[1, 2]", "n=3"]) == {
        "lang": "fa",
        "page": [1, 2],
        "n": 3,
    }
    with pytest.raises(SystemExit):
        parse_filters(["oops"])

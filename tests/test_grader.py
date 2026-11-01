from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from grader.cli import discover_lessons, find_lesson, main
from grader.loader import LEARNER_PATH_ENV, load_impl
from grader.paths import REPO_ROOT


def _grader(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "grader", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")]),
            "NO_COLOR": "1",
        },
    )


def test_discover_lessons() -> None:
    numbers = [lesson.number for lesson in discover_lessons()]
    assert numbers[:4] == ["00", "01", "02", "03"]
    assert find_lesson("2").slug == "02-normalization"


def test_unknown_lesson_exits() -> None:
    with pytest.raises(SystemExit):
        main(["run", "99"])


def test_load_impl_reference_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LEARNER_PATH_ENV, raising=False)
    mod = load_impl("normalize", "parsrag.normalize")
    assert mod.__name__ == "parsrag.normalize"


def test_load_impl_from_learner_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "helper.py").write_text("VALUE = 41\n")
    (tmp_path / "mine.py").write_text("from helper import VALUE\nANSWER = VALUE + 1\n")
    monkeypatch.setenv(LEARNER_PATH_ENV, str(tmp_path))
    assert load_impl("mine", "parsrag.normalize").ANSWER == 42


def test_load_impl_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LEARNER_PATH_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"normalize\.py"):
        load_impl("normalize", "parsrag.normalize")


@pytest.mark.parametrize("lesson", [lesson.number for lesson in discover_lessons()])
def test_starters_fail_until_completed(lesson: str) -> None:
    proc = _grader("run", lesson)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "not implemented yet" in proc.stdout
    assert "💡" in proc.stdout


def test_reference_passes_via_path() -> None:
    proc = _grader("run", "02", "--path", "src/parsrag")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "آفرین" in proc.stdout


def test_syntax_error_is_reported(tmp_path: Path) -> None:
    (tmp_path / "normalize.py").write_text("def broken(:\n")
    proc = _grader("run", "02", "--path", str(tmp_path))
    assert proc.returncode == 1
    assert "SyntaxError" in proc.stdout


def test_check_solutions() -> None:
    proc = _grader("check-solutions")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "All reference solutions pass." in proc.stdout


def test_skipped_checks_are_reported_not_failed(tmp_path: Path) -> None:
    """A check skipped for a missing optional extra shows its reason and does not fail."""
    lesson = tmp_path / "lessons" / "98-skip-demo"
    (lesson / "tests").mkdir(parents=True)
    (lesson / "tests" / "test_demo.py").write_text(
        "import pytest\n\n"
        "def test_ok():\n    assert True\n\n"
        "def test_needs_extra():\n"
        "    pytest.importorskip('no_such_module_xyz', reason='pip install demo')\n"
    )
    from grader.cli import Lesson, run_tests

    result = run_tests(Lesson(number="98", path=lesson), None)
    assert (result.passed, result.skipped, result.total) == (1, 1, 2)
    assert result.ok


def test_starter_problem_classifies_results() -> None:
    from grader.cli import RunResult, starter_problem

    def res(outcome: str, when: str = "call") -> dict[str, object]:
        return {"outcome": outcome, "when": when, "error": "boom"}

    failing = RunResult(results=[res("passed"), res("failed")], exit_code=1)
    assert starter_problem(failing) is None
    assert "already passes" in str(starter_problem(RunResult(results=[res("passed")])))
    unloadable = RunResult(results=[res("failed", "setup")] * 2, exit_code=1)
    assert "cannot be loaded" in str(starter_problem(unloadable))
    assert starter_problem(RunResult()) == "no checks ran"
    broken = RunResult(collection_errors=["SyntaxError"], exit_code=2)
    assert "cannot be collected" in str(starter_problem(broken))


# ----------------------------------------------------------------- progress (0.2.0)


def test_fingerprint_ignores_line_endings_and_caches(tmp_path: Path) -> None:
    from grader.cli import fingerprint

    a, b = tmp_path / "a", tmp_path / "b"
    for folder, text in ((a, b"x = 1\n"), (b, b"x = 1\r\n")):
        (folder / "__pycache__").mkdir(parents=True)
        (folder / "m.py").write_bytes(text)
    (b / "__pycache__" / "m.cpython.pyc").write_bytes(b"junk")
    (b / "__pycache__" / "ignored.py").write_bytes(b"junk")
    assert fingerprint(a) == fingerprint(b)
    (b / "m.py").write_bytes(b"x = 2\n")
    assert fingerprint(a) != fingerprint(b)


def test_progress_table(tmp_path: Path) -> None:
    import shutil

    done = tmp_path / "00"  # the reference solution as the learner's lesson 00
    done.mkdir()
    shutil.copy(REPO_ROOT / "src" / "parsrag" / "doctor.py", done / "doctor.py")
    started = tmp_path / "01-rag-in-50-lines"  # edited starter: runs and fails
    shutil.copytree(REPO_ROOT / "lessons" / "01-rag-in-50-lines" / "starter", started)
    with (started / "naive_rag.py").open("a", encoding="utf-8") as f:
        f.write("\n# my first edit\n")
    blank = tmp_path / "02-normalization"  # untouched starter: not run at all
    shutil.copytree(REPO_ROOT / "lessons" / "02-normalization" / "starter", blank)
    proc = _grader("progress", "--root", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    lines = {line[:2]: line for line in proc.stdout.splitlines() if line[:2].isdigit()}
    assert "passed" in lines["00"] and "10/10" in lines["00"]
    assert "failed" in lines["01"] and "/17" in lines["01"]
    assert "not started" in lines["02"] and "not started" in lines["14"]
    assert "1 passed, 1 failed, 13 not started" in proc.stdout
    assert "python -m grader run 01" in proc.stdout


def test_progress_bad_root_and_manifest_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import grader.cli as cli

    with pytest.raises(SystemExit):
        main(["progress", "--root", str(tmp_path / "missing")])
    # (CI runs `python -m grader fingerprint-starters --check` on the committed starters;
    # here the manifest is redirected so a learner's edited starters never fail this test.)
    monkeypatch.setattr(cli, "STARTERS_MANIFEST", tmp_path / "starters.json")
    assert main(["fingerprint-starters", "--check"]) == 1
    assert main(["fingerprint-starters"]) == 0
    assert main(["fingerprint-starters", "--check"]) == 0
    assert set(cli.load_manifest()) == {lesson.slug for lesson in discover_lessons()}
    capsys.readouterr()

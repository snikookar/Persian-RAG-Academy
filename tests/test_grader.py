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


@pytest.mark.parametrize("lesson", ["00", "01", "02", "03", "04", "05"])
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

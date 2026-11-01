"""Command-line interface: ``python -m grader {list,run,progress,check-solutions,...}``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grader.loader import LEARNER_PATH_ENV
from grader.paths import LESSONS, REPO_ROOT
from grader.plugin import REPORT_ENV

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(t: str) -> str:
    return _c(t, "32")


def red(t: str) -> str:
    return _c(t, "31")


def dim(t: str) -> str:
    return _c(t, "2")


def bold(t: str) -> str:
    return _c(t, "1")


def fa_num(n: int) -> str:
    return str(n).translate(FA_DIGITS)


# --------------------------------------------------------------------------- lessons


@dataclass
class Lesson:
    number: str
    path: Path

    @property
    def slug(self) -> str:
        return self.path.name

    @property
    def tests(self) -> Path:
        return self.path / "tests"

    @property
    def starter(self) -> Path:
        return self.path / "starter"

    @property
    def title(self) -> str:
        readme = self.path / "README.fa.md"
        if readme.is_file():
            for line in readme.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
        return self.slug


def discover_lessons() -> list[Lesson]:
    lessons = []
    for path in sorted(LESSONS.glob("[0-9][0-9]-*")):
        if path.is_dir():
            lessons.append(Lesson(number=path.name[:2], path=path))
    return lessons


def find_lesson(number: str) -> Lesson:
    wanted = number.zfill(2)
    for lesson in discover_lessons():
        if lesson.number == wanted:
            return lesson
    available = ", ".join(lesson.number for lesson in discover_lessons()) or "none"
    raise SystemExit(f"Lesson {wanted} not found (available: {available}) / درس پیدا نشد")


# ---------------------------------------------------------------------- running tests


@dataclass
class RunResult:
    results: list[dict[str, Any]] = field(default_factory=list)
    collection_errors: list[str] = field(default_factory=list)
    exit_code: int = 0
    output: str = ""

    @property
    def passed(self) -> int:
        return sum(r["outcome"] == "passed" for r in self.results)

    @property
    def skipped(self) -> int:
        return sum(r["outcome"] == "skipped" for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def ok(self) -> bool:
        """Every check passed, except those skipped because an optional extra is missing."""
        return self.exit_code == 0 and self.passed > 0 and self.passed + self.skipped == self.total


def run_tests(lesson: Lesson, learner_path: Path | None) -> RunResult:
    """Run a lesson's tests in a subprocess and collect structured results."""
    if not lesson.tests.is_dir():
        raise SystemExit(f"Lesson {lesson.number} has no tests/ directory")
    with tempfile.TemporaryDirectory() as tmp:
        report_file = Path(tmp) / "report.json"
        env = dict(os.environ)
        env[REPORT_ENV] = str(report_file)
        env.pop(LEARNER_PATH_ENV, None)
        if learner_path is not None:
            env[LEARNER_PATH_ENV] = str(learner_path.resolve())
        extra = [str(REPO_ROOT), str(REPO_ROOT / "src")]
        env["PYTHONPATH"] = os.pathsep.join([*extra, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        cmd = [
            sys.executable, "-m", "pytest", str(lesson.tests),
            "-p", "grader.plugin", "-p", "no:cacheprovider", "-q", "--no-header", "-o", "addopts=",
            "--import-mode=importlib",
        ]  # fmt: skip
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        result = RunResult(exit_code=proc.returncode, output=proc.stdout + proc.stderr)
        if report_file.is_file():
            data = json.loads(report_file.read_text(encoding="utf-8"))
            result.results = data["results"]
            result.collection_errors = data["collection_errors"]
        return result


# --------------------------------------------------------------------------- reports


def _group(results: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(r["nodeid"].split("[")[0], []).append(r)
    return list(groups.items())


def print_report(lesson: Lesson, target: str, result: RunResult, verbose: bool = False) -> None:
    print(bold(f"Lesson {lesson.number}: {lesson.title}"))
    print(dim(f"Testing / در حال بررسی: {target}"))
    print()

    if result.collection_errors:
        print(red("✘ The tests could not even be loaded / تست‌ها اجرا نشدند:"))
        for err in result.collection_errors:
            print(f"    {err}")
        print(dim("  Check for syntax errors or missing imports in your file."))
        print(dim("  فایل خود را از نظر خطای نحوی (syntax error) یا import جاافتاده بررسی کنید."))
        print()

    load_errors = {r["error"] for r in result.results if r["when"] == "setup"}
    if (
        result.results
        and len(load_errors) == 1
        and all(r["when"] == "setup" for r in result.results)
    ):
        print(red("✘ Your code could not be loaded / کد شما بارگذاری نشد:"))
        print(f"    {load_errors.pop()}")
        print()
        print(red(bold(f"Result: 0/{result.total} checks passed")))
        print("فایل خود را درست کنید و دوباره اجرا کنید.")
        return

    for _, items in _group(result.results):
        passed = sum(r["outcome"] == "passed" for r in items)
        skipped = [r for r in items if r["outcome"] == "skipped"]
        first = items[0]
        count = f"({passed}/{len(items)})" if len(items) > 1 else ""
        if skipped and passed + len(skipped) == len(items):
            print(f" {dim('⏭')} {first['title']} {dim('— skipped / رد شد')}")
            print(f"     {dim(skipped[0]['error'])}")
            continue
        if passed == len(items):
            print(f" {green('✔')} {first['title']} {dim(count)}")
            continue
        print(f" {red('✘')} {first['title']} {dim(count)}")
        failures = [r for r in items if r["outcome"] == "failed"]
        for r in failures if verbose else failures[:1]:
            if r["param"]:
                print(f"     {dim('case:')}  {r['param']}")
            print(f"     {dim('error:')} {r['error']}")
        if not verbose and len(failures) > 1:
            print(dim(f"     … and {len(failures) - 1} more (use -v to see all)"))
        hint = first.get("hint") or {}
        if hint.get("fa"):
            print(f"     💡 {hint['fa']}")
        if hint.get("en"):
            print(f"     💡 {hint['en']}")

    print()
    if result.total == 0 and not result.collection_errors:
        print(red("No checks ran. / هیچ بررسی‌ای اجرا نشد."))
        print(result.output[-2000:])
        return
    summary = f"Result: {result.passed}/{result.total} checks passed"
    summary_fa = f"نتیجه: {fa_num(result.passed)} از {fa_num(result.total)} بررسی موفق بود"
    if result.skipped:
        summary += f" ({result.skipped} skipped — install the extras to run them)"
        summary_fa += f" ({fa_num(result.skipped)} بررسی رد شد؛ بسته‌های اختیاری را نصب کنید)"
    if result.ok:
        print(green(bold(f"🎉 {summary}")))
        print(green(f"{summary_fa} — آفرین! این درس را کامل کردید."))
    else:
        print(red(bold(summary)))
        print(f"{summary_fa}. ادامه بدهید، دارید نزدیک می‌شوید!")


# -------------------------------------------------------------------------- commands


def cmd_list(_: argparse.Namespace) -> int:
    for lesson in discover_lessons():
        status = "tests" if lesson.tests.is_dir() else "no tests yet"
        print(f"{lesson.number}  {lesson.slug:<32} {dim(status)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    lesson = find_lesson(args.lesson)
    path = Path(args.path) if args.path else lesson.starter
    if not path.is_dir():
        raise SystemExit(f"--path {path} is not a directory / پوشه وجود ندارد")
    result = run_tests(lesson, path)
    print_report(lesson, str(path), result, verbose=args.verbose)
    return 0 if result.ok else 1


def cmd_check_solutions(args: argparse.Namespace) -> int:
    all_ok = True
    for lesson in discover_lessons():
        if not lesson.tests.is_dir():
            continue
        result = run_tests(lesson, None)
        mark = green("✔") if result.ok else red("✘")
        skipped = dim(f" ({result.skipped} skipped)") if result.skipped else ""
        print(f"{mark} {lesson.number} {lesson.slug:<32} {result.passed}/{result.total}{skipped}")
        if not result.ok:
            all_ok = False
            if args.verbose or result.total == 0:
                print(result.output[-4000:])
            else:
                for r in result.results:
                    if r["outcome"] == "failed":
                        print(f"     {r['nodeid']}: {r['error']}")
    print()
    print(green("All reference solutions pass.") if all_ok else red("Some solutions fail."))
    return 0 if all_ok else 1


def starter_problem(result: RunResult) -> str | None:
    """Why a blank starter is broken, or None if it fails the way it should."""
    if result.collection_errors:
        return "the tests cannot be collected: " + "; ".join(result.collection_errors)
    if result.total == 0:
        return "no checks ran"
    if all(r["when"] == "setup" for r in result.results):
        return "the starter cannot be loaded: " + str(result.results[0]["error"])
    if result.ok:
        return "the blank starter already passes every check"
    return None


def cmd_check_starters(args: argparse.Namespace) -> int:
    """Every blank starter must load and fail: otherwise the exercise teaches nothing."""
    all_ok = True
    for lesson in discover_lessons():
        if not (lesson.tests.is_dir() and lesson.starter.is_dir()):
            continue
        result = run_tests(lesson, lesson.starter)
        problem = starter_problem(result)
        mark = green("✔") if problem is None else red("✘")
        print(f"{mark} {lesson.number} {lesson.slug:<32} {result.passed}/{result.total} pass")
        if problem is not None:
            all_ok = False
            print(f"     {problem}")
            if args.verbose:
                print(result.output[-4000:])
    print()
    print(
        green("Every starter loads and fails until the TODOs are done.")
        if all_ok
        else red("Some starters are broken.")
    )
    return 0 if all_ok else 1


# ----------------------------------------------------------------------- progress

#: SHA-256 fingerprints of the blank starters, so ``progress`` can tell "not started"
#: from "started and failing" without git. Refresh with ``fingerprint-starters``.
STARTERS_MANIFEST = Path(__file__).with_name("starters.json")


def fingerprint(folder: Path) -> str:
    """Hash of every ``.py`` file in ``folder`` (names + contents, line endings normalised)."""
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(folder).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def starter_fingerprints() -> dict[str, str]:
    """``{lesson slug: fingerprint}`` of the blank starters, computed now."""
    return {
        lesson.slug: fingerprint(lesson.starter)
        for lesson in discover_lessons()
        if lesson.starter.is_dir()
    }


def load_manifest() -> dict[str, str]:
    if not STARTERS_MANIFEST.is_file():
        return {}
    data: dict[str, str] = json.loads(STARTERS_MANIFEST.read_text(encoding="utf-8"))
    return data


def learner_folder(lesson: Lesson, root: Path | None) -> Path | None:
    """Where the learner's code for ``lesson`` lives: ``root/<slug>``, ``root/<NN>`` or starter/."""
    if root is None:
        return lesson.starter if lesson.starter.is_dir() else None
    for candidate in (root / lesson.slug, root / lesson.number):
        if candidate.is_dir():
            return candidate
    return None


@dataclass
class Progress:
    lesson: Lesson
    status: str  # "passed" | "failed" | "not started"
    passed: int = 0
    total: int = 0
    skipped: int = 0


def lesson_progress(lesson: Lesson, root: Path | None, manifest: dict[str, str]) -> Progress:
    folder = learner_folder(lesson, root)
    if folder is None or not lesson.tests.is_dir():
        return Progress(lesson, "not started")
    if manifest.get(lesson.slug) == fingerprint(folder):
        return Progress(lesson, "not started")  # still the blank starter: no need to run
    result = run_tests(lesson, folder)
    status = "passed" if result.ok else "failed"
    return Progress(lesson, status, result.passed, result.total, result.skipped)


STATUS_LABELS = {
    "passed": ("✔", "passed", "قبول"),
    "failed": ("✘", "failed", "ناتمام"),
    "not started": ("·", "not started", "شروع‌نشده"),
}


def cmd_progress(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else None
    if root is not None and not root.is_dir():
        raise SystemExit(f"--root {root} is not a directory / پوشه وجود ندارد")
    manifest = load_manifest()
    where = str(root) if root else "lessons/*/starter"
    print(bold(f"Progress / پیشرفت — {where}"))
    print()
    rows = [lesson_progress(lesson, root, manifest) for lesson in discover_lessons()]
    for row in rows:
        mark, en, fa = STATUS_LABELS[row.status]
        colour = green if row.status == "passed" else red if row.status == "failed" else dim
        checks = f"{row.passed}/{row.total}" if row.total else "–"
        skipped = dim(f" ({row.skipped} skipped)") if row.skipped else ""
        label = colour(f"{mark} {en:<12} {fa:<10}")
        print(f"{row.lesson.number}  {row.lesson.slug:<28} {label} {checks:>7}{skipped}")
    counts = {key: sum(r.status == key for r in rows) for key in STATUS_LABELS}
    print()
    print(
        f"{counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['not started']} not started / "
        f"{fa_num(counts['passed'])} قبول، {fa_num(counts['failed'])} ناتمام، "
        f"{fa_num(counts['not started'])} شروع‌نشده"
    )
    nxt = next((r for r in rows if r.status != "passed"), None)
    if nxt is not None:
        print(dim(f"next / درس بعدی: python -m grader run {nxt.lesson.number}"))
    return 0


def cmd_fingerprint_starters(args: argparse.Namespace) -> int:
    """Maintainers: refresh (or ``--check``) ``grader/starters.json`` after editing a starter."""
    current = starter_fingerprints()
    if args.check:
        if load_manifest() != current:
            fix = "python -m grader fingerprint-starters"
            print(red(f"grader/starters.json is out of date: run {fix}"))
            return 1
        print(green("grader/starters.json matches the blank starters."))
        return 0
    STARTERS_MANIFEST.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {STARTERS_MANIFEST} ({len(current)} starters)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m grader", description="Persian RAG Academy grader / تصحیح‌کنندهٔ خودکار"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list lessons")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="grade your code for one lesson")
    p_run.add_argument("lesson", help="lesson number, e.g. 02")
    p_run.add_argument("--path", help="folder with your code (default: the lesson's starter/)")
    p_run.add_argument("-v", "--verbose", action="store_true", help="show every failing case")
    p_run.set_defaults(func=cmd_run)

    p_progress = sub.add_parser(
        "progress", help="table of every lesson: passed / failed / not started"
    )
    p_progress.add_argument(
        "--root",
        help="folder with one subfolder per lesson (<slug> or <NN>); default: each starter/",
    )
    p_progress.set_defaults(func=cmd_progress)

    p_fp = sub.add_parser("fingerprint-starters", help="(maintainers) refresh starters.json")
    p_fp.add_argument("--check", action="store_true", help="fail if starters.json is stale")
    p_fp.set_defaults(func=cmd_fingerprint_starters)

    p_check = sub.add_parser("check-solutions", help="run all lesson tests against parsrag")
    p_check.add_argument("-v", "--verbose", action="store_true")
    p_check.set_defaults(func=cmd_check_solutions)

    p_starters = sub.add_parser("check-starters", help="check that every blank starter fails")
    p_starters.add_argument("-v", "--verbose", action="store_true")
    p_starters.set_defaults(func=cmd_check_starters)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # Windows consoles may not default to UTF-8
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return int(args.func(args))

"""pytest plugin that records per-test results (with hints) as JSON for the grader.

Activated with ``-p grader.plugin``; writes to the file named by
``PARSRAG_GRADER_REPORT``.
"""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Generator
from typing import Any

import pytest

REPORT_ENV = "PARSRAG_GRADER_REPORT"

_results: list[dict[str, Any]] = []
_collection_errors: list[str] = []


def _format_params(params: dict[str, Any], limit: int = 90) -> str:
    parts = []
    for key, value in params.items():
        shown = getattr(value, "name", None) if hasattr(value, "is_file") else None
        parts.append(f"{key}={shown if shown is not None else repr(value)}")
    text = ", ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarize_exception(call: pytest.CallInfo[None]) -> str:
    if call.excinfo is None:
        return ""
    exc = call.excinfo.value
    if isinstance(exc, NotImplementedError):
        return f"not implemented yet / هنوز پیاده‌سازی نشده ({exc})"
    if isinstance(exc, AssertionError):
        lines = call.excinfo.exconly().strip().splitlines()
        return lines[0] if lines else "AssertionError"
    summary = traceback.format_exception_only(type(exc), exc)[-1].strip()
    if isinstance(exc, SyntaxError) and exc.filename:
        summary += f" ({os.path.basename(exc.filename)}, line {exc.lineno})"
    return summary


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, None]:
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        func = getattr(item, "function", None)
        doc = (getattr(func, "__doc__", None) or "").strip()
        title = doc.splitlines()[0] if doc else item.name
        marker = item.get_closest_marker("hint")
        hint = dict(marker.kwargs) if marker else {}
        callspec = getattr(item, "callspec", None)
        _results.append(
            {
                "nodeid": item.nodeid,
                "function": getattr(func, "__name__", item.name),
                "title": title,
                "param": _format_params(callspec.params) if callspec else "",
                "outcome": report.outcome,
                "when": report.when,
                "error": _summarize_exception(call) if report.outcome == "failed" else "",
                "hint": hint,
            }
        )


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed:
        _collection_errors.append(str(report.longreprtext).strip().splitlines()[-1])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    path = os.environ.get(REPORT_ENV)
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"results": _results, "collection_errors": _collection_errors, "exit": int(exitstatus)},
            f,
            ensure_ascii=False,
        )

"""Lesson 10 starter (part 3 of 3) — the regression gate.

Fill in every function marked TODO, then check your work:

    python -m grader run 10

Loading, saving and formatting are written for you; compare_metrics is the
heart of the gate. Reference solution: src/parsrag/eval/gate.py.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Metrics = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class Change:
    """One metric in one question set: baseline → current."""

    group: str
    metric: str
    baseline: float
    current: float | None  # None = missing from the current run

    @property
    def delta(self) -> float | None:
        return None if self.current is None else self.current - self.baseline


def compare_metrics(baseline: Metrics, current: Metrics, tolerance: float = 0.02) -> list[Change]:
    """Every baseline metric that is missing or dropped by more than ``tolerance``.

    Improvements and new metrics never fail the gate. ``tolerance`` is absolute
    (0.02 = two points on a 0–1 scale).
    """
    # TODO: implement
    raise NotImplementedError("TODO: compare_metrics")


def all_changes(baseline: Metrics, current: Metrics) -> list[Change]:
    """Every baseline metric with its current value (for the gate's report)."""
    return [
        Change(group, name, float(base), current.get(group, {}).get(name))
        for group, metrics in baseline.items()
        for name, base in metrics.items()
    ]


def load_baseline(path: str | Path) -> tuple[dict[str, dict[str, float]], float]:
    """``(metrics, tolerance)`` from a baseline file."""
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["metrics"], float(data.get("tolerance", 0.02))


def save_baseline(path: str | Path, metrics: Metrics, tolerance: float = 0.02) -> None:
    payload = {"tolerance": tolerance, "metrics": {g: dict(m) for g, m in metrics.items()}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def format_changes(changes: list[Change], failures: list[Change]) -> str:
    """A Markdown table of baseline vs current, marking failures with ✘."""
    failed = {(c.group, c.metric) for c in failures}
    lines = ["| set | metric | baseline | current | Δ | |", "|---|---|---|---|---|---|"]
    for c in changes:
        cur = "missing" if c.current is None else f"{c.current:.3f}"
        delta = "" if c.delta is None else f"{c.delta:+.3f}"
        mark = "✘" if (c.group, c.metric) in failed else "✔"
        lines.append(f"| {c.group} | {c.metric} | {c.baseline:.3f} | {cur} | {delta} | {mark} |")
    return "\n".join(lines)


__all__ = [
    "Change",
    "Metrics",
    "all_changes",
    "compare_metrics",
    "format_changes",
    "load_baseline",
    "save_baseline",
]

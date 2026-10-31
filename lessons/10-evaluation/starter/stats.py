"""Lesson 10 starter (part 2 of 3) — bootstrap confidence intervals.

Fill in every function marked TODO, then check your work:

    python -m grader run 10

Reference solution: src/parsrag/eval/stats.py (try not to peek!).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate with a ``(1 - alpha)`` confidence interval."""

    mean: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(
    values: Sequence[float], *, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Percentile bootstrap: resample ``values`` with replacement, take the mean each time.

    The interval is the ``alpha/2`` and ``1 - alpha/2`` percentiles of those
    means. Deterministic for a given ``seed``.
    """
    # TODO: implement
    raise NotImplementedError("TODO: bootstrap_ci")


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """CI of ``mean(b) - mean(a)`` when both systems answered the *same* questions.

    Resampling question ids (not each system separately) keeps the pairing, so
    "hard question for both" does not count as noise. If the interval contains
    0, the data do not show a difference.
    """
    # TODO: implement
    raise NotImplementedError("TODO: paired_bootstrap")


__all__ = ["Interval", "bootstrap_ci", "paired_bootstrap"]

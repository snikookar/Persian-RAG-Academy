"""How sure are we? Bootstrap confidence intervals for small eval sets (lesson 10).

With 59 questions, one question moves R@1 by 1.7 points. Before celebrating a
+0.02 improvement, resample the questions and see how much the metric moves
by chance alone.
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
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("need at least one value")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.size, size=(n_resamples, data.size))
    means = data[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(data.mean()), float(low), float(high))


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
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape:
        raise ValueError("a and b must have the same length (one value per question)")
    return bootstrap_ci(vb - va, n_resamples=n_resamples, alpha=alpha, seed=seed)


__all__ = ["Interval", "bootstrap_ci", "paired_bootstrap"]

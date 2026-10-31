"""Lesson 12 starter (part 2 of 2) — per-client rate limiting with a token bucket.

Fill in ``RateLimiter.check``, then: python -m grader run 12
Reference solution: src/parsrag/api/ratelimit.py.

Every client has a bucket of ``rate`` tokens that refills at ``rate / period``
tokens per second; a request takes one token. Bursts up to ``rate`` are
allowed, the long-run average is limited. In memory and per process: with
several workers or machines, put the buckets in Redis (or a gateway) instead.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Decision:
    allowed: bool
    remaining: int
    retry_after: float  # seconds until one token is available (0 if allowed)


class RateLimiter:
    """``RateLimiter(60)`` = 60 requests per minute per client; ``rate=0`` disables it."""

    def __init__(
        self,
        rate: int = 60,
        period: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10_000,
    ) -> None:
        if rate < 0 or period <= 0:
            raise ValueError("need rate >= 0 and period > 0")
        self.rate = rate
        self.period = period
        self.clock = clock
        self.max_clients = max_clients
        self._buckets: dict[str, tuple[float, float]] = {}  # client → (tokens, last time)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def check(self, client: str) -> Decision:
        """Take one token for ``client`` if there is one.

        * disabled (``rate == 0``) → always ``Decision(True, 0, 0.0)``;
        * a new client starts with a full bucket: ``self.rate`` tokens;
        * refill: ``tokens = min(rate, tokens + elapsed * rate / period)``;
        * ≥ 1 token → take one, ``Decision(True, int(tokens left), 0.0)``;
          otherwise ``Decision(False, 0, seconds until one token)``
          = ``(1 - tokens) / (rate / period)``;
        * store ``(tokens, now)`` in ``self._buckets[client]`` (under ``self._lock``)
          and, beyond ``max_clients``, forget the clients seen longest ago.
        """
        # TODO: implement
        raise NotImplementedError("TODO: check")


__all__ = ["Decision", "RateLimiter"]

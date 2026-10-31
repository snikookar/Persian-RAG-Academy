"""Per-client rate limiting with a token bucket (lesson 12).

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
        """Take one token for ``client`` if there is one."""
        if not self.enabled:
            return Decision(True, 0, 0.0)
        now = self.clock()
        refill = self.rate / self.period
        with self._lock:
            tokens, last = self._buckets.get(client, (float(self.rate), now))
            tokens = min(float(self.rate), tokens + (now - last) * refill)
            if tokens >= 1.0:
                tokens -= 1.0
                decision = Decision(True, int(tokens), 0.0)
            else:
                decision = Decision(False, 0, (1.0 - tokens) / refill)
            self._buckets[client] = (tokens, now)
            if len(self._buckets) > self.max_clients:  # forget the stalest clients
                for key, _ in sorted(self._buckets.items(), key=lambda kv: kv[1][1])[
                    : len(self._buckets) - self.max_clients
                ]:
                    del self._buckets[key]
        return decision


__all__ = ["Decision", "RateLimiter"]

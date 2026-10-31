"""API building blocks that need no FastAPI: settings, cache, rate limiter, JSON logs."""

from __future__ import annotations

import json
import logging

import pytest

from parsrag.api import (
    AnswerCache,
    JsonFormatter,
    RateLimiter,
    Settings,
    cache_key,
    normalize_question,
    question_fingerprint,
)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_settings_from_env() -> None:
    s = Settings.from_env(
        {"PARSRAG_RATE_LIMIT": "5", "PARSRAG_LLM_TIMEOUT": "2.5", "PARSRAG_DOCS": "", "OTHER": "x"}
    )
    assert s.rate_limit == 5 and s.llm_timeout == 2.5 and s.docs is None
    assert Settings.from_env({}).cache_size == 256


def test_normalize_question_and_key() -> None:
    a = "هزینهٔ پیک چقدر است؟"
    assert (
        normalize_question(a)
        == normalize_question("  هزينه پيک   چقدر است ")
        == "هزینه پیک چقدر است"
    )
    assert cache_key(a, 1) == cache_key("هزينه پيک چقدر است!", 1)
    assert cache_key(a, 1) != cache_key(a, 2)  # a new index version invalidates
    assert cache_key(a) != cache_key("هزینهٔ ارسال چقدر است؟")


def test_cache_lru_ttl_and_stats() -> None:
    clock = Clock()
    cache = AnswerCache(maxsize=2, ttl=10, clock=clock)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # a is now most recent
    cache.set("c", 3)  # evicts b
    assert cache.get("b") is None and cache.get("c") == 3
    clock.t += 11
    assert cache.get("a") is None and len(cache) == 1
    assert cache.stats()["hits"] == 2 and cache.stats()["misses"] == 2
    cache.clear()
    assert len(cache) == 0
    off = AnswerCache(maxsize=0)
    off.set("a", 1)
    assert off.get("a") is None
    with pytest.raises(ValueError):
        AnswerCache(ttl=0)


def test_rate_limiter_token_bucket() -> None:
    clock = Clock()
    limiter = RateLimiter(3, 60, clock=clock)
    assert [limiter.check("ip").allowed for _ in range(4)] == [True, True, True, False]
    blocked = limiter.check("ip")
    assert not blocked.allowed and blocked.retry_after == pytest.approx(20.0)
    assert limiter.check("other").allowed  # buckets are per client
    clock.t += 20
    assert limiter.check("ip").allowed and not limiter.check("ip").allowed
    assert RateLimiter(0).check("x").allowed


def test_rate_limiter_forgets_stale_clients() -> None:
    clock = Clock()
    limiter = RateLimiter(1, 60, clock=clock, max_clients=2)
    for i in range(3):
        clock.t += 1
        limiter.check(f"c{i}")
    assert set(limiter._buckets) == {"c1", "c2"}


def test_json_formatter() -> None:
    record = logging.LogRecord("parsrag.api", logging.INFO, "", 0, "ask", None, None)
    record.fields = {"cached": True, "total_ms": 1.5, "q": "سلام"}
    data = json.loads(JsonFormatter().format(record))
    assert data["event"] == "ask" and data["cached"] is True and data["q"] == "سلام"
    assert data["level"] == "INFO" and data["ts"].endswith("+00:00")
    assert len(question_fingerprint("سلام")) == 12

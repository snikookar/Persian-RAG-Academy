"""Lesson 12 — production API. Run with: python -m grader run 12"""

from __future__ import annotations

from types import ModuleType

import pytest

from parsrag import PersianRAG

hint = pytest.mark.hint

PIK = "هزینهٔ پیک در تهران چقدر است؟"
EXTRA = "API checks need: pip install -e '.[api]'"


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture(scope="module")
def rag() -> PersianRAG:
    return PersianRAG.course_default()


# -------------------------------------------------------------------- cache


@hint(
    fa="normalize → lower → هر نویسهٔ _PUNCT به فاصله → split و join با یک فاصله.",
    en="normalize → lower → every _PUNCT character to a space → split and join with single spaces.",
)
def test_normalize_question(cm: ModuleType) -> None:
    """spelling variants of a question normalise to the same text"""
    assert cm.normalize_question("هزینهٔ پیک چقدر است؟") == "هزینه پیک چقدر است"
    assert cm.normalize_question("  هزينه پيک   چقدر است!! ") == "هزینه پیک چقدر است"
    assert cm.normalize_question("ساعت ۹ باز است؟") == "ساعت 9 باز است"
    assert cm.normalize_question("می‌شود؟") == "می‌شود"  # ZWNJ is part of the word


@hint(
    fa="sha256 روی «نسخه + \\x00 + پرسش نرمال‌شده» و hexdigest.",
    en="sha256 over 'version + \\x00 + normalised question', as hexdigest.",
)
def test_cache_key(cm: ModuleType) -> None:
    """the key ignores spelling but not the index version"""
    a = cm.cache_key(PIK, 1)
    assert len(a) == 64 and int(a, 16) >= 0
    assert a == cm.cache_key("  هزينه پيک در تهران چقدر است", 1)
    assert a != cm.cache_key(PIK, 2)
    assert a != cm.cache_key("هزینهٔ ارسال در تهران چقدر است؟", 1)


@hint(
    fa="get: نبود یا منقضی → None و misses++؛ وگرنه move_to_end و hits++.",
    en="get: missing or expired → None and misses++; else move_to_end and hits++.",
)
def test_cache_get_set_and_ttl(cm: ModuleType) -> None:
    """values come back until they expire, and hits/misses are counted"""
    clock = Clock()
    cache = cm.AnswerCache(maxsize=10, ttl=60, clock=clock)
    assert cache.get("k") is None
    cache.set("k", {"answer": "۴۵ هزار تومان"})
    assert cache.get("k") == {"answer": "۴۵ هزار تومان"}
    clock.t += 59
    assert cache.get("k") is not None
    clock.t += 1
    assert cache.get("k") is None and len(cache) == 0  # expired entries are removed
    assert (cache.hits, cache.misses) == (2, 2)


@hint(
    fa="بعد از set اگر بیش از maxsize شد، از اول OrderedDict حذف کنید (popitem(last=False)).",
    en="After set, beyond maxsize, evict from the front of the OrderedDict (popitem(last=False)).",
)
def test_cache_lru_eviction(cm: ModuleType) -> None:
    """the least recently *used* entry is evicted first"""
    cache = cm.AnswerCache(maxsize=2, ttl=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # a is now more recent than b
    cache.set("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
    cache.set("a", 10)  # overwriting refreshes, does not grow
    assert len(cache) == 2 and cache.get("a") == 10
    off = cm.AnswerCache(maxsize=0)
    off.set("x", 1)
    assert off.get("x") is None and len(off) == 0


# --------------------------------------------------------------- rate limit


@hint(
    fa="مشتری تازه با سطل پر (rate توکن) شروع می‌کند؛ هر درخواست یک توکن برمی‌دارد.",
    en="A new client starts with a full bucket (rate tokens); each request takes one.",
)
def test_bucket_allows_a_burst_then_blocks(rl: ModuleType) -> None:
    """a burst up to the rate is allowed, then requests are refused"""
    limiter = rl.RateLimiter(3, 60, clock=Clock())
    decisions = [limiter.check("1.2.3.4") for _ in range(4)]
    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert [d.remaining for d in decisions[:3]] == [2, 1, 0]


@hint(
    fa="retry_after = (1 − tokens) / (rate / period)؛ با گذشت زمان توکن‌ها پر می‌شوند (حداکثر rate).",
    en="retry_after = (1 − tokens) / (rate / period); tokens refill over time (at most rate).",
)
def test_bucket_refills_over_time(rl: ModuleType) -> None:
    """tokens come back at rate/period per second, and Retry-After says when"""
    clock = Clock()
    limiter = rl.RateLimiter(2, 60, clock=clock)
    limiter.check("a")
    limiter.check("a")
    blocked = limiter.check("a")
    assert not blocked.allowed and blocked.retry_after == pytest.approx(30.0)
    clock.t += 30
    assert limiter.check("a").allowed and not limiter.check("a").allowed
    clock.t += 3600  # a long pause refills only up to the bucket size
    assert [limiter.check("a").allowed for _ in range(3)] == [True, True, False]


@hint(
    fa="هر مشتری سطل خودش را دارد؛ rate=0 یعنی خاموش؛ بیش از max_clients، قدیمی‌ترین‌ها را فراموش کنید.",
    en="One bucket per client; rate=0 means off; beyond max_clients forget the stalest clients.",
)
def test_buckets_are_per_client(rl: ModuleType) -> None:
    """clients do not share buckets; rate 0 disables limiting; memory is bounded"""
    clock = Clock()
    limiter = rl.RateLimiter(1, 60, clock=clock, max_clients=2)
    assert limiter.check("a").allowed and not limiter.check("a").allowed
    assert limiter.check("b").allowed
    clock.t += 1
    limiter.check("c")
    assert set(limiter._buckets) == {"b", "c"}
    off = rl.RateLimiter(0)
    assert all(off.check("x").allowed for _ in range(100))


# ------------------------------------------------------------ in the real app


@hint(
    fa="کش شما داخل اپ واقعی است: پرسش دوم (با املای دیگر) باید cached=true برگردد.",
    en="Your cache runs inside the real app: the second, differently spelled question must be cached.",
)
def test_your_cache_in_the_api(cm: ModuleType, rag: PersianRAG) -> None:
    """with your cache, a repeated question is answered from the cache"""
    pytest.importorskip("fastapi", reason=EXTRA)
    from fastapi.testclient import TestClient

    from parsrag.api import Settings, create_app

    cache = cm.AnswerCache(maxsize=32, ttl=600)
    client = TestClient(create_app(rag, Settings(rate_limit=0), cache=cache))
    first = client.post("/ask", json={"question": PIK}).json()
    second = client.post("/ask", json={"question": "هزينه پيک در تهران چقدر است"}).json()
    assert not first["cached"] and second["cached"] and second["text"] == first["text"]
    assert cache.hits == 1 and len(cache) == 1


@hint(
    fa="محدودکنندهٔ شما داخل اپ: درخواست سوم با rate=2 باید 429 با Retry-After بگیرد.",
    en="Your limiter inside the app: with rate=2 the third request must get 429 with Retry-After.",
)
def test_your_limiter_in_the_api(rl: ModuleType, rag: PersianRAG) -> None:
    """with your limiter, the API answers 429 + Retry-After, per API key"""
    pytest.importorskip("fastapi", reason=EXTRA)
    from fastapi.testclient import TestClient

    from parsrag.api import Settings, create_app

    client = TestClient(create_app(rag, Settings(rate_limit=0), limiter=rl.RateLimiter(2, 60)))
    key = {"X-API-Key": "learner"}
    codes = [client.post("/ask", json={"question": PIK}, headers=key).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/ask", json={"question": PIK}, headers=key)
    assert r.json()["error"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1
    other = client.post("/ask", json={"question": PIK}, headers={"X-API-Key": "other"})
    assert other.status_code == 200


@hint(
    fa="این بررسی کد شما را عوض نمی‌کند؛ اگر رد شد، extra api را نصب کنید.",
    en="This check does not depend on your code; if it fails, install the api extra.",
)
def test_errors_are_clear(rag: PersianRAG) -> None:
    """empty index → 409, bad request → 422, streaming ends with done"""
    pytest.importorskip("fastapi", reason=EXTRA)
    from fastapi.testclient import TestClient

    from parsrag.api import Settings, create_app

    empty = TestClient(create_app(PersianRAG(), Settings(rate_limit=0)))
    r = empty.post("/ask", json={"question": PIK})
    assert r.status_code == 409 and r.json()["error"] == "empty_index"
    assert empty.post("/ask", json={}).status_code == 422
    full = TestClient(create_app(rag, Settings(rate_limit=0)))
    body = full.post("/ask/stream", json={"question": PIK}).text
    assert body.startswith("event: sources") and "event: token" in body
    assert body.rstrip().splitlines()[-2] == "event: done"

"""An answer cache keyed on the normalised question (lesson 12).

«هزینهٔ پیک چقدر است؟», «هزينه پيک  چقدر است» and «هزینهٔ پیک چقدر است»
are the same question: :func:`cache_key` normalises (lesson 02), lowercases,
drops punctuation and collapses spaces before hashing. The key also includes
the index version, so adding documents never serves a stale answer.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from parsrag.normalize import normalize

_PUNCT = re.compile(r"[^\w\s‌]")


def normalize_question(question: str) -> str:
    """The form of a question used for caching: normalised, no punctuation, single spaces."""
    text = _PUNCT.sub(" ", normalize(question).lower())
    return " ".join(text.split())


def cache_key(question: str, index_version: int | str = 0) -> str:
    """``sha256(index_version + normalised question)`` as hex."""
    raw = f"{index_version}\x00{normalize_question(question)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AnswerCache:
    """A thread-safe LRU cache with a time-to-live.

    ``maxsize=0`` disables caching. ``clock`` is injectable for tests.
    """

    def __init__(
        self,
        maxsize: int = 256,
        ttl: float = 3600.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if maxsize < 0 or ttl <= 0:
            raise ValueError("need maxsize >= 0 and ttl > 0")
        self.maxsize = maxsize
        self.ttl = ttl
        self.clock = clock
        self.hits = 0
        self.misses = 0
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] <= self.clock():
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: str, value: Any) -> None:
        if self.maxsize == 0:
            return
        with self._lock:
            self._data[key] = (self.clock() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


__all__ = ["AnswerCache", "cache_key", "normalize_question"]

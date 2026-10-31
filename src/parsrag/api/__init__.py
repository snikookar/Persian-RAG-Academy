"""Production API around :class:`parsrag.PersianRAG` (lesson 12).

The building blocks — :mod:`~parsrag.api.settings`, :mod:`~parsrag.api.cache`,
:mod:`~parsrag.api.ratelimit` and :mod:`~parsrag.api.logs` — are plain Python
and work on the core install. :func:`create_app` needs FastAPI
(``pip install 'parsrag[api]'``) and is imported on first use.
"""

from __future__ import annotations

from typing import Any

from parsrag.api.cache import AnswerCache, cache_key, normalize_question
from parsrag.api.logs import JsonFormatter, get_logger, log_event, question_fingerprint
from parsrag.api.ratelimit import Decision, RateLimiter
from parsrag.api.settings import Settings


def __getattr__(name: str) -> Any:
    if name in ("create_app", "build_rag", "RAGService"):
        try:
            from parsrag.api import app as app_module
        except ImportError as exc:
            raise ImportError(
                "The API needs FastAPI. Install it with: pip install 'parsrag[api]'"
            ) from exc
        return getattr(app_module, name)
    raise AttributeError(f"module 'parsrag.api' has no attribute {name!r}")


__all__ = [
    "AnswerCache",
    "Decision",
    "JsonFormatter",
    "RAGService",
    "RateLimiter",
    "Settings",
    "build_rag",
    "cache_key",
    "create_app",
    "get_logger",
    "log_event",
    "normalize_question",
    "question_fingerprint",
]

"""Structured JSON logging (lesson 12): one JSON object per line, easy to search and chart.

>>> log = get_logger()
>>> log_event(log, "ask", cached=False, retrieve_ms=3.1, generate_ms=812.4)  # doctest: +SKIP
{"ts": "2026-10-31T12:00:00.000Z", "level": "INFO", "event": "ask", "cached": false, ...}

Questions are logged as a short hash and a length, not as text: logs are
often kept longer and seen by more people than the data they describe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

LOGGER = "parsrag.api"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        data.update(getattr(record, "fields", {}))
        if record.exc_info:
            data["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(data, ensure_ascii=False, default=str)


def get_logger(level: str = "INFO") -> logging.Logger:
    """The API logger with a JSON handler on stderr (added once)."""
    logger = logging.getLogger(LOGGER)
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level.upper())
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})


def question_fingerprint(question: str) -> str:
    """A short, stable id for a question that does not reveal its text."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


__all__ = ["JsonFormatter", "get_logger", "log_event", "question_fingerprint"]

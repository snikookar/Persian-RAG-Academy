"""Loading question/answer evaluation sets (JSONL)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class QAItem(BaseModel):
    """One evaluation question.

    ``doc_id`` is the id of the document that contains the answer, or ``None``
    when the answer is *not* in the corpus (the system should say "I don't know").
    """

    id: str
    question: str
    answer: str
    doc_id: str | None = None

    @property
    def answerable(self) -> bool:
        return self.doc_id is not None


def load_qa(path: str | Path) -> list[QAItem]:
    """Read a JSONL file with one :class:`QAItem` per line (blank lines ignored)."""
    items: list[QAItem] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(QAItem.model_validate(json.loads(line)))
    return items

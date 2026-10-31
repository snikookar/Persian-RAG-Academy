"""Loading question/answer evaluation sets (JSONL)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel

from parsrag.types import Document


class QAItem(BaseModel):
    """One evaluation question.

    ``doc_id`` is the id of the document that contains the answer, or ``None``
    when the answer is *not* in the corpus (the system should say "I don't know").
    ``evidence`` is a short span copied verbatim from that document which
    contains the answer. It allows *passage-level* evaluation: did the
    retrieved chunk actually contain the answer, not just the right document?
    ``source_id`` links a paraphrased question (``qa_paraphrase.jsonl``) to the
    original question it rewords.
    """

    id: str
    question: str
    answer: str
    doc_id: str | None = None
    evidence: str | None = None
    source_id: str | None = None

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


def validate_qa(
    items: Iterable[QAItem], documents: Mapping[str, Document] | Iterable[Document]
) -> list[str]:
    r"""Problems in a hand-written QA set, checked against the indexed documents (lesson 13).

    A typo in an evidence span silently turns a good answer into a "miss"
    in every passage-level metric, so check before trusting any number:
    duplicate ids, empty questions, unknown ``doc_id``\s, answerable
    questions without evidence, and evidence that is not in its document
    (compared after the same normalisation the metrics use).
    """
    from parsrag.eval.retrieval import match_key  # local: retrieval imports this module

    docs = dict(documents) if isinstance(documents, Mapping) else {d.id: d for d in documents}
    keys = {doc_id: match_key(d.text) for doc_id, d in docs.items()}
    problems: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            problems.append(f"{item.id}: duplicate id")
        seen.add(item.id)
        if not item.question.strip():
            problems.append(f"{item.id}: empty question")
        if item.doc_id is None:
            if item.evidence:
                problems.append(f"{item.id}: unanswerable (no doc_id) but has evidence")
            continue
        if item.doc_id not in docs:
            problems.append(f"{item.id}: unknown doc_id {item.doc_id!r}")
        elif not item.evidence:
            problems.append(f"{item.id}: answerable but has no evidence span")
        elif match_key(item.evidence) not in keys[item.doc_id]:
            problems.append(f"{item.id}: evidence not found in {item.doc_id!r}: {item.evidence!r}")
    return problems

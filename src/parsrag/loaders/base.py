"""Shared helpers for loaders: sections, pages and positions inside a document.

Every loader returns a :class:`~parsrag.types.Document` whose text is a light
Markdown rendering of the source (headings become ``#`` lines) and whose
metadata records where things came from::

    {
        "source": "data/fixtures/documents/kahroba-catalog.pdf",
        "format": "pdf",
        "sections": [{"title": ..., "level": 1, "start": 0, "end": 120}, ...],
        "pages": [{"page": 1, "start": 0, "end": 410}, ...],   # PDF only
    }

``start``/``end`` are character offsets into ``Document.text``. Lesson 04's
chunkers use :func:`locate` to tell every chunk which page and section it
came from, so answers can cite "page 2, section «گارانتی»".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from parsrag.types import Document

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def markdown_sections(text: str) -> list[dict[str, Any]]:
    """Find Markdown headings and the character span each section covers.

    A section runs from its heading to the next heading of the same or a
    higher level (so a ``##`` section contains its ``###`` subsections).
    """
    heads = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in _HEADING_RE.finditer(text)]
    sections = []
    for i, (start, level, title) in enumerate(heads):
        end = next((s for s, lvl, _ in heads[i + 1 :] if lvl <= level), len(text))
        sections.append({"title": title, "level": level, "start": start, "end": end})
    return sections


def locate(doc: Document, offset: int) -> dict[str, Any]:
    """Return the page and section path that contain character ``offset`` of ``doc``.

    >>> locate(doc, 250)
    {'page': 1, 'section': 'هزینهٔ نصب', 'section_path': ['راهنمای نصب', 'هزینهٔ نصب']}
    """
    page = None
    for p in doc.metadata.get("pages", []):
        if p["start"] <= offset < max(p["end"], p["start"] + 1):
            page = p["page"]
            break
    path = [
        s["title"]
        for s in sorted(doc.metadata.get("sections", []), key=lambda s: (s["start"], s["level"]))
        if s["start"] <= offset < s["end"]
    ]
    return {"page": page, "section": path[-1] if path else None, "section_path": path}


def make_document(
    path: Path, text: str, fmt: str, *, title: str = "", doc_id: str | None = None, **extra: Any
) -> Document:
    """Build a Document with the standard metadata (source, format, sections)."""
    text = text.strip()
    sections = markdown_sections(text)
    if not title:
        title = next((s["title"] for s in sections if s["level"] == 1), "")
    metadata: dict[str, Any] = {"source": str(path), "format": fmt, "sections": sections, **extra}
    return Document(id=doc_id or path.stem, title=title, text=text, metadata=metadata)


def missing_extra(package: str, loader: str) -> ImportError:
    return ImportError(
        f"{loader} needs the '{package}' package. Install it with: pip install 'parsrag[loaders]'"
    )

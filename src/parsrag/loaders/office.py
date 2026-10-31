"""DOCX loader (needs ``python-docx``)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from parsrag.loaders.base import make_document, missing_extra
from parsrag.loaders.bidi import strip_bidi_controls
from parsrag.types import Document

_HEADING_STYLE = re.compile(r"heading\s*(\d)", re.I)


def heading_level(style_name: str) -> int | None:
    """Markdown level for a Word style: Title → 1, Heading 1 → 2, ... (max 6); else None."""
    if style_name.lower() == "title":
        return 1
    m = _HEADING_STYLE.fullmatch(style_name.strip())
    return min(int(m.group(1)) + 1, 6) if m else None


def load_docx(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Load a Word document in reading order: paragraphs, headings, lists and tables.

    The ``Title`` style becomes ``#``, ``Heading 1`` ``##``, ``Heading 2``
    ``###`` ...; list paragraphs get ``- ``; table rows become ``a | b``.
    Word often inserts invisible RLM/LRM marks around Latin text inside
    Persian paragraphs; they are removed.
    """
    try:
        import docx
        from docx.table import Table
    except ImportError as exc:
        raise missing_extra("python-docx", "load_docx") from exc

    path = Path(path)
    document: Any = docx.Document(str(path))
    blocks: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            for row in item.rows:
                cells = [strip_bidi_controls(c.text).strip() for c in row.cells]
                if any(cells):
                    blocks.append(" | ".join(cells))
            continue
        text = strip_bidi_controls(item.text).strip()
        if not text:
            continue
        style = item.style.name if item.style is not None else ""
        level = heading_level(style)
        if level:
            blocks.append(f"{'#' * level} {text}")
        elif style.lower().startswith("list"):
            blocks.append(f"- {text}")
        else:
            blocks.append(text)
    title = (document.core_properties.title or "").strip()
    return make_document(path, "\n\n".join(blocks), "docx", title=title, doc_id=doc_id)

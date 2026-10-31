"""PDF loader with Persian repairs (needs ``pypdf``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from parsrag.loaders.base import make_document, missing_extra
from parsrag.loaders.bidi import repair_pdf_text
from parsrag.types import Document

PAGE_SEPARATOR = "\n\n"


def load_pdf(
    path: str | Path,
    *,
    doc_id: str | None = None,
    repair: bool = True,
    fix_visual: bool | Literal["auto"] = "auto",
) -> Document:
    """Load a PDF, one page after another, recording page offsets in metadata.

    With ``repair=True`` (default) every page goes through
    :func:`~parsrag.loaders.bidi.repair_pdf_text`: visual order is detected
    and fixed per page, lost ZWNJs are recovered from glyph shapes, and
    presentation forms are mapped back to letters. ``repair=False`` returns
    exactly what ``pypdf`` extracted — useful to *see* the problems.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise missing_extra("pypdf", "load_pdf") from exc

    path = Path(path)
    reader: Any = PdfReader(str(path))
    parts: list[str] = []
    pages: list[dict[str, Any]] = []
    offset = 0
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        visual = False
        if repair:
            text, visual = repair_pdf_text(text, fix_visual=fix_visual)
        text = text.strip()
        if parts:
            offset += len(PAGE_SEPARATOR)
        pages.append(
            {"page": number, "start": offset, "end": offset + len(text), "visual_order": visual}
        )
        parts.append(text)
        offset += len(text)
    meta_title = ""
    if reader.metadata is not None and reader.metadata.title:
        meta_title = str(reader.metadata.title).strip()
    return make_document(
        path, PAGE_SEPARATOR.join(parts), "pdf", title=meta_title, doc_id=doc_id, pages=pages
    )

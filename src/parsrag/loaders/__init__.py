"""Loading Persian documents from files (lesson 03).

>>> from parsrag.loaders import load_document, load_directory
>>> doc = load_document("data/fixtures/documents/sepidar-newsletter.pdf")
>>> doc.metadata["pages"][0]["visual_order"]   # it was stored backwards — and fixed
True

TXT and Markdown need nothing extra; HTML, DOCX and PDF need
``pip install 'parsrag[loaders]'`` (beautifulsoup4, python-docx, pypdf),
imported lazily so ``import parsrag.loaders`` always works.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from parsrag.loaders.base import locate, make_document, markdown_sections
from parsrag.loaders.bidi import (
    looks_visual_order,
    recover_zwnj,
    repair_pdf_text,
    shape,
    strip_bidi_controls,
    to_visual,
    visual_order_votes,
    visual_to_logical,
)
from parsrag.loaders.office import heading_level, load_docx
from parsrag.loaders.pdf import load_pdf
from parsrag.loaders.text import load_markdown, load_text, read_text_file
from parsrag.loaders.web import html_to_markdown, load_html, remove_boilerplate
from parsrag.types import Document

LOADERS: dict[str, Callable[..., Document]] = {
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".docx": load_docx,
    ".pdf": load_pdf,
}


def load_document(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Load one file with the loader that matches its extension."""
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        supported = ", ".join(sorted(LOADERS))
        raise ValueError(f"unsupported file type {path.suffix!r} (supported: {supported})")
    return loader(path, doc_id=doc_id)


def load_directory(
    directory: str | Path, pattern: str = "*", *, recursive: bool = False
) -> list[Document]:
    """Load every supported file in ``directory`` (sorted by name).

    Unsupported files are skipped. Ids are file stems; if two files share a
    stem (``guide.pdf`` and ``guide.docx``) both get their extension appended.
    """
    root = Path(directory)
    paths = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    paths = [p for p in paths if p.is_file() and p.suffix.lower() in LOADERS]
    stems = [p.stem for p in paths]
    return [load_document(p, doc_id=p.name if stems.count(p.stem) > 1 else p.stem) for p in paths]


__all__ = [
    "LOADERS",
    "heading_level",
    "html_to_markdown",
    "load_directory",
    "load_document",
    "load_docx",
    "load_html",
    "load_markdown",
    "load_pdf",
    "load_text",
    "locate",
    "looks_visual_order",
    "make_document",
    "markdown_sections",
    "read_text_file",
    "recover_zwnj",
    "remove_boilerplate",
    "repair_pdf_text",
    "shape",
    "strip_bidi_controls",
    "to_visual",
    "visual_order_votes",
    "visual_to_logical",
]

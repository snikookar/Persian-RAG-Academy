"""Plain-text and Markdown loaders (no extra dependencies)."""

from __future__ import annotations

from pathlib import Path

from parsrag.loaders.base import make_document
from parsrag.loaders.bidi import strip_bidi_controls
from parsrag.types import Document

# Tried in order. Windows-1256 is the legacy Windows code page for Arabic
# script; many old Persian .txt files use it. It must come last because it
# can decode *any* byte sequence (so it never fails, even on wrong guesses).
ENCODINGS = ("utf-8", "cp1256")


def read_text_file(path: str | Path) -> tuple[str, str]:
    """Read a text file, guessing its encoding. Returns ``(text, encoding)``.

    BOMs are honoured (UTF-8 and UTF-16), then strict UTF-8 is tried, then
    Windows-1256. Line endings are normalised to ``\\n``.
    """
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        text, encoding = raw[3:].decode("utf-8"), "utf-8-sig"
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text, encoding = raw.decode("utf-16"), "utf-16"
    else:
        for encoding in ENCODINGS:
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - cp1256 decodes everything
            raise UnicodeDecodeError("unknown", raw, 0, 1, "no encoding matched")
    return text.replace("\r\n", "\n").replace("\r", "\n"), encoding


def load_text(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Load a ``.txt`` file. The first non-empty line becomes the title."""
    path = Path(path)
    text, encoding = read_text_file(path)
    text = strip_bidi_controls(text)
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return make_document(path, text, "txt", title=first, doc_id=doc_id, encoding=encoding)


def load_markdown(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Load a Markdown file; headings are kept and recorded as sections."""
    path = Path(path)
    text, encoding = read_text_file(path)
    return make_document(
        path, strip_bidi_controls(text), "markdown", doc_id=doc_id, encoding=encoding
    )

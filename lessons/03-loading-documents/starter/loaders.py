"""Lesson 03 starter — loading Persian documents.

Fill in every function marked TODO, then check your work:

    python -m grader run 03

Glyph tables, the toy shaper and the file-format plumbing are written for you;
your job is the Persian-specific repairs and a few structural pieces.
Reference solution: src/parsrag/loaders/ (try not to peek until you have tried!).

HTML, DOCX and PDF checks need the optional extra:  pip install -e ".[loaders]"
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

# Helpers you may use (they are not part of the exercise):
#   GLYPH_INFO[glyph] -> (base_letters, form)   form: "isolated" | "final" | "initial" | "medial"
#   joins_both_sides(letter)  True for dual-joining letters (ب، س، ک، ی، ...)
#   joins_right(letter)       True for letters that connect to the previous one (also ا، د، ر، و)
#   is_mark(ch)               True for combining marks (they are transparent for joining)
from parsrag.loaders.bidi import GLYPH_INFO, is_mark, joins_both_sides, joins_right, shape
from parsrag.normalize import ZWNJ, fix_presentation_forms
from parsrag.types import Document

BIDI_CONTROLS = "‎‏؜‪‫‬‭‮⁦⁧⁨⁩"
ENCODINGS = ("utf-8", "cp1256")


# =============================================================== 1. plain text


def strip_bidi_controls(text: str) -> str:
    """Remove the invisible bidi formatting characters listed in BIDI_CONTROLS.

    ZWNJ (U+200C) is NOT a bidi control: keep it!
    """
    # TODO: implement (re.sub with a character class, or str.translate)
    raise NotImplementedError("TODO: strip_bidi_controls")


def read_text_file(path: str | Path) -> tuple[str, str]:
    """Read a text file and guess its encoding. Return (text, encoding_name).

    1. UTF-8 BOM (b"\\xef\\xbb\\xbf")   -> decode the rest as UTF-8, name "utf-8-sig"
    2. UTF-16 BOM (b"\\xff\\xfe" or b"\\xfe\\xff") -> raw.decode("utf-16"), name "utf-16"
    3. otherwise try each name in ENCODINGS in order; the first that decodes wins
       (catch UnicodeDecodeError). cp1256 decodes anything, so it must be last.
    Finally turn "\\r\\n" and "\\r" into "\\n".
    """
    # TODO: implement (Path(path).read_bytes())
    raise NotImplementedError("TODO: read_text_file")


# ============================================================ 2. structure

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def markdown_sections(text: str) -> list[dict[str, Any]]:
    """Return one dict per Markdown heading: {"title", "level", "start", "end"}.

    start = offset of the heading line; end = offset of the next heading whose
    level is <= this one (so a "##" section contains its "###" subsections),
    or len(text). Use _HEADING_RE.finditer: group(1) is the #s, group(2) the title.
    """
    # TODO: implement
    raise NotImplementedError("TODO: markdown_sections")


def locate(doc: Document, offset: int) -> dict[str, Any]:
    """Page and section path containing character ``offset`` (already done for you)."""
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
    """Build a Document with metadata {source, format, sections, ...} (already done)."""
    text = text.strip()
    sections = markdown_sections(text)
    if not title:
        title = next((s["title"] for s in sections if s["level"] == 1), "")
    metadata = {"source": str(path), "format": fmt, "sections": sections, **extra}
    return Document(id=doc_id or path.stem, title=title, text=text, metadata=metadata)


# ====================================================== 3. visual order (RTL)

_LATIN, _DIGIT = "A-Za-z", "0-9۰-۹٠-٩"
# Latin words (several, separated by spaces) and numbers stay left-to-right.
_LTR_RUN_RE = re.compile(
    rf"[{_LATIN}][{_LATIN}{_DIGIT}_\-.@/]*(?: +[{_LATIN}{_DIGIT}][{_LATIN}{_DIGIT}_\-.@/]*)*"
    rf"|[{_DIGIT}](?:[{_DIGIT}]|[.,:/٫٬%\-](?=[{_DIGIT}]))*"
)
_MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")


def visual_to_logical(line: str) -> str:
    """Turn one visual-order line back into logical order.

    Split the line into tokens: every match of _LTR_RUN_RE is ONE token, every
    other character is its own token. Reverse the list of tokens. Single-character
    tokens are mirrored with .translate(_MIRROR); LTR runs stay as they are.

        "تسا 45 تمیق"      -> "قیمت 45 است"
        "تسا PDF file هخسن" -> "نسخه PDF file است"
    """
    # TODO: implement
    raise NotImplementedError("TODO: visual_to_logical")


def to_visual(text: str) -> str:
    """Simulate visual order (visual_to_logical is its own inverse). Done for you."""
    return "\n".join(visual_to_logical(line) for line in text.split("\n"))


_FORWARD_WORDS = frozenset(
    {"از", "به", "این", "که", "را", "با", "است", "برای", "آن", "های", "شده", "باید", "یک"}
)
_BACKWARD_WORDS = frozenset(w[::-1] for w in _FORWARD_WORDS)
_ARABIC_WORD_RE = re.compile(r"[؀-ۿﭐ-﷿ﹰ-﻿]+")


def visual_order_votes(text: str) -> tuple[int, int]:
    """Count evidence (logical_votes, visual_votes).

    1. Glyph shapes — for every match of _ARABIC_WORD_RE in ``text``, ignore
       marks (is_mark), look up the first and last glyph in GLYPH_INFO:
         first glyph "initial"            -> +1 logical
         first glyph "final" or "medial"  -> +1 visual
         last glyph  "final"              -> +1 logical
         last glyph  "initial" or "medial"-> +1 visual
       (glyphs missing from GLYPH_INFO, or "isolated", give no vote)
    2. Common words — for every match of _ARABIC_WORD_RE in
       fix_presentation_forms(text): in _FORWARD_WORDS -> +1 logical,
       in _BACKWARD_WORDS -> +1 visual.
    """
    # TODO: implement
    raise NotImplementedError("TODO: visual_order_votes")


def looks_visual_order(text: str, min_votes: int = 3) -> bool:
    """True when visual evidence wins and there is enough of it. Done for you."""
    logical, visual = visual_order_votes(text)
    return visual > logical and visual >= min_votes


# ===================================================================== 4. ZWNJ


def recover_zwnj(text: str) -> str:
    """Re-insert lost ZWNJs using glyph shapes (text must still contain presentation forms).

    Insert ZWNJ before character j when:
      * the character i before it (skipping marks between i and j) is in GLYPH_INFO,
        its base is ONE letter that joins_both_sides, and its form is "final" or "isolated";
      * character j is in GLYPH_INFO with form "initial" or "isolated", and
        joins_right(first letter of its base).
    Example: shape("می‌شود") == "ﻣﯽﺷﻮﺩ" -> "ﻣﯽ‌ﺷﻮﺩ"
    """
    # TODO: implement
    raise NotImplementedError("TODO: recover_zwnj")


def repair_pdf_text(
    text: str, *, fix_visual: bool | Literal["auto"] = "auto", zwnj: bool = True
) -> tuple[str, bool]:
    """Repair one PDF page. Return (text, visual_order_was_fixed).

    Order: strip_bidi_controls -> (if fix_visual is True, or "auto" and
    looks_visual_order(text)) visual_to_logical on every line -> recover_zwnj
    (if zwnj) -> fix_presentation_forms.
    """
    # TODO: implement
    raise NotImplementedError("TODO: repair_pdf_text")


# ================================================================= 5. loaders


def _missing(package: str) -> ImportError:
    return ImportError(f"needs '{package}': pip install 'parsrag[loaders]'")


def load_text(path: str | Path, *, doc_id: str | None = None) -> Document:
    path = Path(path)
    text, encoding = read_text_file(path)
    text = strip_bidi_controls(text)
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return make_document(path, text, "txt", title=first, doc_id=doc_id, encoding=encoding)


def load_markdown(path: str | Path, *, doc_id: str | None = None) -> Document:
    path = Path(path)
    text, encoding = read_text_file(path)
    return make_document(
        path, strip_bidi_controls(text), "markdown", doc_id=doc_id, encoding=encoding
    )


BOILERPLATE_TAGS = ("nav", "header", "footer", "aside")
BOILERPLATE_ROLES = ("navigation", "banner", "contentinfo", "complementary", "search")
BOILERPLATE_HINT = re.compile(
    r"cookie|consent|breadcrumb|banner|share|social|newsletter|sidebar|menu|advert|popup", re.I
)
DROP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "form", "button")


def remove_boilerplate(soup: Any) -> None:
    """Delete page furniture from a BeautifulSoup tree, in place.

    Call .decompose() on:
      * every tag in DROP_TAGS and BOILERPLATE_TAGS       (soup.find_all(names))
      * every tag whose role attribute is in BOILERPLATE_ROLES
                                                          (soup.find_all(attrs={"role": [...]}))
      * every tag whose class list or id matches BOILERPLATE_HINT
        (skip tags that were already removed: `tag.decomposed` is True)
    """
    # TODO: implement
    raise NotImplementedError("TODO: remove_boilerplate")


def html_to_markdown(html: str) -> tuple[str, str]:
    """HTML -> (markdown_text, <title>). Done for you, except remove_boilerplate."""
    try:
        from bs4 import BeautifulSoup, NavigableString, Tag
    except ImportError as exc:
        raise _missing("beautifulsoup4") from exc

    def clean(s: str) -> str:
        return re.sub(r"\s+", " ", strip_bidi_controls(s)).strip()

    soup = BeautifulSoup(html, "html.parser")
    title = clean(soup.title.get_text()) if soup.title else ""
    remove_boilerplate(soup)
    root: Any = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[str] = []

    def walk(node: Any) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if type(child) is NavigableString and clean(str(child)):
                    blocks.append(clean(str(child)))
            elif isinstance(child, Tag):
                if re.fullmatch(r"h[1-6]", child.name):
                    blocks.append(f"{'#' * int(child.name[1])} {clean(child.get_text())}")
                elif child.name == "table":
                    for row in child.find_all("tr"):
                        cells = [clean(c.get_text()) for c in row.find_all(("th", "td"))]
                        blocks.append(" | ".join(cells))
                elif child.name in ("p", "li", "blockquote", "pre", "figcaption", "dt", "dd"):
                    text = clean(child.get_text())
                    if text:
                        blocks.append(f"- {text}" if child.name == "li" else text)
                else:
                    walk(child)

    walk(root)
    return "\n\n".join(blocks), title


def load_html(path: str | Path, *, doc_id: str | None = None) -> Document:
    path = Path(path)
    html, encoding = read_text_file(path)
    text, page_title = html_to_markdown(html)
    doc = make_document(path, text, "html", doc_id=doc_id, encoding=encoding)
    doc.title = doc.title or page_title
    return doc


def heading_level(style_name: str) -> int | None:
    """Markdown heading level for a Word paragraph style, or None for body text.

    "Title" -> 1, "Heading 1" -> 2, "Heading 2" -> 3, ... (at most 6);
    case-insensitive ("heading 1" too); anything else -> None.
    """
    # TODO: implement
    raise NotImplementedError("TODO: heading_level")


def load_docx(path: str | Path, *, doc_id: str | None = None) -> Document:
    try:
        import docx
        from docx.table import Table
    except ImportError as exc:
        raise _missing("python-docx") from exc
    path = Path(path)
    document: Any = docx.Document(str(path))
    blocks: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            for row in item.rows:
                blocks.append(" | ".join(strip_bidi_controls(c.text).strip() for c in row.cells))
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


def load_pdf(
    path: str | Path,
    *,
    doc_id: str | None = None,
    repair: bool = True,
    fix_visual: bool | Literal["auto"] = "auto",
) -> Document:
    """Load a PDF page by page.

    For each page (numbered from 1): text = page.extract_text() or "";
    if repair: text, visual = repair_pdf_text(text, fix_visual=fix_visual)
    (otherwise visual = False); then strip it.
    Join the pages with "\\n\\n" and record, for each page,
    {"page": n, "start": ..., "end": ..., "visual_order": visual}
    where start/end are offsets of that page's text inside the joined text.
    Title: reader.metadata.title if present. Use make_document(..., pages=pages).
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise _missing("pypdf") from exc
    reader: Any = PdfReader(str(path))
    _ = reader
    # TODO: implement
    raise NotImplementedError("TODO: load_pdf")


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
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported file type {path.suffix!r}")
    return loader(path, doc_id=doc_id)


def load_directory(
    directory: str | Path, pattern: str = "*", *, recursive: bool = False
) -> list[Document]:
    root = Path(directory)
    paths = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    paths = [p for p in paths if p.is_file() and p.suffix.lower() in LOADERS]
    stems = [p.stem for p in paths]
    return [load_document(p, doc_id=p.name if stems.count(p.stem) > 1 else p.stem) for p in paths]


# Imported for you — you will need them.
_ = (GLYPH_INFO, is_mark, joins_both_sides, joins_right, shape, ZWNJ, fix_presentation_forms)

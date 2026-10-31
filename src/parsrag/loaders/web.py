"""HTML loader: keep the content, drop the boilerplate (needs ``beautifulsoup4``)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from parsrag.loaders.base import make_document, missing_extra
from parsrag.loaders.bidi import strip_bidi_controls
from parsrag.loaders.text import read_text_file
from parsrag.types import Document

# Elements that never carry article content.
DROP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "form", "button")
# Page furniture: site header, menus, sidebars, footers.
BOILERPLATE_TAGS = ("nav", "header", "footer", "aside")
BOILERPLATE_ROLES = ("navigation", "banner", "contentinfo", "complementary", "search")
BOILERPLATE_HINT = re.compile(
    r"cookie|consent|breadcrumb|banner|share|social|newsletter|sidebar|menu|advert|popup", re.I
)
_BLOCKS = ("p", "li", "blockquote", "pre", "figcaption", "dt", "dd", "caption")
_SPACE = re.compile(r"[ \t\r\n\f\v]+")


def _clean(text: str) -> str:
    return _SPACE.sub(" ", strip_bidi_controls(text)).strip()


def remove_boilerplate(soup: Any) -> None:
    """Delete scripts, menus, headers, footers, sidebars and cookie banners, in place.

    Three signals: the tag name (``nav``, ``footer`` ...), the ARIA ``role``
    attribute, and class/id names that look like boilerplate (``cookie``,
    ``breadcrumb``, ``sidebar`` ...). Simple rules like these remove most
    furniture from typical sites; for hard cases use a dedicated tool such as
    trafilatura.
    """
    for tag in soup.find_all([*DROP_TAGS, *BOILERPLATE_TAGS]):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": list(BOILERPLATE_ROLES)}):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.decomposed:
            continue
        hint = " ".join([*tag.get("class", []), tag.get("id", "")])
        if hint and BOILERPLATE_HINT.search(hint):
            tag.decompose()


def html_to_markdown(html: str) -> tuple[str, str]:
    """Convert HTML to Markdown-ish text. Returns ``(text, <title>)``.

    Headings become ``#`` lines, list items ``- ``, table rows ``a | b``.
    """
    try:
        from bs4 import BeautifulSoup, NavigableString, Tag
    except ImportError as exc:
        raise missing_extra("beautifulsoup4", "load_html") from exc

    soup = BeautifulSoup(html, "html.parser")
    title = _clean(soup.title.get_text()) if soup.title else ""
    remove_boilerplate(soup)

    root: Any = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[str] = []

    def walk(node: Any) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if type(child) is NavigableString and _clean(str(child)):
                    blocks.append(_clean(str(child)))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if re.fullmatch(r"h[1-6]", name):
                blocks.append(f"{'#' * int(name[1])} {_clean(child.get_text())}")
            elif name == "table":
                for row in child.find_all("tr"):
                    cells = [_clean(c.get_text()) for c in row.find_all(("th", "td"))]
                    if any(cells):
                        blocks.append(" | ".join(cells))
            elif name in _BLOCKS:
                text = _clean(child.get_text())
                if text:
                    blocks.append(f"- {text}" if name == "li" else text)
            else:
                walk(child)

    walk(root)
    return "\n\n".join(blocks), title


def load_html(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Load an HTML page: boilerplate removed, headings kept as Markdown."""
    path = Path(path)
    html, encoding = read_text_file(path)
    text, page_title = html_to_markdown(html)
    doc = make_document(path, text, "html", doc_id=doc_id, encoding=encoding)
    if not doc.title:
        doc.title = page_title
    doc.metadata["html_title"] = page_title
    return doc

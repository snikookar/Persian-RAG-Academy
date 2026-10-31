from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from grader.paths import CORPUS, FIXTURES
from parsrag.loaders import (
    load_directory,
    load_document,
    load_markdown,
    load_text,
    locate,
    looks_visual_order,
    markdown_sections,
    read_text_file,
    recover_zwnj,
    repair_pdf_text,
    shape,
    strip_bidi_controls,
    to_visual,
    visual_order_votes,
    visual_to_logical,
)
from parsrag.loaders.bidi import GLYPH_INFO
from parsrag.normalize import ZWNJ, fix_presentation_forms
from parsrag.types import Document

DOCS = FIXTURES / "documents"
SOURCES = DOCS / "sources"


def _source(name: str) -> str:
    text = (SOURCES / name).read_text(encoding="utf-8")
    return "\n\n".join(page.strip() for page in text.split("\f"))


# ------------------------------------------------------------------- bidi


def test_shape_uses_contextual_forms_and_lam_alef() -> None:
    shaped = shape("بلا")
    assert [GLYPH_INFO[c][1] for c in shaped] == ["initial", "final"]
    assert GLYPH_INFO[shaped[1]][0] == "لا"
    assert fix_presentation_forms(shape("کتابخانه")) == "کتابخانه"


def test_shape_drops_zwnj_unless_asked() -> None:
    assert ZWNJ not in shape(f"می{ZWNJ}شود")
    assert ZWNJ in shape(f"می{ZWNJ}شود", keep_zwnj=True)


@pytest.mark.parametrize(
    "word", [f"می{ZWNJ}شود", f"کتاب{ZWNJ}ها", f"خانهٔ{ZWNJ}ای", f"نمی{ZWNJ}دانم", "کتابخانه"]
)
def test_recover_zwnj_round_trip(word: str) -> None:
    assert recover_zwnj(shape(word)) == shape(word, keep_zwnj=True)
    assert fix_presentation_forms(recover_zwnj(shape(word))) == word


def test_recover_zwnj_needs_glyph_shapes() -> None:
    # Already-normalised text has no shape information: nothing to recover.
    assert recover_zwnj("میشود") == "میشود"


@pytest.mark.parametrize(
    ("visual", "logical"),
    [
        ("مالس", "سلام"),
        ("تسا 45 تمیق", "قیمت 45 است"),
        ("تسا PDF file هخسن", "نسخه PDF file است"),
        ("(مالس)", "(سلام)"),
        ("«کتاب»", "«باتک»"),
        ("ir.mehregan-library.www", "ir.mehregan-library.www"),
    ],
)
def test_visual_to_logical(visual: str, logical: str) -> None:
    assert visual_to_logical(visual) == logical


def test_visual_to_logical_is_an_involution_on_the_corpus() -> None:
    for path in sorted(CORPUS.glob("*.md"))[:8]:
        for line in path.read_text(encoding="utf-8").splitlines():
            assert visual_to_logical(visual_to_logical(line)) == line


def test_visual_order_detection() -> None:
    text = (CORPUS / "kahroba-seller-guide.md").read_text(encoding="utf-8")
    assert not looks_visual_order(text)
    assert looks_visual_order(to_visual(text))  # word votes only
    assert looks_visual_order(to_visual(shape(text)))  # glyph + word votes
    assert not looks_visual_order(shape(text))
    assert not looks_visual_order("مالس")  # too little evidence
    logical, visual = visual_order_votes(to_visual(shape(text)))
    assert visual > 10 * max(logical, 1)


def test_repair_pdf_text_recovers_simulated_legacy_text() -> None:
    original = _source("mehregan-newsletter.txt")
    repaired, visual = repair_pdf_text(to_visual(shape(original)))
    assert visual and repaired == original
    repaired, visual = repair_pdf_text(shape(original), fix_visual="auto")
    assert not visual and repaired == original


def test_strip_bidi_controls() -> None:
    assert strip_bidi_controls("مدل ‏KX-2400‏ و ⁧abc⁩") == "مدل KX-2400 و abc"
    assert strip_bidi_controls(f"می{ZWNJ}شود") == f"می{ZWNJ}شود"  # ZWNJ is not a bidi control


# --------------------------------------------------------- sections & text


def test_markdown_sections_nest() -> None:
    text = "# A\nintro\n## B\nb text\n### C\nc text\n## D\nd text"
    sections = markdown_sections(text)
    assert [(s["title"], s["level"]) for s in sections] == [("A", 1), ("B", 2), ("C", 3), ("D", 2)]
    by_title = {s["title"]: s for s in sections}
    assert by_title["A"]["end"] == len(text)
    assert by_title["B"]["end"] == by_title["D"]["start"]
    assert text[by_title["C"]["start"] : by_title["C"]["end"]] == "### C\nc text\n"


def test_locate_returns_page_and_section_path() -> None:
    text = "# A\nintro\n## B\nb text"
    doc = Document(
        id="d",
        text=text,
        metadata={
            "sections": markdown_sections(text),
            "pages": [{"page": 1, "start": 0, "end": 9}, {"page": 2, "start": 9, "end": len(text)}],
        },
    )
    assert locate(doc, 5) == {"page": 1, "section": "A", "section_path": ["A"]}
    assert locate(doc, text.index("b text")) == {
        "page": 2,
        "section": "B",
        "section_path": ["A", "B"],
    }


def test_read_text_file_encodings(tmp_path: Path) -> None:
    text, enc = read_text_file(DOCS / "kahroba-notice.txt")
    assert enc == "cp1256" and "كهربا" in text and "\r" not in text
    (tmp_path / "bom.txt").write_bytes("﻿سلام".encode())
    assert read_text_file(tmp_path / "bom.txt") == ("سلام", "utf-8-sig")
    (tmp_path / "u16.txt").write_bytes("سلام".encode("utf-16"))
    assert read_text_file(tmp_path / "u16.txt") == ("سلام", "utf-16")


def test_load_text_and_markdown() -> None:
    notice = load_text(DOCS / "kahroba-notice.txt")
    assert notice.id == "kahroba-notice" and notice.metadata["encoding"] == "cp1256"
    assert notice.title.startswith("اطلاعيه")
    guide = load_markdown(CORPUS / "kahroba-seller-guide.md")
    assert guide.title == "راهنمای فروشندگان بازارچهٔ کهربا"
    titles = [s["title"] for s in guide.metadata["sections"]]
    assert "کارمزد فروش" in titles and len(titles) == 9
    assert guide.metadata["format"] == "markdown"


def test_load_document_rejects_unknown_types(tmp_path: Path) -> None:
    (tmp_path / "x.xyz").write_text("?")
    with pytest.raises(ValueError, match="unsupported"):
        load_document(tmp_path / "x.xyz")


def test_load_directory_dedupes_ids(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("الف", encoding="utf-8")
    (tmp_path / "a.md").write_text("# ب", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\0")
    assert [d.id for d in load_directory(tmp_path)] == ["a.md", "a.txt"]


def test_missing_extra_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "bs4", None)
    with pytest.raises(ImportError, match=r"parsrag\[loaders\]"):
        load_document(DOCS / "mehregan-events.html")


# ----------------------------------------------------- loaders with extras


def test_load_html_strips_boilerplate() -> None:
    pytest.importorskip("bs4")
    doc = load_document(DOCS / "mehregan-events.html")
    assert doc.title == "رویدادهای پاییز کتابخانهٔ مهرگان"
    assert doc.text.startswith("# رویدادهای پاییز")
    assert "## هفتهٔ کتاب" in doc.text and "### هزینهٔ ثبت‌نام" in doc.text
    assert "کارگاه داستان‌نویسی | ۲۰۰ هزار تومان | ۴۰۰ هزار تومان" in doc.text
    assert f"می{ZWNJ}کند" in doc.text  # &zwnj; entity kept
    assert "- کارگاه سواد اطلاعاتی" in doc.text
    for junk in ("قبول", "کوکی", "©", "پربازدیدترین", "analytics", "› رویدادها", "عضویت"):
        assert junk not in doc.text
    assert "‎" not in doc.text


def test_load_docx_keeps_structure() -> None:
    pytest.importorskip("docx")
    doc = load_document(DOCS / "kahroba-installation-guide.docx")
    assert doc.title == "راهنمای نصب کولر گازی کهربا"
    assert "## هزینهٔ نصب" in doc.text and "### زمان مراجعهٔ نصاب" in doc.text
    assert "۱۸ هزار | ۵٫۲ میلیون تومان" in doc.text
    assert "- نصاب معمولاً ظرف ۳ روز کاری مراجعه می‌کند." in doc.text
    assert "‏" not in doc.text and "KX-2400" in doc.text
    hit = doc.text.index("فیوز")
    assert locate(doc, hit)["section"] == "پیش از نصب"


def test_load_pdf_repairs_presentation_forms_and_zwnj() -> None:
    pytest.importorskip("pypdf")
    from parsrag.loaders import load_pdf

    raw = load_pdf(DOCS / "kahroba-catalog.pdf", repair=False)
    assert any(c in GLYPH_INFO for c in raw.text) and ZWNJ not in raw.text
    doc = load_pdf(DOCS / "kahroba-catalog.pdf")
    assert doc.text == _source("kahroba-catalog.txt")
    assert doc.title == "کاتالوگ لوازم برقی کهربا"
    pages = doc.metadata["pages"]
    assert [p["page"] for p in pages] == [1, 2]
    assert not any(p["visual_order"] for p in pages)
    assert locate(doc, doc.text.index("اصفهان"))["page"] == 2


def test_load_pdf_fixes_visual_order() -> None:
    pytest.importorskip("pypdf")
    from parsrag.loaders import load_pdf

    raw = load_pdf(DOCS / "mehregan-newsletter.pdf", repair=False)
    assert looks_visual_order(raw.text)
    doc = load_pdf(DOCS / "mehregan-newsletter.pdf")
    assert doc.metadata["pages"][0]["visual_order"] is True
    assert doc.text == _source("mehregan-newsletter.txt")


def test_committed_fixtures_are_reproducible() -> None:
    """DOCX/TXT builders are deterministic and need only the standard library."""
    proc = subprocess.run(
        [sys.executable, str(FIXTURES / "build_fixtures.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

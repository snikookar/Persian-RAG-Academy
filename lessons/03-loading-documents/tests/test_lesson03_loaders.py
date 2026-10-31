"""Lesson 03 — loading Persian documents. Run with: python -m grader run 03"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from grader.paths import CORPUS, FIXTURES
from parsrag import HashingEmbedding, InMemoryVectorStore
from parsrag.loaders.bidi import GLYPH_INFO, shape, to_visual
from parsrag.normalize import ZWNJ, normalize
from parsrag.types import Chunk

DOCS = FIXTURES / "documents"
hint = pytest.mark.hint
EXTRA = "pip install -e '.[loaders]'  (بستهٔ اختیاری loaders نصب نیست)"


def _source(name: str) -> str:
    text = (DOCS / "sources" / name).read_text(encoding="utf-8")
    return "\n\n".join(page.strip() for page in text.split("\f"))


# ------------------------------------------------------------------ plain text


@hint(
    fa="فقط کاراکترهای BIDI_CONTROLS را حذف کنید؛ نیم‌فاصله (U+200C) کنترل دوجهته نیست و باید بماند.",
    en="Remove only BIDI_CONTROLS characters; ZWNJ (U+200C) is not one and must stay.",
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("مدل ‏KX-2400‏", "مدل KX-2400"),
        ("‫سلام‬", "سلام"),
        ("⁧abc⁩ و ‎تست", "abc و تست"),
        (f"می{ZWNJ}شود", f"می{ZWNJ}شود"),
        ("بدون تغییر", "بدون تغییر"),
    ],
)
def test_strip_bidi_controls(ld: ModuleType, raw: str, expected: str) -> None:
    """strip_bidi_controls removes LRM/RLM/embedding marks but keeps ZWNJ"""
    assert ld.strip_bidi_controls(raw) == expected


@hint(
    fa="اول BOM را بررسی کنید، بعد UTF-8 سخت‌گیرانه و در آخر cp1256. انتهای خط \\r\\n را به \\n تبدیل کنید.",
    en="Check BOMs first, then strict UTF-8, then cp1256 last. Convert \\r\\n to \\n.",
)
def test_read_text_file_encodings(ld: ModuleType, tmp_path: Path) -> None:
    """read_text_file detects UTF-8, UTF-8 with BOM, UTF-16 and Windows-1256"""
    text, enc = ld.read_text_file(DOCS / "kahroba-notice.txt")
    assert enc == "cp1256"
    assert "كهربا" in text and "\r" not in text
    (tmp_path / "a.txt").write_bytes("سلام\r\nدنیا".encode())
    assert ld.read_text_file(tmp_path / "a.txt") == ("سلام\nدنیا", "utf-8")
    (tmp_path / "b.txt").write_bytes("﻿سلام".encode())
    assert ld.read_text_file(tmp_path / "b.txt") == ("سلام", "utf-8-sig")
    (tmp_path / "c.txt").write_bytes("سلام".encode("utf-16"))
    assert ld.read_text_file(tmp_path / "c.txt") == ("سلام", "utf-16")


def test_legacy_text_becomes_searchable(ld: ModuleType) -> None:
    """a Windows-1256 file loads and normalizes to clean Persian"""
    doc = ld.load_text(DOCS / "kahroba-notice.txt")
    assert doc.metadata["encoding"] == "cp1256"
    assert "انبار مرکزی کهربا" in normalize(doc.text)


# ------------------------------------------------------------------- structure


@hint(
    fa="پایان هر بخش، شروع اولین سرتیتر بعدی با سطح کوچک‌تر یا مساوی است (یا انتهای متن).",
    en="A section ends where the next heading of the same or higher level starts (or at the end).",
)
def test_markdown_sections(ld: ModuleType) -> None:
    """markdown_sections finds headings with nested character spans"""
    text = "# الف\nمقدمه\n## ب\nمتن ب\n### پ\nمتن پ\n## ت\nمتن ت"
    sections = ld.markdown_sections(text)
    assert [(s["title"], s["level"]) for s in sections] == [
        ("الف", 1), ("ب", 2), ("پ", 3), ("ت", 2),
    ]  # fmt: skip
    spans = {s["title"]: (s["start"], s["end"]) for s in sections}
    assert spans["الف"] == (0, len(text))
    assert spans["ب"] == (text.index("## ب"), text.index("## ت"))
    assert text[slice(*spans["پ"])] == "### پ\nمتن پ\n"
    assert ld.markdown_sections("بدون سرتیتر") == []


def test_markdown_loader_records_sections(ld: ModuleType) -> None:
    """load_markdown keeps headings and records them as sections"""
    doc = ld.load_markdown(CORPUS / "mehregan-annual-report.md")
    assert doc.title.startswith("گزارش سالانهٔ کتابخانهٔ مهرگان")
    titles = [s["title"] for s in doc.metadata["sections"]]
    assert "پروژهٔ دیجیتال‌سازی" in titles
    where = ld.locate(doc, doc.text.index("۹٫۶ میلیارد"))
    assert where["section"] == "بودجه و هزینه‌ها"


# ---------------------------------------------------------------- visual order


@hint(
    fa="توکن‌ها: هر تطبیق _LTR_RUN_RE یک توکن، هر کاراکتر دیگر یک توکن. فهرست توکن‌ها را برعکس کنید و پرانتزها را آینه‌ای کنید.",
    en="Tokens: each _LTR_RUN_RE match is one token, every other char its own. Reverse the tokens; mirror brackets.",
)
@pytest.mark.parametrize(
    ("visual", "logical"),
    [
        ("مالس", "سلام"),
        ("تسا 45 تمیق", "قیمت 45 است"),
        ("تسا PDF file هخسن", "نسخه PDF file است"),
        ("(مالس)", "(سلام)"),
        ("۱۴۰۳ لاس", "سال ۱۴۰۳"),
        ("", ""),
    ],
)
def test_visual_to_logical(ld: ModuleType, visual: str, logical: str) -> None:
    """visual_to_logical reverses Persian but keeps numbers and Latin words intact"""
    assert ld.visual_to_logical(visual) == logical


def test_visual_to_logical_round_trip(ld: ModuleType) -> None:
    """visual_to_logical undoes simulated visual order on real text"""
    for line in (CORPUS / "kahroba-shipping.md").read_text(encoding="utf-8").splitlines():
        assert ld.visual_to_logical(to_visual(line)) == line


@hint(
    fa="شکل حروف را از GLYPH_INFO بخوانید: در ترتیب منطقی کلمه با شکل آغازی (initial) شروع و با شکل پایانی (final) تمام می‌شود.",
    en="Read glyph forms from GLYPH_INFO: logical words start with an initial form and end with a final form.",
)
def test_visual_order_detection(ld: ModuleType) -> None:
    """looks_visual_order spots reversed text, with and without presentation forms"""
    text = (CORPUS / "kahroba-privacy.md").read_text(encoding="utf-8")
    assert not ld.looks_visual_order(text)
    assert not ld.looks_visual_order(shape(text))
    assert ld.looks_visual_order(to_visual(text))
    assert ld.looks_visual_order(to_visual(shape(text)))
    assert not ld.looks_visual_order("مالس")
    logical, visual = ld.visual_order_votes(to_visual(shape("کتابخانهٔ مهرگان برای اعضا است")))
    assert visual > logical


# ------------------------------------------------------------------------ ZWNJ


@hint(
    fa="حرفِ دوطرفه‌چسبان در شکل پایانی/تنها که بلافاصله بعدش حرفی در شکل آغازی/تنها آمده، یعنی آن‌جا نیم‌فاصله بوده.",
    en="A dual-joining letter in final/isolated form directly followed by an initial/isolated letter means a ZWNJ was there.",
)
@pytest.mark.parametrize(
    "word",
    [
        f"می{ZWNJ}شود",
        f"کتاب{ZWNJ}ها",
        f"خانهٔ{ZWNJ}ای",
        f"پیش{ZWNJ}بینی",
        "کتابخانه",
        "دارد",
        "لاله",
    ],
)
def test_recover_zwnj(ld: ModuleType, word: str) -> None:
    """recover_zwnj puts back the ZWNJs a PDF lost — and only those"""
    assert ld.recover_zwnj(shape(word)) == shape(word, keep_zwnj=True)


@hint(
    fa="ترتیب: حذف کنترل‌های دوجهته → اصلاح ترتیب دیداری (هر خط) → بازیابی نیم‌فاصله → NFKC.",
    en="Order: strip bidi controls → fix visual order per line → recover ZWNJ → NFKC.",
)
def test_repair_pdf_text(ld: ModuleType) -> None:
    """repair_pdf_text restores simulated legacy PDF text exactly"""
    original = _source("mehregan-newsletter.txt")
    assert ld.repair_pdf_text(to_visual(shape(original))) == (original, True)
    assert ld.repair_pdf_text(shape(original)) == (original, False)
    text, visual = ld.repair_pdf_text(to_visual(shape(original)), fix_visual=False)
    assert not visual and text != original


# ------------------------------------------------------- real file formats


@hint(
    fa="برچسب‌های DROP_TAGS و BOILERPLATE_TAGS، نقش‌های role و class/idهایی را که با BOILERPLATE_HINT جور می‌شوند decompose کنید.",
    en="Decompose DROP_TAGS/BOILERPLATE_TAGS, boilerplate roles, and class/id names matching BOILERPLATE_HINT.",
)
def test_load_html(ld: ModuleType) -> None:
    """load_html keeps headings and content, drops menus, banners and footers"""
    pytest.importorskip("bs4", reason=EXTRA)
    doc = ld.load_document(DOCS / "mehregan-events.html")
    assert doc.title == "رویدادهای پاییز کتابخانهٔ مهرگان"
    assert "## هفتهٔ کتاب" in doc.text
    assert "کارگاه داستان‌نویسی | ۴۰ هزار تومان | ۸۰ هزار تومان" in doc.text
    assert f"برگزار می{ZWNJ}کند" in doc.text
    for junk in ("قبول", "کوکی", "©", "پربازدیدترین", "› رویدادها"):
        assert junk not in doc.text, f"boilerplate left in text: {junk}"


@hint(
    fa="«Title» سطح ۱ است و «Heading N» سطح N+1؛ بزرگ و کوچکی حروف مهم نیست.",
    en="'Title' is level 1 and 'Heading N' is level N+1; ignore case.",
)
@pytest.mark.parametrize(
    ("style", "level"),
    [
        ("Title", 1),
        ("Heading 1", 2),
        ("heading 2", 3),
        ("Heading 9", 6),
        ("Normal", None),
        ("List Bullet", None),
    ],
)
def test_heading_level(ld: ModuleType, style: str, level: int | None) -> None:
    """heading_level maps Word styles to Markdown heading levels"""
    assert ld.heading_level(style) == level


def test_load_docx(ld: ModuleType) -> None:
    """load_docx keeps headings, lists and tables, and removes RLM marks"""
    pytest.importorskip("docx", reason=EXTRA)
    doc = ld.load_document(DOCS / "kahroba-installation-guide.docx")
    assert doc.title == "راهنمای نصب کولر گازی کهربا"
    assert "## هزینهٔ نصب" in doc.text
    assert "۱۸ هزار | ۱٫۲ میلیون تومان" in doc.text
    assert "‏" not in doc.text
    assert ld.locate(doc, doc.text.index("فیوز"))["section"] == "پیش از نصب"


@hint(
    fa="آفست شروع هر صفحه = طول متن صفحه‌های قبلی + ۲ نویسه برای هر جداکنندهٔ «\\n\\n».",
    en="Each page starts after the previous pages' text plus 2 characters per '\\n\\n' separator.",
)
def test_load_pdf_presentation_forms(ld: ModuleType) -> None:
    """load_pdf repairs a PDF stored as presentation forms and records pages"""
    pytest.importorskip("pypdf", reason=EXTRA)
    raw = ld.load_pdf(DOCS / "kahroba-catalog.pdf", repair=False)
    assert any(c in GLYPH_INFO for c in raw.text)
    doc = ld.load_pdf(DOCS / "kahroba-catalog.pdf")
    assert doc.text == _source("kahroba-catalog.txt")
    pages = doc.metadata["pages"]
    assert [p["page"] for p in pages] == [1, 2]
    for p in pages:
        assert doc.text[p["start"] : p["end"]] == doc.text[p["start"] : p["end"]].strip()
    assert ld.locate(doc, doc.text.index("اصفهان"))["page"] == 2


def test_load_pdf_visual_order(ld: ModuleType) -> None:
    """load_pdf detects and fixes a PDF stored in visual order"""
    pytest.importorskip("pypdf", reason=EXTRA)
    doc = ld.load_document(DOCS / "mehregan-newsletter.pdf")
    assert doc.metadata["pages"][0]["visual_order"] is True
    assert doc.text == _source("mehregan-newsletter.txt")


# ------------------------------------------------------------------ the payoff


def test_repaired_pdf_is_findable(ld: ModuleType) -> None:
    """after repair, a search finds the newsletter; before repair it cannot"""
    pytest.importorskip("pypdf", reason=EXTRA)
    query = normalize("قفسهٔ هوشمند برای امانت شبانه")

    def best(repair: bool) -> tuple[str, float]:
        store = InMemoryVectorStore(HashingEmbedding())
        docs = [
            ld.load_pdf(DOCS / f"{n}.pdf", repair=repair)
            for n in ("kahroba-catalog", "mehregan-newsletter")
        ]
        store.add(Chunk(id=d.id, doc_id=d.id, text=normalize(d.text)) for d in docs)
        top = store.search(query, k=1)[0]
        return top.chunk.doc_id, top.score

    repaired_doc, repaired_score = best(True)
    _, raw_score = best(False)
    assert repaired_doc == "mehregan-newsletter"
    assert repaired_score > raw_score + 0.2

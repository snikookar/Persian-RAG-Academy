"""Generate the binary loader fixtures in ``data/fixtures/documents/`` (lesson 03).

    python data/fixtures/build_fixtures.py            # rebuild everything
    python data/fixtures/build_fixtures.py --check    # verify committed DOCX/TXT are up to date

All text is original and lives in ``documents/sources/`` (UTF-8, logical order).
The generated files reproduce problems found in real Persian documents:

* ``kahroba-catalog.pdf`` — glyphs stored as Arabic *presentation forms* in
  logical order, ZWNJs lost (typical of many PDF producers).
* ``sepidar-newsletter.pdf`` — presentation forms stored in *visual* order,
  ZWNJs lost (typical of older PDF tools): naive extraction reads backwards.
* ``kahroba-installation-guide.docx`` — headings, a table, and invisible
  RLM marks around Latin model numbers, as Word inserts them.
* ``kahroba-notice.txt`` — encoded in Windows-1256 with Arabic ي/ك, like
  files saved by old Windows software.

The DOCX and TXT builders only use the standard library and are byte-for-byte
deterministic. The PDF builder needs ``fpdf2`` and a TrueType font with Arabic
glyphs (DejaVu Sans; matplotlib ships one): ``pip install -e ".[dev]"``.
Nothing here is needed to *run* the tests — the generated files are committed.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE / "documents"
SOURCES = DOCS / "sources"
sys.path.insert(0, str(HERE.parent.parent / "src"))

from parsrag.loaders.bidi import shape, to_visual  # noqa: E402

RLM = "‏"
PAGE_BREAK = "\f"

# --------------------------------------------------------------------- PDF

_LTR_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-/]*(?: +[A-Za-z0-9][A-Za-z0-9._\-/]*)*")


def _find_font() -> Path:
    candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    try:
        import matplotlib

        candidates.insert(0, Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    except ImportError:
        pass
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("DejaVuSans.ttf not found: pip install matplotlib, or pass --font")


def _tokens(line: str) -> list[str]:
    """Split a logical line into LTR runs and single characters."""
    out: list[str] = []
    pos = 0
    for m in _LTR_TOKEN.finditer(line):
        out.extend(line[pos : m.start()])
        out.append(m.group())
        pos = m.end()
    out.extend(line[pos:])
    return out


def build_pdf(source: Path, target: Path, *, visual: bool, font: Path, title: str) -> None:
    """Draw each line right-aligned, one glyph (or LTR run) per text operation.

    The *content-stream order* decides what text extractors return, so we
    control it explicitly: logical order for the catalog, display order
    (left to right) for the legacy newsletter. Both look identical on screen.
    """
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_creation_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    pdf.set_title(title)
    pdf.set_producer("parsrag build_fixtures.py")
    pdf.add_font("dejavu", "", str(font))
    pdf.set_font("dejavu", size=12)
    right, line_height = 190.0, 9.0
    for page in source.read_text(encoding="utf-8").split(PAGE_BREAK):
        pdf.add_page()
        y = 30.0
        for line in page.strip().split("\n"):
            shaped = shape(line)  # presentation forms, ZWNJ dropped
            if visual:
                display = to_visual(shaped)
                x = right - pdf.get_string_width(display)
                if x < 20:
                    raise ValueError(f"line too long for the page: {line!r}")
                for ch in display:  # stream order == display order
                    pdf.text(x, y, ch)
                    # a combining mark comes right before its base and sits on it
                    if unicodedata.category(ch) != "Mn":
                        x += pdf.get_string_width(ch)
            else:
                tokens = _tokens(shaped)
                widths = [pdf.get_string_width(t) for t in tokens]
                if sum(widths) > right - 20:
                    raise ValueError(f"line too long for the page: {line!r}")
                x = right
                for token, width in zip(
                    tokens, widths, strict=True
                ):  # stream order == logical order
                    x -= width
                    pdf.text(x, y, token)
            y += line_height
    pdf.output(str(target))


# -------------------------------------------------------------------- DOCX

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

DOCX_CONTENT: list[tuple[str, object]] = [
    ("Title", "راهنمای نصب کولر گازی کهربا"),
    ("Normal", "این راهنما مراحل درخواست نصب کولر گازی خریداری‌شده از کهربا را توضیح می‌دهد."),
    ("Heading1", "پیش از نصب"),
    ("Normal", "محل نصب پنل داخلی باید دست‌کم ۱۵ سانتی‌متر با سقف فاصله داشته باشد."),
    (
        "Normal",
        f"برای مدل‌های {RLM}KX-2400{RLM} و {RLM}KX-3000{RLM} پریز برق جداگانه با فیوز ۲۵ آمپر لازم است.",
    ),
    ("Heading1", "هزینهٔ نصب"),
    ("Normal", "هزینهٔ نصب بسته به ظرفیت دستگاه در جدول زیر آمده است:"),
    (
        "Table",
        [
            ["ظرفیت دستگاه", "هزینهٔ نصب"],
            ["۹ هزار", "۹۰۰ هزار تومان"],
            ["۱۸ هزار", "۱٫۲ میلیون تومان"],
            ["۲۴ هزار", "۱٫۵ میلیون تومان"],
        ],
    ),
    ("Normal", "لوله‌کشی بیش از ۳ متر جداگانه حساب می‌شود."),
    ("Heading2", "زمان مراجعهٔ نصاب"),
    ("ListBullet", "درخواست نصب را از بخش «سفارش‌های من» ثبت کنید."),
    ("ListBullet", "نصاب معمولاً ظرف ۳ روز کاری مراجعه می‌کند."),
    ("Heading1", "گارانتی"),
    ("Normal", "اگر دستگاه را نصاب غیرمجاز نصب کند و به همین دلیل آسیب ببیند، گارانتی باطل می‌شود."),
]


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _paragraph(style: str, text: str) -> str:
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{style}"/><w:bidi/></w:pPr>'
        f'<w:r><w:rPr><w:rtl/></w:rPr><w:t xml:space="preserve">{_esc(text)}</w:t></w:r></w:p>'
    )


def _table(rows: list[list[str]]) -> str:
    body = "".join(
        "<w:tr>"
        + "".join(f"<w:tc><w:tcPr/>{_paragraph('Normal', cell)}</w:tc>" for cell in row)
        + "</w:tr>"
        for row in rows
    )
    return f'<w:tbl><w:tblPr><w:bidiVisual/><w:tblStyle w:val="TableGrid"/></w:tblPr>{body}</w:tbl>'


def _style(style_id: str, name: str, *, para: bool = True) -> str:
    kind = "paragraph" if para else "table"
    return f'<w:style w:type="{kind}" w:styleId="{style_id}"><w:name w:val="{name}"/></w:style>'


def build_docx(target: Path) -> None:
    blocks = [
        _table(value) if style == "Table" else _paragraph(style, str(value))  # type: ignore[arg-type]
        for style, value in DOCX_CONTENT
    ]
    document = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W}"><w:body>{"".join(blocks)}</w:body></w:document>'
    )
    styles = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="{_W}">'
        + _style("Normal", "Normal")
        + _style("Title", "Title")
        + _style("Heading1", "heading 1")
        + _style("Heading2", "heading 2")
        + _style("ListBullet", "List Bullet")
        + _style("TableGrid", "Table Grid", para=False)
        + "</w:styles>"
    )
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{pkg}">'
            f'<Relationship Id="rId1" Type="{rel}/officeDocument" Target="word/document.xml"/>'
            f'<Relationship Id="rId2" Type="{pkg}/metadata/core-properties" Target="docProps/core.xml"/>'
            "</Relationships>"
        ),
        "word/_rels/document.xml.rels": (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{pkg}">'
            f'<Relationship Id="rId1" Type="{rel}/styles" Target="styles.xml"/></Relationships>'
        ),
        "word/document.xml": document,
        "word/styles.xml": styles,
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/">'
            "<dc:title>راهنمای نصب کولر گازی کهربا</dc:title></cp:coreProperties>"
        ),
    }
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, content.encode("utf-8"))


# --------------------------------------------------------------------- TXT


def legacy_cp1256(text: str) -> bytes:
    """What old Windows software saved: Arabic ي/ك, no hamza-above, CRLF, Windows-1256."""
    text = text.replace("ی", "ي").replace("ک", "ك").replace("هٔ", "ه")
    return text.replace("\n", "\r\n").encode("cp1256")


def build_txt(target: Path) -> None:
    target.write_bytes(legacy_cp1256((SOURCES / "kahroba-notice.txt").read_text(encoding="utf-8")))


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify DOCX/TXT are up to date")
    parser.add_argument("--font", type=Path, help="TrueType font with Arabic glyphs")
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            stale = []
            for name, builder in (
                ("kahroba-installation-guide.docx", build_docx),
                ("kahroba-notice.txt", build_txt),
            ):
                builder(Path(tmp) / name)
                if (Path(tmp) / name).read_bytes() != (DOCS / name).read_bytes():
                    stale.append(name)
        print("stale: " + ", ".join(stale) if stale else "fixtures are up to date")
        return 1 if stale else 0

    DOCS.mkdir(exist_ok=True)
    build_docx(DOCS / "kahroba-installation-guide.docx")
    build_txt(DOCS / "kahroba-notice.txt")
    if not args.skip_pdf:
        font = args.font or _find_font()
        build_pdf(
            SOURCES / "kahroba-catalog.txt",
            DOCS / "kahroba-catalog.pdf",
            visual=False,
            font=font,
            title="کاتالوگ لوازم برقی کهربا",
        )
        build_pdf(
            SOURCES / "sepidar-newsletter.txt",
            DOCS / "sepidar-newsletter.pdf",
            visual=True,
            font=font,
            title="خبرنامهٔ کتابخانهٔ سپیدار",
        )
    for path in sorted(DOCS.glob("*.*")):
        print(f"{path.stat().st_size:>8,} bytes  {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Draw the GitHub social preview image (1280×640) with the name in English and Persian.

    pip install -e ".[dev]"                     # Pillow comes with matplotlib
    python scripts/make_social_preview.py       # → docs/images/social-preview.png
    python scripts/make_social_preview.py --font path/to/Vazirmatn-Bold.ttf

Persian needs shaping (joined letter forms) and right-to-left ordering. When
Pillow is built with libraqm it does both itself. Otherwise we fall back to the
toy shaper from lesson 03 (``parsrag.loaders.bidi.to_visual``): it turns the text
into presentation forms in visual order, which any font with those glyphs (DejaVu
Sans has them) can draw left to right.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images" / "social-preview.png"
SIZE = (1280, 640)
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)
BG, INK, MUTED, ACCENT, ACCENT_2 = "#fcfcfb", "#1f1f1d", "#52514e", "#2a78d6", "#1baf7a"


def find_font(explicit: str | None) -> str:
    for candidate in (explicit, *FONT_CANDIDATES):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("no font with Persian glyphs found; pass --font path/to/font.ttf")


def use_raqm() -> bool:
    return bool(features.check("raqm"))


def persian(text: str) -> tuple[str, dict[str, object]]:
    """Return (text, draw kwargs) so that Persian renders correctly with or without raqm."""
    if use_raqm():
        return text, {"direction": "rtl", "language": "fa"}
    from parsrag.loaders.bidi import to_visual

    return to_visual(text), {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font", help="TTF font with Arabic-script glyphs (e.g. Vazirmatn)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    font_path = find_font(args.font)

    def font(size: int) -> ImageFont.FreeTypeFont:
        engine = ImageFont.Layout.RAQM if use_raqm() else ImageFont.Layout.BASIC
        return ImageFont.truetype(font_path, size, layout_engine=engine)

    img = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(img)
    w, h = SIZE

    # A thin pipeline motif along the bottom: documents → chunks → search → answer.
    y = h - 78
    stops = ["normalize", "load", "chunk", "embed", "hybrid search", "rerank", "cite", "evaluate"]
    x0, x1 = 90, w - 90
    draw.line([(x0, y), (x1, y)], fill="#d9d8d2", width=4)
    small = font(22)
    for i, label in enumerate(stops):
        x = x0 + (x1 - x0) * i / (len(stops) - 1)
        color = ACCENT if i % 2 == 0 else ACCENT_2
        draw.ellipse([x - 11, y - 11, x + 11, y + 11], fill=color, outline=BG, width=3)
        tw = draw.textlength(label, font=small)
        draw.text((x - tw / 2, y + 22), label, font=small, fill=MUTED)

    # English name and pitch (left aligned), Persian name (right aligned).
    draw.rectangle([90, 96, 102, 300], fill=ACCENT)
    draw.text((130, 92), "Persian RAG", font=font(96), fill=INK)
    draw.text((130, 196), "Academy", font=font(96), fill=INK)
    draw.text(
        (130, 330), "Build, evaluate and ship RAG for Persian text", font=font(34), fill=MUTED
    )
    draw.text(
        (130, 378), "14 lessons · parsrag library · offline, zero API keys", font=font(28),
        fill=MUTED,
    )  # fmt: skip

    fa_title, kw = persian("آکادمی RAG فارسی")
    big = font(70)
    box = draw.textbbox((0, 0), fa_title, font=big, **kw)  # type: ignore[arg-type]
    draw.text((w - 90 - (box[2] - box[0]), 420), fa_title, font=big, fill=ACCENT, **kw)  # type: ignore[arg-type]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out, optimize=True)
    shaping = "libraqm" if kw else "parsrag.loaders.bidi.to_visual"
    print(f"saved {args.out} ({w}×{h}, font {Path(font_path).name}, shaping: {shaping})")


if __name__ == "__main__":
    main()

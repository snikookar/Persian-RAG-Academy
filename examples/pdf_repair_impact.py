"""How much does each PDF repair step recover? (lesson 03)

    pip install -e ".[loaders]"
    python examples/pdf_repair_impact.py

For each fixture PDF we compare the extracted text with the original source
text (``data/fixtures/documents/sources/``) after each cumulative step:

* word recall   — share of the source's words found in the extracted text
                  (words are split on whitespace, so a missing ZWNJ counts as an error)
* phrase search — can a plain substring search find a few key phrases?
* exact         — is the result identical to the source?

The fixtures were generated with parsrag's own shaper (data/fixtures/build_fixtures.py),
so they follow the same model the repair code assumes. Real PDFs are messier.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from parsrag.loaders import load_pdf, looks_visual_order, recover_zwnj, visual_to_logical
from parsrag.normalize import fix_presentation_forms

DOCS = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "documents"
PHRASES = {
    "kahroba-catalog": ["پرفروش‌ترین لوازم برقی", "18 ماه گارانتی", "مرکز خدمات پس از فروش"],
    "mehregan-newsletter": ["قفسهٔ هوشمند", "امانت شبانه", "می‌توانند کتاب‌های رزروشده"],
}


def visual_fix(text: str) -> str:
    if not looks_visual_order(text):
        return text
    return "\n".join(visual_to_logical(line) for line in text.split("\n"))


STEPS: list[tuple[str, Callable[[str], str]]] = [
    ("raw pypdf text", lambda t: t),
    ("+ NFKC presentation forms", fix_presentation_forms),
    ("+ visual-order fix", lambda t: fix_presentation_forms(visual_fix(t))),
    ("+ ZWNJ recovery (= load_pdf)", lambda t: fix_presentation_forms(recover_zwnj(visual_fix(t)))),
]


def word_recall(text: str, reference: str) -> float:
    ref, got = Counter(reference.split()), Counter(text.split())
    return sum((ref & got).values()) / sum(ref.values())


def main() -> None:
    for name, phrases in PHRASES.items():
        source = (DOCS / "sources" / f"{name}.txt").read_text(encoding="utf-8")
        reference = "\n\n".join(page.strip() for page in source.split("\f"))
        raw = load_pdf(DOCS / f"{name}.pdf", repair=False).text
        print(f"\n{name}.pdf")
        print(f"{'step':<30} {'word recall':>11} {'phrases':>8} {'exact':>6}")
        for label, step in STEPS:
            text = step(raw)
            found = sum(p in text for p in phrases)
            print(
                f"{label:<30} {word_recall(text, reference):>11.3f} "
                f"{found:>6}/{len(phrases)} {text == reference!s:>6}"
            )


if __name__ == "__main__":
    main()

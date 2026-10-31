"""Repairing Persian text extracted from PDFs and other "visual" sources (lesson 03).

PDF files store *glyphs and positions*, not text. When a PDF is converted
back to text, Persian suffers from four well-known problems:

1. **Arabic presentation forms.** Many PDF producers write the contextual
   glyph (``ﺑ`` BEH INITIAL FORM, U+FE91) instead of the letter (``ب``
   U+0628). The text *looks* right but no search engine will match it.
   Fix: Unicode NFKC on the presentation-form ranges
   (:func:`parsrag.normalize.fix_presentation_forms`).
2. **Visual order.** Older tools wrote right-to-left lines in *display
   order*, left to right, so ``سلام`` is extracted as ``مالس``.
   Detect with :func:`looks_visual_order`, repair line by line with
   :func:`visual_to_logical` (a heuristic — see its docstring for limits).
3. **Lost ZWNJ.** A zero-width non-joiner is not a glyph, so it rarely
   survives: ``می‌شود`` comes back as ``میشود`` or ``می شود``. If the text
   still contains presentation forms we can *recover* it from the glyph
   shapes (:func:`recover_zwnj`), because a non-joined final form followed by
   an initial form inside a word is exactly what a ZWNJ produces.
4. **Bidi control characters.** Word processors and web pages sprinkle
   invisible LRM/RLM/embedding marks into mixed Persian/English lines. They
   break exact matching, so :func:`strip_bidi_controls` removes them.

:func:`shape` and :func:`to_visual` do the *opposite* transformations. They
exist to build realistic test fixtures (``data/fixtures/build_fixtures.py``)
and to test the repair functions on any text.
"""

from __future__ import annotations

import re
import unicodedata
from itertools import chain
from typing import Literal

from parsrag.normalize import ZWNJ, fix_presentation_forms

Form = Literal["isolated", "final", "initial", "medial"]

# Explicit bidi formatting characters: LRM, RLM, ALM, LRE..RLO, LRI..PDI.
BIDI_CONTROLS = "‎‏؜‪‫‬‭‮⁦⁧⁨⁩"
_BIDI_CONTROLS_RE = re.compile(f"[{BIDI_CONTROLS}]")

# ------------------------------------------------------------ glyph tables
#
# Built from the Unicode database instead of being typed by hand: every
# presentation form decomposes (compatibility mapping) to its base letter(s)
# with a tag that names its position, e.g. U+FE91 -> "<initial> 0628".

_FORM_OF: dict[tuple[str, str], str] = {}  # (base, form) -> glyph
GLYPH_INFO: dict[str, tuple[str, str]] = {}  # glyph -> (base, form)

for _cp in chain(range(0xFE70, 0xFF00), range(0xFB50, 0xFE00)):
    _ch = chr(_cp)
    _decomp = unicodedata.decomposition(_ch).split()
    if not _decomp or _decomp[0] not in ("<isolated>", "<final>", "<initial>", "<medial>"):
        continue
    _base = "".join(chr(int(code, 16)) for code in _decomp[1:])
    _form = _decomp[0][1:-1]
    GLYPH_INFO[_ch] = (_base, _form)
    _FORM_OF.setdefault((_base, _form), _ch)


def joins_both_sides(letter: str) -> bool:
    """Dual-joining letters (ب، س، ک، ی، ...) have a medial form."""
    return (letter, "medial") in _FORM_OF


def joins_right(letter: str) -> bool:
    """Letters that connect to the previous letter: dual- and right-joining (ا، د، ر، و، ...)."""
    return (letter, "final") in _FORM_OF


def is_mark(ch: str) -> bool:
    """Combining marks (harakat, hamza above, ...) are transparent for joining."""
    return unicodedata.category(ch) == "Mn"


def is_rtl(ch: str) -> bool:
    """True for strong right-to-left characters (Arabic/Persian letters, Hebrew, ...)."""
    return unicodedata.bidirectional(ch) in ("R", "AL")


def strip_bidi_controls(text: str) -> str:
    """Remove invisible bidi formatting characters (LRM, RLM, ALM, embeddings, isolates)."""
    return _BIDI_CONTROLS_RE.sub("", text)


# ------------------------------------------------------------------ shaping

_LAM = "ل"
_LAM_ALEF = {"ا", "آ", "أ", "إ"}


def shape(text: str, *, keep_zwnj: bool = False) -> str:
    """Replace letters by their contextual presentation forms, like a PDF producer would.

    ``شود`` becomes ``ﺷﻮﺩ``; lam + alef becomes the ligature ``ﻻ``. A ZWNJ
    breaks the joining (``می‌شود`` → ``ﻣﯽﺷﻮﺩ``) and is then dropped unless
    ``keep_zwnj`` is true — which is exactly how ZWNJs get lost in PDFs.
    This is a deliberately small shaper for test data, not a full text-layout engine.
    """
    out: list[str] = []
    n = len(text)

    def neighbour(i: int, step: int) -> str | None:
        j = i + step
        while 0 <= j < n and is_mark(text[j]):
            j += step
        return text[j] if 0 <= j < n else None

    i = 0
    while i < n:
        ch = text[i]
        if ch == ZWNJ:
            if keep_zwnj:
                out.append(ch)
            i += 1
            continue
        if not joins_right(ch):
            out.append(ch)
            i += 1
            continue
        prev = neighbour(i, -1)
        joins_prev = prev is not None and joins_both_sides(prev)
        nxt = neighbour(i, +1)
        if ch == _LAM and nxt in _LAM_ALEF:
            ligature = _LAM + nxt
            form = "final" if joins_prev else "isolated"
            out.append(_FORM_OF.get((ligature, form), ligature))
            # skip marks between lam and alef, then the alef itself
            i += 1
            while i < n and is_mark(text[i]):
                out.append(text[i])
                i += 1
            i += 1
            continue
        joins_next = nxt is not None and joins_both_sides(ch) and joins_right(nxt)
        form = {
            (True, True): "medial",
            (True, False): "final",
            (False, True): "initial",
            (False, False): "isolated",
        }[(joins_prev, joins_next)]
        out.append(_FORM_OF.get((ch, form), ch))
        i += 1
    return "".join(out)


# ------------------------------------------------------------ visual order

# Runs that stay left-to-right inside a right-to-left line: Latin words
# (possibly several, separated by spaces) and numbers (with their separators).
_LATIN = "A-Za-z"
_DIGIT = "0-9۰-۹٠-٩"
_LTR_RUN_RE = re.compile(
    rf"[{_LATIN}][{_LATIN}{_DIGIT}_\-.@/]*(?: +[{_LATIN}{_DIGIT}][{_LATIN}{_DIGIT}_\-.@/]*)*"
    rf"|[{_DIGIT}](?:[{_DIGIT}]|[.,:/٫٬%\-](?=[{_DIGIT}]))*"
)
_MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")


def visual_to_logical(line: str) -> str:
    """Convert one line from visual (display) order to logical order — a heuristic.

    The line is reversed as a sequence of *tokens*: every Persian character is
    its own token, while Latin words and numbers are kept as left-to-right
    runs; brackets and guillemets are mirrored. Because the operation is its
    own inverse, the same function simulates visual order (:func:`to_visual`).

    Combining marks (e.g. the hamza in «هٔ») are assumed to be stored just
    before their base glyph in visual order, which reversal puts right back.

    Limits (be honest with your users): it assumes a right-to-left paragraph,
    one visual line at a time. Real bidi layout (the Unicode Bidirectional
    Algorithm) has more rules — numbers next to neutrals, nested embeddings,
    multi-column pages, lines that were wrapped *before* reordering — so
    complex mixed lines may still come out wrong. Always spot-check.
    """
    tokens: list[str] = []
    pos = 0
    for m in _LTR_RUN_RE.finditer(line):
        tokens.extend(line[pos : m.start()])
        tokens.append(m.group())
        pos = m.end()
    tokens.extend(line[pos:])
    return "".join(t.translate(_MIRROR) if len(t) == 1 else t for t in reversed(tokens))


def to_visual(text: str) -> str:
    """Simulate visual-order storage, line by line (the inverse of :func:`visual_to_logical`)."""
    return "\n".join(visual_to_logical(line) for line in text.split("\n"))


# Frequent Persian words whose reversal is *not* a common word. Palindromes
# and pairs like «در»/«رد» are left out on purpose.
_FORWARD_WORDS = frozenset(
    {"از", "به", "این", "که", "را", "با", "است", "برای", "آن", "های", "شده", "باید", "یک"}
)
_BACKWARD_WORDS = frozenset(w[::-1] for w in _FORWARD_WORDS)
_ARABIC_WORD_RE = re.compile(r"[؀-ۿﭐ-﷿ﹰ-﻿]+")


def visual_order_votes(text: str) -> tuple[int, int]:
    """Count evidence for (logical, visual) order in ``text``.

    Two independent clues:

    * **Glyph shapes** (when presentation forms are present): in logical order
      a word starts with an initial form and ends with a final form; in visual
      order it is the other way round.
    * **Common words**: «است»، «این»، «برای» ... are frequent; their reversals
      («تسا»، «نیا»، «یارب») are not Persian words.
    """
    logical = visual = 0
    for word in _ARABIC_WORD_RE.findall(text):
        glyphs = [c for c in word if not is_mark(c)]
        first, last = GLYPH_INFO.get(glyphs[0]), GLYPH_INFO.get(glyphs[-1])
        if first:
            logical += first[1] == "initial"
            visual += first[1] in ("final", "medial")
        if last:
            logical += last[1] == "final"
            visual += last[1] in ("initial", "medial")
    for word in _ARABIC_WORD_RE.findall(fix_presentation_forms(text)):
        logical += word in _FORWARD_WORDS
        visual += word in _BACKWARD_WORDS
    return logical, visual


def looks_visual_order(text: str, min_votes: int = 3) -> bool:
    """Guess whether ``text`` was extracted in visual (reversed) order.

    Returns ``True`` only when visual-order evidence wins *and* there is at
    least ``min_votes`` of it, so short or ambiguous snippets are left alone.
    """
    logical, visual = visual_order_votes(text)
    return visual > logical and visual >= min_votes


# ---------------------------------------------------------------- ZWNJ


def recover_zwnj(text: str) -> str:
    """Re-insert ZWNJs that PDF extraction lost, using the glyph shapes.

    Inside a word, a dual-joining letter in its *final* or *isolated* form
    that is directly followed by a letter in its *initial* or *isolated* form
    means the two were *not* joined although they could have been — the
    signature of a ZWNJ (``ﻣﯽﺷﻮﺩ`` → ``می‌شود``). Must run on logical-order
    text *before* NFKC, which throws the shape information away.
    """
    n = len(text)
    insert_before: set[int] = set()
    for i, ch in enumerate(text):
        info = GLYPH_INFO.get(ch)
        if info is None or len(info[0]) != 1 or info[1] not in ("final", "isolated"):
            continue
        if not joins_both_sides(info[0]):
            continue
        j = i + 1
        while j < n and is_mark(text[j]):  # e.g. hamza above in «هٔ»
            j += 1
        nxt = GLYPH_INFO.get(text[j]) if j < n else None
        if nxt is not None and nxt[1] in ("initial", "isolated") and joins_right(nxt[0][0]):
            insert_before.add(j)
    return "".join(ZWNJ + c if i in insert_before else c for i, c in enumerate(text))


# ------------------------------------------------------------ the pipeline


def repair_pdf_text(
    text: str,
    *,
    fix_visual: bool | Literal["auto"] = "auto",
    zwnj: bool = True,
) -> tuple[str, bool]:
    """Repair text extracted from a PDF page. Returns ``(text, visual_order_was_fixed)``.

    Order matters: strip bidi controls → fix visual order (line by line) →
    recover ZWNJs from glyph shapes → NFKC the presentation forms.
    Full Persian normalisation (lesson 02) is still applied afterwards, at
    indexing time, like for every other source.
    """
    text = strip_bidi_controls(text)
    visual = looks_visual_order(text) if fix_visual == "auto" else bool(fix_visual)
    if visual:
        text = "\n".join(visual_to_logical(line) for line in text.split("\n"))
    if zwnj:
        text = recover_zwnj(text)
    return fix_presentation_forms(text), visual


__all__ = [
    "BIDI_CONTROLS",
    "GLYPH_INFO",
    "is_mark",
    "is_rtl",
    "joins_both_sides",
    "joins_right",
    "looks_visual_order",
    "recover_zwnj",
    "repair_pdf_text",
    "shape",
    "strip_bidi_controls",
    "to_visual",
    "visual_order_votes",
    "visual_to_logical",
]

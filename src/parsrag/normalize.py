"""Persian text normalisation (lesson 02).

Persian text found in the wild mixes visually identical but *different*
Unicode characters (Arabic ``ي``/``ك`` vs Persian ``ی``/``ک``), three digit
systems, optional diacritics, decorative tatweel, and inconsistent
zero-width non-joiners (ZWNJ, ``\\u200c``). To a computer ``كتاب`` and
``کتاب`` are different strings, which silently hurts search.

Every step is a small pure function, and :class:`NormalizerConfig` lets you
switch each one on or off::

    >>> normalize("كتاب هاي  من ۱۲٣")
    'کتاب‌های من 123'
    >>> normalize("۱۲٣", digits="fa")
    '۱۲۳'
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Literal

ZWNJ = "‌"

DigitTarget = Literal["en", "fa", "keep"]

# Arabic code points that have a distinct Persian counterpart.
ARABIC_TO_PERSIAN: dict[str, str] = {
    "ي": "ی",  # ي ARABIC YEH            -> ی FARSI YEH
    "ى": "ی",  # ى ALEF MAKSURA          -> ی
    "ك": "ک",  # ك ARABIC KAF            -> ک KEHEH
    "ۀ": "هٔ",  # ۀ HEH WITH YEH ABOVE -> هٔ (canonical decomposed form)
    "ە": "ه",  # ە AE (used as heh)       -> ه
}

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"

# Harakat, tanwin, shadda, sukun (U+064B..U+0652) and superscript alef (U+0670).
# Hamza above/below (U+0654/U+0655) and maddah (U+0653) are *letters* in
# Persian spelling (هٔ, آ), so they are kept.
DIACRITICS_RE = re.compile("[ً-ْٰ]")
TATWEEL = "ـ"

# Characters that are often (mis)used in place of a ZWNJ.
ZWNJ_LOOKALIKES = "‍­​﻿"  # ZWJ, soft hyphen, ZWSP, BOM

_PERSIAN_LETTER = "ء-غف-يپچژکگیٔ"
_L = f"[{_PERSIAN_LETTER}]"

# "می رود" / "نمی دانم"  ->  "می‌رود" / "نمی‌دانم"  (only when separated by spaces)
_PREFIX_RE = re.compile(rf"(?<!{_L})(ن?می)[ {ZWNJ}]+(?={_L})")

# "کتاب ها" / "بزرگ ترین"  ->  "کتاب‌ها" / "بزرگ‌ترین"
_SUFFIXES = "هایشان|هایتان|هایمان|هایی|هایم|هایت|هایش|های|ها|ترین|تری|تر"
# The previous word must have 2+ letters so "خشک و تر" (dry and wet) is untouched.
_SUFFIX_RE = re.compile(rf"(?<={_L}{_L}) +({_SUFFIXES})(?!{_L})")

# Copula/indefinite after a word ending in ه: "خانه ای" -> "خانه‌ای", "رفته اند" -> "رفته‌اند"
_HEH_SUFFIX_RE = re.compile(rf"(?<={_L}ه) +(ای|ام|ایم|اید|اند)(?!{_L})")

_CHAR_TABLE = str.maketrans(ARABIC_TO_PERSIAN)
_PRESENTATION_FORMS_RE = re.compile("[ﭐ-﷿ﹰ-﻿]")


@dataclass(frozen=True)
class NormalizerConfig:
    """Which normalisation steps to apply.

    Attributes:
        presentation_forms: map Arabic presentation forms (common in PDF text)
            back to base letters.
        unify_chars: map Arabic ي/ى/ك/ۀ/ە to their Persian forms.
        digits: unify all digits to ``"en"`` (0-9), ``"fa"`` (۰-۹) or ``"keep"``.
            ``"en"`` is the default because most tokenizers and embedding
            models see ASCII digits far more often.
        remove_diacritics: drop harakat/tanwin/shadda/sukun.
        remove_tatweel: drop the decorative stretch character ``ـ``.
        fix_zwnj: clean up stray ZWNJs and insert them for common
            prefixes (می/نمی) and suffixes (ها، تر، ترین، ...).
        normalize_whitespace: collapse runs of spaces/tabs, tidy newlines, strip.
    """

    presentation_forms: bool = True
    unify_chars: bool = True
    digits: DigitTarget = "en"
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    fix_zwnj: bool = True
    normalize_whitespace: bool = True


# ----------------------------------------------------------------- single steps


def fix_presentation_forms(text: str) -> str:
    """Replace Arabic presentation forms (U+FB50–U+FDFF, U+FE70–U+FEFF) with base letters."""
    return _PRESENTATION_FORMS_RE.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)


def unify_characters(text: str) -> str:
    """Map Arabic letters to their Persian equivalents (ي→ی, ك→ک, ...)."""
    return text.translate(_CHAR_TABLE)


def unify_digits(text: str, target: DigitTarget = "en") -> str:
    """Convert Persian, Arabic-Indic and Latin digits to one system.

    ``target`` is ``"en"`` (0-9), ``"fa"`` (۰-۹) or ``"keep"`` (no change).
    """
    if target == "keep":
        return text
    if target == "en":
        table = str.maketrans(PERSIAN_DIGITS + ARABIC_DIGITS, LATIN_DIGITS * 2)
    elif target == "fa":
        table = str.maketrans(LATIN_DIGITS + ARABIC_DIGITS, PERSIAN_DIGITS * 2)
    else:
        raise ValueError(f"unknown digit target: {target!r} (expected 'en', 'fa' or 'keep')")
    return text.translate(table)


def remove_diacritics(text: str) -> str:
    """Remove short-vowel marks, tanwin, shadda and sukun; keep hamza and maddah."""
    return DIACRITICS_RE.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove the tatweel/kashida stretch character (``ـ``)."""
    return text.replace(TATWEEL, "")


def fix_zwnj(text: str) -> str:
    """Repair zero-width non-joiners.

    1. ZWNJ look-alikes (ZWJ, soft hyphen, zero-width space, BOM) become ZWNJ.
    2. Repeated ZWNJs collapse to one; ZWNJs next to spaces or at the start/end
       of the text are dropped.
    3. ``می``/``نمی`` written as separate words are joined to the verb with a ZWNJ.
    4. Plural/comparative suffixes (``ها``، ``های``، ``تر``، ``ترین``، ...) written
       as separate words are joined with a ZWNJ, as are ``ای``/``ام``/``اند``...
       after words ending in ``ه``.

    Ambiguous *attached* forms such as ``میرود`` (vs ``میز``) are left alone:
    fixing them needs a dictionary.
    """
    text = re.sub(f"[{ZWNJ_LOOKALIKES}]", ZWNJ, text)
    text = re.sub(f"{ZWNJ}{{2,}}", ZWNJ, text)
    text = re.sub(f"{ZWNJ}*(\\s){ZWNJ}*", r"\1", text)
    text = text.strip(ZWNJ)
    text = _PREFIX_RE.sub(rf"\1{ZWNJ}", text)
    text = _SUFFIX_RE.sub(rf"{ZWNJ}\1", text)
    text = _HEH_SUFFIX_RE.sub(rf"{ZWNJ}\1", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse spaces/tabs, remove spaces around newlines and before punctuation, strip."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t  -   　]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +([.,!?:;،؛؟»)])", r"\1", text)
    text = re.sub(r"([«(]) +", r"\1", text)
    return text.strip()


# ---------------------------------------------------------------- composition


def normalize(text: str, config: NormalizerConfig | None = None, **overrides: Any) -> str:
    """Normalise Persian ``text``.

    ``config`` selects the steps; keyword ``overrides`` tweak single fields,
    e.g. ``normalize(text, digits="fa", fix_zwnj=False)``.
    """
    cfg = config or NormalizerConfig()
    if overrides:
        cfg = replace(cfg, **overrides)
    # NFC first so decomposed sequences (e.g. ا + U+0653) become single letters (آ).
    text = unicodedata.normalize("NFC", text)
    if cfg.presentation_forms:
        text = fix_presentation_forms(text)
    if cfg.unify_chars:
        text = unify_characters(text)
    text = unify_digits(text, cfg.digits)
    if cfg.remove_diacritics:
        text = remove_diacritics(text)
    if cfg.remove_tatweel:
        text = remove_tatweel(text)
    if cfg.normalize_whitespace:
        text = normalize_whitespace(text)
    if cfg.fix_zwnj:
        text = fix_zwnj(text)
    return text


class Normalizer:
    """A reusable, configured normaliser: ``Normalizer(digits="fa")(text)``."""

    def __init__(self, config: NormalizerConfig | None = None, **overrides: Any) -> None:
        cfg = config or NormalizerConfig()
        self.config = replace(cfg, **overrides) if overrides else cfg

    def __call__(self, text: str) -> str:
        return normalize(text, self.config)


__all__ = [
    "ZWNJ",
    "Normalizer",
    "NormalizerConfig",
    "fix_presentation_forms",
    "fix_zwnj",
    "normalize",
    "normalize_whitespace",
    "remove_diacritics",
    "remove_tatweel",
    "unify_characters",
    "unify_digits",
]

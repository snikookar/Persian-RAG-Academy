"""Lesson 02 starter — Persian text normalization.

Fill in every function marked TODO, then check your work:

    python -m grader run 02

The constants and the `normalize()` pipeline are already written for you;
your job is the individual steps. Reference solution: src/parsrag/normalize.py
(try not to peek until you have tried!).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Literal

ZWNJ = "‌"

DigitTarget = Literal["en", "fa", "keep"]

ARABIC_TO_PERSIAN: dict[str, str] = {
    "ي": "ی",  # ي ARABIC YEH       -> ی FARSI YEH
    "ى": "ی",  # ى ALEF MAKSURA     -> ی
    "ك": "ک",  # ك ARABIC KAF       -> ک KEHEH
    "ۀ": "هٔ",  # ۀ            -> هٔ
    "ە": "ه",  # ە AE               -> ه
}

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"

TATWEEL = "ـ"
ZWNJ_LOOKALIKES = "‍­​﻿"  # ZWJ, soft hyphen, zero-width space, BOM

# A character class matching one Persian letter (useful in your regexes).
_L = "[ء-غف-يپچژکگیٔ]"


@dataclass(frozen=True)
class NormalizerConfig:
    presentation_forms: bool = True
    unify_chars: bool = True
    digits: DigitTarget = "en"
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    fix_zwnj: bool = True
    normalize_whitespace: bool = True


def fix_presentation_forms(text: str) -> str:
    """Replace Arabic presentation forms (U+FB50–U+FDFF, U+FE70–U+FEFF) with base letters.

    Hint: unicodedata.normalize("NFKC", ch) turns 'ﻛ' into 'ك'. Apply it only to
    characters in those two ranges (a regex with re.sub and a function works well).
    """
    # TODO: implement
    raise NotImplementedError("TODO: fix_presentation_forms")


def unify_characters(text: str) -> str:
    """Map Arabic letters to Persian ones using ARABIC_TO_PERSIAN (ي→ی, ك→ک, ...)."""
    # TODO: implement (str.maketrans + str.translate)
    raise NotImplementedError("TODO: unify_characters")


def unify_digits(text: str, target: DigitTarget = "en") -> str:
    """Convert Persian, Arabic-Indic and Latin digits to one system.

    target: "en" -> 0-9, "fa" -> ۰-۹, "keep" -> unchanged. Anything else: ValueError.
    """
    # TODO: implement
    raise NotImplementedError("TODO: unify_digits")


def remove_diacritics(text: str) -> str:
    """Remove U+064B..U+0652 (harakat, tanwin, shadda, sukun) and U+0670 (superscript alef).

    Keep hamza (U+0654) and maddah (U+0653): they are part of Persian spelling.
    """
    # TODO: implement
    raise NotImplementedError("TODO: remove_diacritics")


def remove_tatweel(text: str) -> str:
    """Remove the tatweel/kashida character (TATWEEL)."""
    # TODO: implement
    raise NotImplementedError("TODO: remove_tatweel")


def fix_zwnj(text: str) -> str:
    """Repair zero-width non-joiners (ZWNJ).

    Steps:
      1. Replace every character in ZWNJ_LOOKALIKES with ZWNJ.
      2. Collapse repeated ZWNJs; drop ZWNJs next to whitespace and at the start/end.
      3. "می رود" / "نمی دانم" -> join می/نمی to the next word with ZWNJ
         (only when می/نمی is a separate word: "میز" and "همی" stay as they are).
      4. "کتاب ها", "بزرگ ترین", "سفارش هایشان" -> join the suffix with ZWNJ.
         Suffixes: ها های هایی هایم هایت هایش هایمان هایتان هایشان تر تری ترین.
         The previous word needs 2+ letters, so "خشک و تر" stays unchanged.
      5. After a word ending in ه: "خانه ای" -> "خانه‌ای" (ای ام ایم اید اند).

    Hint: regex lookbehind (?<=...) and lookahead (?=...) with _L are your friends.
    """
    # TODO: implement
    raise NotImplementedError("TODO: fix_zwnj")


def normalize_whitespace(text: str) -> str:
    """Tidy whitespace.

    - collapse runs of spaces/tabs (and other horizontal spaces) into one space
    - remove spaces around newlines
    - remove spaces before punctuation . , ! ? : ; ، ؛ ؟ » )  and after « (
    - strip the ends
    """
    # TODO: implement
    raise NotImplementedError("TODO: normalize_whitespace")


# ---------------------------------------------------------------------------
# Already done for you: the pipeline that combines the steps.


def normalize(text: str, config: NormalizerConfig | None = None, **overrides: Any) -> str:
    cfg = config or NormalizerConfig()
    if overrides:
        cfg = replace(cfg, **overrides)
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
    def __init__(self, config: NormalizerConfig | None = None, **overrides: Any) -> None:
        cfg = config or NormalizerConfig()
        self.config = replace(cfg, **overrides) if overrides else cfg

    def __call__(self, text: str) -> str:
        return normalize(text, self.config)


# `re` is imported for you — you will need it.
_ = re

"""Library-level normalize tests (lesson 02's tests cover the learner-facing API)."""

from __future__ import annotations

import pytest

from parsrag.normalize import ZWNJ, Normalizer, NormalizerConfig, normalize


def test_config_is_frozen() -> None:
    cfg = NormalizerConfig()
    with pytest.raises(AttributeError):
        cfg.digits = "fa"  # type: ignore[misc]


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(TypeError):
        normalize("متن", not_a_step=True)


def test_everything_off_only_applies_nfc() -> None:
    cfg = NormalizerConfig(
        presentation_forms=False,
        unify_chars=False,
        digits="keep",
        remove_diacritics=False,
        remove_tatweel=False,
        fix_zwnj=False,
        normalize_whitespace=False,
    )
    messy = "كتاب  ٣ مُـحمد ﻛ"
    assert normalize(messy, cfg) == messy
    assert normalize("آ", cfg) == "آ"  # NFC composes alef + maddah


def test_normalizer_reuses_config() -> None:
    n = Normalizer(NormalizerConfig(digits="fa"), fix_zwnj=False)
    assert n.config.digits == "fa" and n.config.fix_zwnj is False
    assert n("کتاب ها ٣") == "کتاب ها ۳"


def test_paragraphs_are_preserved() -> None:
    assert normalize("خط  اول\n\n\n\nخط دوم") == "خط اول\n\nخط دوم"


def test_full_sentence() -> None:
    raw = "ما  مي خواهيم كتاب هاي بزرگ ترين كتابخانه را ببينيم ."
    expected = f"ما می{ZWNJ}خواهیم کتاب{ZWNJ}های بزرگ{ZWNJ}ترین کتابخانه را ببینیم."
    assert normalize(raw) == expected

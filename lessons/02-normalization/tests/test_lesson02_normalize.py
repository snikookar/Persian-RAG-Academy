"""Lesson 02 — Persian normalization. Run with: python -m grader run 02"""

from __future__ import annotations

from types import ModuleType

import pytest

from grader.paths import CORPUS, QA_FILE
from parsrag import FakeLLM, HashingEmbedding
from parsrag.eval import hit_rate, load_qa
from parsrag.naive_rag import NaiveRAG, load_documents

Z = "‌"
hint = pytest.mark.hint


# ----------------------------------------------------------------- characters


@hint(
    fa="از str.maketrans و text.translate استفاده کنید؛ جدول ARABIC_TO_PERSIAN آماده است.",
    en="Use str.maketrans + text.translate with the ARABIC_TO_PERSIAN table.",
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("كتاب", "کتاب"),  # Arabic kaf
        ("علي", "علی"),  # Arabic yeh
        ("موسى", "موسی"),  # alef maksura
        ("كيك", "کیک"),  # mixed
        ("خانۀ", "خانهٔ"),  # heh with yeh above -> heh + hamza above
        ("کتاب", "کتاب"),  # already Persian: unchanged
        ("hello 123", "hello 123"),  # non-Persian untouched
    ],
)
def test_unify_characters(nz: ModuleType, raw: str, expected: str) -> None:
    """unify_characters maps Arabic letters to Persian ones"""
    assert nz.unify_characters(raw) == expected


# --------------------------------------------------------------------- digits


@hint(
    fa="برای هر هدف ('en' یا 'fa') یک جدول ترجمه بسازید که هر دو دستگاه دیگر را تبدیل کند.",
    en="Build one translate table per target that maps BOTH other digit systems.",
)
@pytest.mark.parametrize(
    ("raw", "target", "expected"),
    [
        ("۱۴۰۳", "en", "1403"),  # Persian -> Latin
        ("٤٥٦", "en", "456"),  # Arabic-Indic -> Latin
        ("۱٢3", "en", "123"),  # mixed -> Latin
        ("2024", "fa", "۲۰۲۴"),  # Latin -> Persian
        ("٤٥٦", "fa", "۴۵۶"),  # Arabic-Indic -> Persian (they look alike but differ!)
        ("۱٢3", "keep", "۱٢3"),  # keep = untouched
        ("بدون عدد", "en", "بدون عدد"),
    ],
)
def test_unify_digits(nz: ModuleType, raw: str, target: str, expected: str) -> None:
    """unify_digits converts every digit system to the chosen one"""
    assert nz.unify_digits(raw, target) == expected


@hint(fa="برای هدف ناشناخته ValueError بدهید.", en="Raise ValueError for an unknown target.")
def test_unify_digits_rejects_unknown_target(nz: ModuleType) -> None:
    """unify_digits rejects an unknown target"""
    with pytest.raises(ValueError):
        nz.unify_digits("۱۲۳", "roman")


# ------------------------------------------------------- diacritics & tatweel


@hint(
    fa="فقط بازهٔ U+064B تا U+0652 و U+0670 را حذف کنید؛ همزه و مد (آ) را نگه دارید.",
    en="Remove only U+064B–U+0652 and U+0670; keep hamza and maddah.",
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("مُحَمَّد", "محمد"),  # damma, fatha, shadda
        ("کِتاب", "کتاب"),  # kasra
        ("حتماً", "حتما"),  # tanwin
        ("خانهٔ", "خانهٔ"),  # hamza above is kept
        ("آب", "آب"),  # alef with madda is kept
        ("رحمٰن", "رحمن"),  # superscript alef
    ],
)
def test_remove_diacritics(nz: ModuleType, raw: str, expected: str) -> None:
    """remove_diacritics drops harakat but keeps hamza and maddah"""
    assert nz.remove_diacritics(raw) == expected


@hint(fa="کاراکتر کشیده (tatweel) همان U+0640 است.", en="Tatweel is U+0640.")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("کـــتاب", "کتاب"), ("سـلام", "سلام"), ("بدون", "بدون")],
)
def test_remove_tatweel(nz: ModuleType, raw: str, expected: str) -> None:
    """remove_tatweel removes the stretch character"""
    assert nz.remove_tatweel(raw) == expected


@hint(
    fa="unicodedata.normalize('NFKC', ...) را فقط روی بازه‌های presentation forms اجرا کنید.",
    en="Apply unicodedata NFKC only to characters in the presentation-form ranges.",
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("ﺭﺍﻳﮕﺎﻥ", "رايگان"), ("ﺳﻼﻡ", "سلام"), ("سلام", "سلام")],
)
def test_fix_presentation_forms(nz: ModuleType, raw: str, expected: str) -> None:
    """fix_presentation_forms turns PDF-style glyph forms into normal letters"""
    assert nz.fix_presentation_forms(raw) == expected


# ----------------------------------------------------------------------- ZWNJ


@hint(
    fa="«می» و «نمی» اگر جدا نوشته شده‌اند با نیم‌فاصله به فعل بچسبند؛ «میز» نباید تغییر کند.",
    en="Join separated می/نمی to the verb with ZWNJ; words like میز must not change.",
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("می رود", f"می{Z}رود"),
        ("نمی دانم", f"نمی{Z}دانم"),
        ("او می خواهد برود", f"او می{Z}خواهد برود"),
        (f"می{Z}رود", f"می{Z}رود"),  # already correct
        ("میز بزرگ", "میز بزرگ"),  # not a prefix
        ("همی رود", "همی رود"),  # "می" inside a word is not a prefix
    ],
)
def test_fix_zwnj_prefixes(nz: ModuleType, raw: str, expected: str) -> None:
    """fix_zwnj joins the verbal prefixes می/نمی"""
    assert nz.fix_zwnj(raw) == expected


@hint(
    fa="پسوندهای ها/های/تر/ترین اگر با فاصله جدا شده‌اند، با نیم‌فاصله بچسبند.",
    en="Attach suffixes like ها/های/تر/ترین that were separated by a space.",
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("کتاب ها", f"کتاب{Z}ها"),
        ("کتاب های من", f"کتاب{Z}های من"),
        ("بزرگ تر", f"بزرگ{Z}تر"),
        ("بزرگ ترین شهر", f"بزرگ{Z}ترین شهر"),
        ("سفارش هایشان", f"سفارش{Z}هایشان"),
        ("خانه ای", f"خانه{Z}ای"),
        ("رفته اند", f"رفته{Z}اند"),
        ("خشک و تر", "خشک و تر"),  # «تر» = wet, not a suffix
        ("تهران", "تهران"),
    ],
)
def test_fix_zwnj_suffixes(nz: ModuleType, raw: str, expected: str) -> None:
    """fix_zwnj attaches plural and comparative suffixes"""
    assert nz.fix_zwnj(raw) == expected


@hint(
    fa="نیم‌فاصله‌های تکراری را یکی کنید و نیم‌فاصلهٔ کنار فاصله یا اول/آخر متن را حذف کنید.",
    en="Collapse repeated ZWNJs; drop ZWNJs next to spaces or at text edges.",
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (f"کتاب{Z}{Z}ها", f"کتاب{Z}ها"),
        (f"{Z}سلام", "سلام"),
        (f"سلام{Z}", "سلام"),
        (f"سلام{Z} دنیا", "سلام دنیا"),
        ("کتاب‍ها", f"کتاب{Z}ها"),  # ZWJ used by mistake
        ("کتاب­ها", f"کتاب{Z}ها"),  # soft hyphen used by mistake
    ],
)
def test_fix_zwnj_cleanup(nz: ModuleType, raw: str, expected: str) -> None:
    """fix_zwnj cleans up stray and look-alike ZWNJs"""
    assert nz.fix_zwnj(raw) == expected


# ----------------------------------------------------------------- whitespace


@hint(
    fa="با re.sub فاصله‌های پشت‌سرهم را یکی کنید و فاصلهٔ قبل از علائم نگارشی را بردارید.",
    en="Use re.sub to collapse spaces and remove spaces before punctuation.",
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("سلام    دنیا", "سلام دنیا"),
        ("  سلام  ", "سلام"),
        ("سلام\t\tدنیا", "سلام دنیا"),
        ("خوبی ؟", "خوبی؟"),
        ("بله ، حتما .", "بله، حتما."),
        ("خط اول   \n   خط دوم", "خط اول\nخط دوم"),
        ("« سلام »", "«سلام»"),
    ],
)
def test_normalize_whitespace(nz: ModuleType, raw: str, expected: str) -> None:
    """normalize_whitespace tidies spaces"""
    assert nz.normalize_whitespace(raw) == expected


# ---------------------------------------------------------------- normalize()


@hint(
    fa="normalize باید همهٔ مراحل را به ترتیب و بر اساس NormalizerConfig اجرا کند.",
    en="normalize() must run every enabled step, in order, according to NormalizerConfig.",
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("كتاب هاي  من", f"کتاب{Z}های من"),
        ("قيمت: ٣٥ هزار تومان .", "قیمت: 35 هزار تومان."),
        ("مي خواهم بزرگ ترين را بخرم", f"می{Z}خواهم بزرگ{Z}ترین را بخرم"),
        ("گارانـــتی ۱۸ ماهه", "گارانتی 18 ماهه"),
        ("ﺭﺍﻳﮕﺎﻥ", "رایگان"),
        ("کِتاب‌ها", f"کتاب{Z}ها"),
        ("", ""),
    ],
)
def test_normalize_default(nz: ModuleType, raw: str, expected: str) -> None:
    """normalize() applies all steps with the default config"""
    assert nz.normalize(raw) == expected


@hint(
    fa="با dataclasses.replace(config, **overrides) تنظیمات را جایگزین کنید.",
    en="Apply keyword overrides with dataclasses.replace(config, **overrides).",
)
def test_normalize_is_configurable(nz: ModuleType) -> None:
    """normalize() respects NormalizerConfig and keyword overrides"""
    text = "كتاب ۱۲"
    assert nz.normalize(text, digits="fa") == "کتاب ۱۲"
    assert nz.normalize(text, digits="keep") == "کتاب ۱۲"
    assert nz.normalize(text, unify_chars=False) == "كتاب 12"
    config = nz.NormalizerConfig(digits="fa", fix_zwnj=False)
    assert nz.normalize("کتاب ها ٣", config) == "کتاب ها ۳"
    assert nz.Normalizer(digits="fa")("٣") == "۳"


@hint(
    fa="نرمال‌سازی دوباره نباید چیزی را تغییر دهد (idempotent).",
    en="Normalizing twice must give the same result (idempotence).",
)
@pytest.mark.parametrize("path", sorted(CORPUS.glob("*.md")), ids=lambda p: p.stem)
def test_normalize_is_idempotent_on_corpus(nz: ModuleType, path) -> None:
    """normalize() is idempotent on every fixture document"""
    once = nz.normalize(path.read_text(encoding="utf-8"))
    assert nz.normalize(once) == once
    assert "ي" not in once and "ك" not in once
    assert not any(ch in once for ch in "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")


# ------------------------------------------------------------------ the payoff


@hint(
    fa="اگر بقیهٔ تست‌ها سبز باشند این تست هم پاس می‌شود: نرمال‌سازی باید بازیابی را بهتر کند.",
    en="Once the other checks pass this one should too: normalization must improve retrieval.",
)
def test_normalization_improves_retrieval(nz: ModuleType) -> None:
    """normalization measurably improves retrieval hit-rate on the fixtures"""
    docs = load_documents(CORPUS)
    qa = load_qa(QA_FILE)

    def score(preprocess: object) -> float:
        rag = NaiveRAG(HashingEmbedding(), FakeLLM(), chunk_size=300, preprocess=preprocess)  # type: ignore[arg-type]
        rag.index(docs)
        return hit_rate(qa, rag.retrieve, k=1)

    raw, normalized = score(None), score(nz.normalize)
    assert normalized > raw, f"hit@1 raw={raw:.3f} normalized={normalized:.3f}"
    assert normalized >= 0.85

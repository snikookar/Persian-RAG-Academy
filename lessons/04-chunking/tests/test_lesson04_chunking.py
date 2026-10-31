"""Lesson 04 — chunking Persian text. Run with: python -m grader run 04"""

from __future__ import annotations

from itertools import pairwise
from types import ModuleType

import pytest

from grader.paths import CORPUS, QA_FILE
from parsrag import HashingEmbedding, InMemoryVectorStore
from parsrag.eval import evaluate_retrieval, load_qa
from parsrag.loaders import load_directory, load_markdown
from parsrag.normalize import normalize
from parsrag.types import Document

hint = pytest.mark.hint


def _texts(spans: list) -> list[str]:  # type: ignore[type-arg]
    return [s.text for s in spans]


# ------------------------------------------------------------------ sentences


@hint(
    fa="نقطه فقط وقتی جمله را تمام می‌کند که بعدش فاصله یا پایان متن بیاید و قبلش اختصار، حرف تنها یا شمارهٔ فهرست نباشد.",
    en="A dot ends a sentence only before whitespace/end, and not after an abbreviation, a single letter or a list number.",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("سلام. خوبی؟ بله!", ["سلام.", "خوبی؟", "بله!"]),
        ("قیمت 2.5 میلیون است. تمام.", ["قیمت 2.5 میلیون است.", "تمام."]),
        ("بودجه ۴۲٫۵ میلیارد بود؛ کافی نبود.", ["بودجه ۴۲٫۵ میلیارد بود؛", "کافی نبود."]),
        ("نشانی www.kahroba.ir است. بیا", ["نشانی www.kahroba.ir است.", "بیا"]),
        ("ساعت ۱۰ ق.ظ. شروع شد.", ["ساعت ۱۰ ق.ظ. شروع شد."]),
        ("دکتر م. رضایی آمد. رفت.", ["دکتر م. رضایی آمد.", "رفت."]),
        ("«آیا می‌آیی؟» گفت.", ["«آیا می‌آیی؟»", "گفت."]),
        ("صبر کن... باشه… خب", ["صبر کن...", "باشه…", "خب"]),
        ("واقعاً؟! آره", ["واقعاً؟!", "آره"]),
        ("1. اول\n2. دوم", ["1. اول", "2. دوم"]),
        ("خط اول\n\nخط دوم", ["خط اول", "خط دوم"]),
        ("", []),
    ],
)
def test_split_sentences(ck: ModuleType, text: str, expected: list[str]) -> None:
    """split_sentences handles Persian punctuation, decimals and abbreviations"""
    spans = ck.split_sentences(text)
    assert _texts(spans) == expected
    assert all(text[s.start : s.end] == s.text for s in spans), "offsets must point into the text"


@hint(
    fa="آخرین توکن پیش از نقطه را با _LAST_TOKEN_RE پیدا کنید.",
    en="Find the last token before the dot with _LAST_TOKEN_RE.",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ساعت ۵ ب.ظ.", True),
        ("آقای م.", True),
        ("Dr.", True),
        ("1.", True),
        ("سال 1.", False),
        ("کتاب.", False),
    ],
)
def test_is_abbreviation(ck: ModuleType, text: str, expected: bool) -> None:
    """is_abbreviation recognises abbreviations, initials and list numbers"""
    assert ck.is_abbreviation(text, len(text) - 1) is expected


@hint(
    fa="برای هر قطعهٔ _PRETOKEN_RE: max(1, ceil(بایت‌های UTF-8 / bytes_per_token)).",
    en="For each _PRETOKEN_RE piece: max(1, ceil(utf8 bytes / bytes_per_token)).",
)
def test_estimate_tokens(ck: ModuleType) -> None:
    """estimate_tokens counts UTF-8 bytes per word; Persian costs more than English"""
    assert ck.estimate_tokens("") == 0
    assert ck.estimate_tokens("a b c") == 3
    assert ck.estimate_tokens("abcdefgh") == 2
    assert ck.estimate_tokens("کتاب") == 2  # 8 bytes
    en = "Members can borrow up to three books at the same time."
    fa = "اعضا می‌توانند هم‌زمان حداکثر سه کتاب امانت بگیرند."
    assert ck.estimate_tokens(fa) > ck.estimate_tokens(en)
    with pytest.raises(ValueError):
        ck.estimate_tokens("x", bytes_per_token=0)


# -------------------------------------------------------------------- chunkers


@hint(
    fa="بعد از هر پنجره، شروع بعدی = end - overlap (با _snap)؛ اگر جلو نرفت، از end ادامه دهید.",
    en="After each window the next start is end - overlap (snapped); if it does not advance, use end.",
)
def test_fixed_size_chunker(ck: ModuleType) -> None:
    """FixedSizeChunker makes overlapping windows that never cut a word"""
    text = " ".join(f"کلمه{i}" for i in range(150))
    spans = ck.FixedSizeChunker(100, 20).split(text)
    assert all(len(s.text) <= 100 for s in spans)
    assert all(text[s.start : s.end] == s.text for s in spans)
    words = set(text.split())
    assert all(set(s.text.split()) <= words for s in spans), "a word was cut in half"
    assert all(b.start < a.end for a, b in pairwise(spans)), "no overlap"
    assert spans[-1].end == len(text)
    raw = ck.FixedSizeChunker(10, 0, snap_to_words=False).split("abcdefghijklmnopqrstuvwxy")
    assert _texts(raw) == ["abcdefghij", "klmnopqrst", "uvwxy"]
    assert ck.FixedSizeChunker(10, 2).split("   ") == []


@hint(
    fa="j را تا جایی که طول source[units[i].start:units[j+1].end] <= size است جلو ببرید؛ هم‌پوشانی فقط از واحدهای کامل.",
    en="Grow j while source[units[i].start:units[j+1].end] fits; overlap reuses whole units only.",
)
def test_pack(ck: ModuleType) -> None:
    """pack joins whole units up to size and overlaps by whole units"""
    src = "aa bb cc dd ee"
    units = [ck.Span(w, src.index(w), src.index(w) + 2) for w in src.split()]
    assert _texts(ck.pack(src, units, 8, 2)) == ["aa bb cc", "cc dd ee"]
    assert _texts(ck.pack(src, units, 5, 0)) == ["aa bb", "cc dd", "ee"]
    assert _texts(ck.pack(src, units, 100, 0)) == ["aa bb cc dd ee"]


@hint(
    fa="اگر تکه جا شد همان را برگردانید؛ وگرنه با جداکنندهٔ همین سطح بشکنید و هر بخش را با سطح بعدی ادامه دهید.",
    en="Return the span if it fits; otherwise split with this level's splitter and recurse with level + 1.",
)
def test_recursive_chunker_prefers_structure(ck: ModuleType) -> None:
    """RecursiveChunker keeps sections and sentences together when they fit"""
    text = "# عنوان\n\n## بخش یک\nجملهٔ اول. جملهٔ دوم.\n\n## بخش دو\nجملهٔ سوم."
    assert _texts(ck.RecursiveChunker(200, 0).split(text)) == [
        "# عنوان\n\n## بخش یک\nجملهٔ اول. جملهٔ دوم.",
        "## بخش دو\nجملهٔ سوم.",
    ]
    assert len(ck.RecursiveChunker(200, 0, respect_sections=False).split(text)) == 1
    small = ck.RecursiveChunker(20, 0).split(text)
    assert all(len(s.text) <= 20 for s in small)
    long_word = ck.RecursiveChunker(5, 0).split("الف" * 10)
    assert "".join(_texts(long_word)) == "الف" * 10 and all(len(s.text) <= 5 for s in long_word)


def test_recursive_chunker_on_a_long_document(ck: ModuleType) -> None:
    """RecursiveChunker cuts a long document within size, at sentence ends"""
    doc = load_markdown(CORPUS / "kahroba-privacy.md")
    spans = ck.RecursiveChunker(400, 60).split(doc.text)
    assert all(len(s.text) <= 400 and doc.text[s.start : s.end] == s.text for s in spans)
    ends = sum(s.text.rstrip().endswith((".", "؟", "!", "؛")) or "\n" not in s.text for s in spans)
    assert ends / len(spans) > 0.9


@hint(
    fa="برای جملهٔ i از جملهٔ i-window تا i+window (محدود به ابتدا و انتها) را بردارید.",
    en="For sentence i take sentences i-window .. i+window, clamped at both ends.",
)
def test_sentence_windows(ck: ModuleType) -> None:
    """SentenceWindowChunker returns each sentence with its neighbours"""
    text = "یک. دو. سه. چهار."
    assert _texts(ck.SentenceWindowChunker(1).windows(text)) == [
        "یک. دو.", "یک. دو. سه.", "دو. سه. چهار.", "سه. چهار.",
    ]  # fmt: skip
    assert _texts(ck.SentenceWindowChunker(0).windows(text)) == ["یک.", "دو.", "سه.", "چهار."]


# ------------------------------------------------------------ chunk_document


@hint(
    fa="برای هر span: locate(doc, span.start) و متادیتای start/end/index/section/section_path/page.",
    en="For each span: locate(doc, span.start) and metadata start/end/index/section/section_path/page.",
)
def test_chunk_document_metadata(ck: ModuleType) -> None:
    """chunk_document records where every chunk came from"""
    doc = load_markdown(CORPUS / "kahroba-seller-guide.md")
    chunks = ck.chunk_document(doc, ck.RecursiveChunker(400, 50), preprocess=normalize)
    assert chunks[0].id == "kahroba-seller-guide#0"
    by_source = {doc.text[c.metadata["start"] : c.metadata["end"]]: c for c in chunks}
    source, fee = next((s, c) for s, c in by_source.items() if "کارمزد صنایع دستی" in s)
    assert fee.metadata["section"] == "کارمزد فروش"
    assert fee.metadata["section_path"][0] == "راهنمای فروشندگان بازارچهٔ کهربا"
    assert fee.text == normalize(source)
    assert [c.metadata["index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_document_heading_context_and_windows(ck: ModuleType) -> None:
    """chunk_document adds heading context and sentence windows"""
    doc = load_markdown(CORPUS / "kahroba-seller-guide.md")
    with_ctx = ck.chunk_document(doc, ck.RecursiveChunker(400, 50), heading_context=True)
    fee = next(c for c in with_ctx if "کارمزد صنایع دستی" in c.text)
    assert fee.text.startswith("راهنمای فروشندگان بازارچهٔ کهربا › کارمزد فروش\n")
    pages = Document(
        id="d",
        text="یک. دو.\n\nسه.",
        metadata={"pages": [{"page": 1, "start": 0, "end": 7}, {"page": 2, "start": 9, "end": 12}]},
    )
    chunks = ck.chunk_document(pages, ck.SentenceWindowChunker(1))
    assert [c.text for c in chunks] == ["یک.", "دو.", "سه."]
    assert chunks[2].metadata["window"] == "دو.\n\nسه."
    assert [c.metadata["page"] for c in chunks] == [1, 1, 2]


# ------------------------------------------------------------------ the payoff


def _passage_mrr(ck: ModuleType, chunker: object, heading_context: bool = False) -> float:
    store = InMemoryVectorStore(HashingEmbedding())
    for doc in load_directory(CORPUS, "*.md"):
        store.add(
            ck.chunk_document(doc, chunker, preprocess=normalize, heading_context=heading_context)
        )
    report = evaluate_retrieval(
        load_qa(QA_FILE), lambda q, k: store.search(normalize(q), k), ks=(3,), level="passage"
    )
    return report.mrr


def test_structure_aware_chunking_beats_fixed_windows(ck: ModuleType) -> None:
    """recursive chunking with heading context finds the answer passage more often"""
    fixed = _passage_mrr(ck, ck.FixedSizeChunker(200, 25))
    best = _passage_mrr(ck, ck.RecursiveChunker(200, 25), heading_context=True)
    assert best > fixed + 0.1, f"passage MRR fixed={fixed:.3f} recursive+headings={best:.3f}"

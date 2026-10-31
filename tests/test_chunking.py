from __future__ import annotations

from itertools import pairwise

import pytest

from grader.paths import CORPUS
from parsrag.chunking import (
    Chunker,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    Span,
    chunk_document,
    estimate_tokens,
    pack,
    split_headings,
    split_sentences,
)
from parsrag.loaders import load_markdown
from parsrag.normalize import normalize
from parsrag.types import Document


def _texts(spans: list[Span]) -> list[str]:
    return [s.text for s in spans]


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
        ("صبر کن... باشه", ["صبر کن...", "باشه"]),
        ("واقعاً؟! آره", ["واقعاً؟!", "آره"]),
        ("1. اول\n2. دوم", ["1. اول", "2. دوم"]),
        ("خط اول\nخط دوم", ["خط اول", "خط دوم"]),
        ("Dr. Smith is here. Yes.", ["Dr. Smith is here.", "Yes."]),
        ("  \n ", []),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    spans = split_sentences(text)
    assert _texts(spans) == expected
    assert all(text[s.start : s.end] == s.text for s in spans)


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a b c") == 3
    en = "Members can borrow up to three books at the same time."
    fa = "اعضا می‌توانند هم‌زمان حداکثر سه کتاب امانت بگیرند."
    assert estimate_tokens(fa) > estimate_tokens(en)
    assert estimate_tokens(fa, bytes_per_token=1000) == len(fa.split()) + 1  # words + "."
    with pytest.raises(ValueError):
        estimate_tokens("x", bytes_per_token=0)


@pytest.mark.parametrize(
    "chunker",
    [FixedSizeChunker(300, 40), RecursiveChunker(300, 60), SentenceWindowChunker(1)],
    ids=["fixed", "recursive", "window"],
)
def test_spans_point_into_the_source(chunker: Chunker) -> None:
    doc = load_markdown(CORPUS / "mehregan-annual-report.md")
    spans = chunker.split(doc.text)
    assert spans
    assert all(doc.text[s.start : s.end] == s.text and s.text == s.text.strip() for s in spans)
    assert [s.start for s in spans] == sorted(s.start for s in spans)


def test_fixed_size_respects_size_overlap_and_words() -> None:
    text = " ".join(f"کلمه{i}" for i in range(200))
    spans = FixedSizeChunker(100, 20).split(text)
    assert all(len(s) <= 100 for s in spans)
    words = set(text.split())
    assert all(set(s.text.split()) <= words for s in spans)  # no word cut in half
    assert all(b.start < a.end for a, b in pairwise(spans))  # overlap
    assert spans[-1].end == len(text)
    raw = FixedSizeChunker(10, 0, snap_to_words=False).split("abcdefghijklmnopqrstuvwxy")
    assert _texts(raw) == ["abcdefghij", "klmnopqrst", "uvwxy"]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_chunkers_reject_bad_sizes(size: int, overlap: int) -> None:
    for cls in (FixedSizeChunker, RecursiveChunker):
        with pytest.raises(ValueError):
            cls(size, overlap)
    with pytest.raises(ValueError):
        SentenceWindowChunker(-1)


def test_pack_overlaps_whole_units() -> None:
    src = "aa bb cc dd ee"
    units = [Span(w, src.index(w), src.index(w) + 2) for w in src.split()]
    out = pack(src, units, size=8, overlap=2)
    assert _texts(out) == ["aa bb cc", "cc dd ee"]
    assert _texts(pack(src, units, size=5, overlap=0)) == ["aa bb", "cc dd", "ee"]


def test_recursive_prefers_structure() -> None:
    text = "# عنوان\n\n## بخش یک\nجملهٔ اول. جملهٔ دوم.\n\n## بخش دو\nجملهٔ سوم."
    spans = RecursiveChunker(200, 0).split(text)
    assert _texts(spans) == [
        "# عنوان\n\n## بخش یک\nجملهٔ اول. جملهٔ دوم.",
        "## بخش دو\nجملهٔ سوم.",
    ]
    merged = RecursiveChunker(200, 0, respect_sections=False).split(text)
    assert len(merged) == 1
    small = RecursiveChunker(20, 0).split(text)
    assert "جملهٔ اول." in _texts(small) or any(t.endswith("جملهٔ اول.") for t in _texts(small))
    assert all(len(s) <= 20 for s in small)


def test_recursive_hard_cuts_long_words() -> None:
    spans = RecursiveChunker(5, 0).split("الف" * 10)
    assert all(len(s) <= 5 for s in spans) and "".join(_texts(spans)) == "الف" * 10


def test_split_headings_attaches_empty_heading() -> None:
    text = "# A\n## B\nbody\n## C\nmore"
    parts = split_headings(text, Span(text, 0, len(text)))
    assert _texts(parts) == ["# A\n## B\nbody", "## C\nmore"]


def test_recursive_can_count_tokens() -> None:
    doc = load_markdown(CORPUS / "kahroba-privacy.md")
    spans = RecursiveChunker(80, 10, length=estimate_tokens).split(doc.text)
    assert all(estimate_tokens(s.text) <= 80 for s in spans)


def test_sentence_windows() -> None:
    text = "یک. دو. سه. چهار."
    chunker = SentenceWindowChunker(1)
    assert _texts(chunker.windows(text)) == ["یک. دو.", "یک. دو. سه.", "دو. سه. چهار.", "سه. چهار."]
    assert _texts(SentenceWindowChunker(0).windows(text)) == ["یک.", "دو.", "سه.", "چهار."]


def test_chunk_document_metadata() -> None:
    doc = load_markdown(CORPUS / "kahroba-seller-guide.md")
    chunks = chunk_document(doc, RecursiveChunker(400, 50), preprocess=normalize)
    assert [c.id for c in chunks[:2]] == ["kahroba-seller-guide#0", "kahroba-seller-guide#1"]
    fee = next(
        c
        for c in chunks
        if "کارمزد صنایع دستی" in doc.text[c.metadata["start"] : c.metadata["end"]]
    )
    assert fee.metadata["section"] == "کارمزد فروش"
    assert fee.metadata["section_path"] == ["راهنمای فروشندگان بازارچهٔ کهربا", "کارمزد فروش"]
    assert fee.text == normalize(doc.text[fee.metadata["start"] : fee.metadata["end"]])
    ctx = chunk_document(doc, RecursiveChunker(400, 50), heading_context=True)
    assert ctx[fee.metadata["index"]].text.startswith(
        "راهنمای فروشندگان بازارچهٔ کهربا › کارمزد فروش\n"
    )


def test_chunk_document_windows_and_pages() -> None:
    doc = Document(
        id="d",
        text="یک. دو.\n\nسه.",
        metadata={"pages": [{"page": 1, "start": 0, "end": 7}, {"page": 2, "start": 9, "end": 12}]},
    )
    chunks = chunk_document(doc, SentenceWindowChunker(1))
    assert [c.text for c in chunks] == ["یک.", "دو.", "سه."]
    assert chunks[1].metadata["window"] == "یک. دو.\n\nسه."
    assert [c.metadata["page"] for c in chunks] == [1, 1, 2]


def _example(name: str):  # type: ignore[no-untyped-def]
    import importlib.util
    import sys

    from grader.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "examples" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need their module registered
    spec.loader.exec_module(module)
    return module


def test_numbers_quoted_in_lesson04() -> None:
    """The benchmark rows quoted in lessons/04-chunking/README.fa.md."""
    bench = _example("chunking_benchmark")
    docs = bench.load_directory(CORPUS, "*.md")
    n, _, fixed = bench.run(docs, FixedSizeChunker(200, 25))
    assert n == 175
    assert (round(fixed["P-R@3"], 3), round(fixed["P-MRR"], 3)) == (0.780, 0.670)
    n, _, best = bench.run(docs, RecursiveChunker(400, 50), heading_context=True)
    assert n == 109
    assert (round(best["P-R@3"], 3), round(best["P-MRR"], 3)) == (0.983, 0.858)

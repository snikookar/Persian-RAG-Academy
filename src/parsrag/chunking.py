"""Chunking Persian text (lesson 04).

Every chunker works on character **spans** of the original text, so each
chunk knows exactly where it came from (``start``/``end`` offsets, and via
:func:`parsrag.loaders.locate` its page and section).

* :func:`split_sentences` — a Persian-aware sentence splitter.
* :class:`FixedSizeChunker` — fixed character windows with overlap (lesson 01's
  baseline, now with offsets and word-boundary snapping).
* :class:`RecursiveChunker` — split by heading → paragraph → line → sentence →
  word until pieces fit, then pack neighbouring pieces up to ``size``.
* :class:`SentenceWindowChunker` — embed single sentences, but hand the LLM
  the sentence plus its neighbours (``metadata["window"]``).
* :func:`chunk_document` — run a chunker over a :class:`Document` and build
  :class:`Chunk` objects with positional metadata.
* :func:`estimate_tokens` — a rough, provider-independent token estimate.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from parsrag.types import Chunk, Document


@dataclass(frozen=True)
class Span:
    """A piece of a source text: ``text == source[start:end]``."""

    text: str
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


def _span(source: str, start: int, end: int, *, strip: bool = True) -> Span | None:
    """Span for ``source[start:end]`` with surrounding whitespace trimmed (None if empty)."""
    if strip:
        while start < end and source[start].isspace():
            start += 1
        while end > start and source[end - 1].isspace():
            end -= 1
    return Span(source[start:end], start, end) if end > start else None


# ------------------------------------------------------------------ sentences

# Things that end with a dot but do not end a sentence.
ABBREVIATIONS = frozenset(
    {
        # Persian: ante/post meridiem, solar/lunar hijri, BCE, "see", volume, page ...
        "ق.ظ", "ب.ظ", "ه.ش", "ه.ق", "ق.م", "ر.ک", "ن.ک", "ج.ا.ا", "ص", "ج", "م", "ش", "ه",
        # English
        "Dr", "Mr", "Mrs", "Ms", "Prof", "e.g", "i.e", "etc", "vs", "No", "Inc", "Ltd",
    }
)  # fmt: skip

# A terminator run (… or ... or any mix of . ! ? ؟ ؛) plus closing quotes/brackets,
# followed by whitespace or the end of the text — or a newline.
_BOUNDARY_RE = re.compile(r"(?:\.{2,}|…|[.!?؟؛]+)[»\"'”’)\]]*(?=\s|$)|\n")
_LAST_TOKEN_RE = re.compile(r"(\S+)$")


def is_abbreviation(text: str, dot: int) -> bool:
    """Is the ``.`` at index ``dot`` part of an abbreviation, an initial or a list number?"""
    m = _LAST_TOKEN_RE.search(text, 0, dot)
    if not m:
        return False
    token = m.group(1).lstrip("«(\"'")
    if token in ABBREVIATIONS:
        return True
    if len(token) == 1 and token.isalpha():  # an initial: «م. رضایی», «J. Smith»
        return True
    # "1." / "۲." at the start of a line is a list number, not a sentence
    line_start = text.rfind("\n", 0, m.start()) + 1
    return token.isdigit() and not text[line_start : m.start()].strip()


def split_sentences(text: str) -> list[Span]:
    """Split Persian (or mixed) text into sentences, keeping character offsets.

    Boundaries are ``.`` ``!`` ``?`` ``؟`` ``؛`` ``…`` (and runs like ``?!`` or
    ``...``) followed by whitespace, and newlines. A dot does **not** end a
    sentence inside numbers (``2.5``), web addresses (``www.kahroba.ir``),
    after abbreviations (``ق.ظ.``، ``Dr.``), initials (``م. رضایی``) or list
    numbers at the start of a line (``1.``). Persian decimal separators
    (``۹٫۶``) are not dots at all.
    """
    spans: list[Span] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        single_dot = m.group().rstrip("»\"'”’)]") == "."
        if single_dot and is_abbreviation(text, m.start()):
            continue
        end = m.start() if m.group() == "\n" else m.end()
        span = _span(text, start, end)
        if span:
            spans.append(span)
        start = m.end()
    tail = _span(text, start, len(text))
    if tail:
        spans.append(tail)
    return spans


# --------------------------------------------------------------------- tokens

_PRETOKEN_RE = re.compile(r"[\w‌]+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str, bytes_per_token: float = 4.0) -> int:
    """A rough, provider-independent token estimate: ~1 token per ``bytes_per_token`` UTF-8 bytes.

    Each word, number and punctuation mark costs ``ceil(utf8_bytes / bytes_per_token)``
    tokens (at least 1). Many modern tokenizers (byte-level BPE) start from UTF-8
    bytes and merge frequent sequences; Persian letters take 2 bytes (ZWNJ takes 3)
    and get fewer learned merges than English, so the *same content* costs more
    tokens. ``bytes_per_token`` is an assumption, not a measured constant — check
    your model's real tokenizer before budgeting.
    """
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    return sum(
        max(1, math.ceil(len(piece.encode("utf-8")) / bytes_per_token))
        for piece in _PRETOKEN_RE.findall(text)
    )


# ------------------------------------------------------------------- chunkers


@runtime_checkable
class Chunker(Protocol):
    """Anything that cuts a text into spans."""

    def split(self, text: str) -> list[Span]: ...


def _check_size(size: int, overlap: int) -> None:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")


class FixedSizeChunker:
    """Windows of ``size`` characters, each starting ``size - overlap`` after the previous one.

    With ``snap_to_words=True`` (default) window edges move back to the
    nearest whitespace so words are never cut in half (unless a single word
    is longer than the window).
    """

    def __init__(self, size: int = 400, overlap: int = 50, *, snap_to_words: bool = True) -> None:
        _check_size(size, overlap)
        self.size, self.overlap, self.snap_to_words = size, overlap, snap_to_words

    def _snap(self, text: str, pos: int, lo: int) -> int:
        """Move ``pos`` back to just after a whitespace character (not before ``lo``)."""
        if not self.snap_to_words or pos >= len(text) or text[pos].isspace():
            return pos
        cut = pos
        while cut > lo and not text[cut - 1].isspace():
            cut -= 1
        return cut if cut > lo else pos

    def split(self, text: str) -> list[Span]:
        spans: list[Span] = []
        start, n = 0, len(text)
        while start < n:
            while start < n and text[start].isspace():
                start += 1
            if start >= n:
                break
            end = self._snap(text, min(start + self.size, n), start)
            span = _span(text, start, end)
            if span:
                spans.append(span)
            if end >= n:
                break
            next_start = self._snap(text, max(end - self.overlap, start + 1), start)
            start = next_start if next_start > start else end
        return spans


Splitter = Callable[[str, Span], list[Span]]
_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]", re.MULTILINE)
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")


def _split_by(pattern: re.Pattern[str], before: bool) -> Splitter:
    """Splitter that cuts a span at every match (before it, or around it)."""

    def split(source: str, span: Span) -> list[Span]:
        cuts = [span.start]
        ends = []
        for m in pattern.finditer(source, span.start, span.end):
            if before:
                ends.append(m.start())
                cuts.append(m.start())
            else:
                ends.append(m.start())
                cuts.append(m.end())
        ends.append(span.end)
        return [s for a, b in zip(cuts, ends, strict=True) if (s := _span(source, a, b))]

    return split


def split_headings(source: str, span: Span) -> list[Span]:
    """Cut before every Markdown heading; a heading with no body stays with what follows."""
    parts = _split_by(_HEADING_LINE_RE, before=True)(source, span)
    merged: list[Span] = []
    pending: Span | None = None
    for part in parts:
        if pending is not None:
            part = Span(source[pending.start : part.end], pending.start, part.end)
            pending = None
        if _HEADING_LINE_RE.match(part.text) and "\n" not in part.text:
            pending = part  # heading only: attach to the next section
            continue
        merged.append(part)
    if pending is not None:
        merged.append(pending)
    return merged


def split_paragraphs(source: str, span: Span) -> list[Span]:
    """Cut at blank lines."""
    return _split_by(_PARAGRAPH_RE, before=False)(source, span)


def split_lines(source: str, span: Span) -> list[Span]:
    """Cut at single newlines."""
    return _split_by(re.compile(r"\n"), before=False)(source, span)


def split_sentence_spans(source: str, span: Span) -> list[Span]:
    """Cut into sentences with :func:`split_sentences`."""
    return [
        Span(s.text, span.start + s.start, span.start + s.end)
        for s in split_sentences(source[span.start : span.end])
    ]


def split_words(source: str, span: Span) -> list[Span]:
    """Cut at whitespace."""
    piece = source[span.start : span.end]
    return [
        Span(m.group(), span.start + m.start(), span.start + m.end())
        for m in re.finditer(r"\S+", piece)
    ]


DEFAULT_SPLITTERS: tuple[Splitter, ...] = (
    split_paragraphs,
    split_lines,
    split_sentence_spans,
    split_words,
)


def pack(source: str, units: list[Span], size: int, overlap: int,
         length: Callable[[str], int] = len) -> list[Span]:  # fmt: skip
    """Greedily join consecutive ``units`` into spans of at most ``size``.

    Each new span starts with the last units of the previous one, as long as
    they are at most ``overlap`` long — so overlap never cuts a sentence.
    """
    out: list[Span] = []
    i, n = 0, len(units)
    while i < n:
        j = i
        while j + 1 < n and length(source[units[i].start : units[j + 1].end]) <= size:
            j += 1
        out.append(Span(source[units[i].start : units[j].end], units[i].start, units[j].end))
        if j + 1 >= n:
            break
        k = j + 1
        while k - 1 > i and length(source[units[k - 1].start : units[j].end]) <= overlap:
            k -= 1
        i = k
    return out


class RecursiveChunker:
    """Split by structure first, size second.

    1. With ``respect_sections`` (default) the text is cut at Markdown headings
       and each section is chunked on its own, so a chunk never mixes two sections.
    2. Inside a section, pieces longer than ``size`` are split with the next
       splitter in line: paragraph → line → sentence → word (→ hard cut).
    3. The resulting pieces are packed back together up to ``size``; the next
       chunk repeats trailing *whole* pieces up to ``overlap``.

    ``length`` measures size (characters by default; pass
    :func:`estimate_tokens` to chunk by approximate tokens).
    """

    def __init__(
        self,
        size: int = 400,
        overlap: int = 50,
        *,
        respect_sections: bool = True,
        splitters: tuple[Splitter, ...] = DEFAULT_SPLITTERS,
        length: Callable[[str], int] = len,
    ) -> None:
        _check_size(size, overlap)
        self.size, self.overlap = size, overlap
        self.respect_sections = respect_sections
        self.splitters = splitters
        self.length = length

    def _atomize(self, source: str, span: Span, level: int) -> list[Span]:
        if self.length(span.text) <= self.size:
            return [span]
        if level >= len(self.splitters):  # nothing left: hard cut
            step = max(self.size, 1)
            return [
                s for a in range(span.start, span.end, step)
                if (s := _span(source, a, min(a + step, span.end)))
            ]  # fmt: skip
        parts = self.splitters[level](source, span)
        if len(parts) <= 1:
            return self._atomize(source, span, level + 1)
        return [unit for part in parts for unit in self._atomize(source, part, level + 1)]

    def split(self, text: str) -> list[Span]:
        whole = _span(text, 0, len(text))
        if whole is None:
            return []
        sections = split_headings(text, whole) if self.respect_sections else [whole]
        out: list[Span] = []
        for section in sections:
            units = self._atomize(text, section, 0)
            out.extend(pack(text, units, self.size, self.overlap, self.length))
        return out


class SentenceWindowChunker:
    """One chunk per sentence, plus the ``window`` sentences on each side as context.

    ``split`` returns the sentences (what gets embedded — small and precise);
    :meth:`windows` returns the matching context spans that
    :func:`chunk_document` stores in ``metadata["window"]`` for the LLM.
    Overlap is built in: neighbouring windows share sentences.
    """

    def __init__(self, window: int = 1) -> None:
        if window < 0:
            raise ValueError("window must be >= 0")
        self.window = window

    def split(self, text: str) -> list[Span]:
        return split_sentences(text)

    def windows(self, text: str, sentences: list[Span] | None = None) -> list[Span]:
        sents = sentences if sentences is not None else self.split(text)
        out = []
        for i in range(len(sents)):
            a, b = sents[max(0, i - self.window)], sents[min(len(sents) - 1, i + self.window)]
            out.append(Span(text[a.start : b.end], a.start, b.end))
        return out


# ----------------------------------------------------------- documents → chunks


def chunk_document(
    doc: Document,
    chunker: Chunker,
    *,
    preprocess: Callable[[str], str] | None = None,
    heading_context: bool = False,
) -> list[Chunk]:
    """Chunk ``doc`` and attach positional metadata to every chunk.

    ``metadata`` gets ``start``/``end`` (offsets in ``doc.text``), ``index``,
    ``section``/``section_path``/``page`` (from :func:`parsrag.loaders.locate`)
    and, for sentence windows, ``window``. ``preprocess`` (e.g. ``normalize``)
    is applied to the stored text *after* cutting, so offsets always refer to
    the original document. With ``heading_context`` the document title and
    section path are prepended to the text that gets embedded — a cheap way to
    give small chunks their context.
    """
    from parsrag.loaders.base import locate  # local import: loaders is a sibling package

    prep = preprocess or (lambda s: s)
    spans = chunker.split(doc.text)
    windows = None
    if isinstance(chunker, SentenceWindowChunker):
        windows = chunker.windows(doc.text, spans)
    chunks = []
    for i, span in enumerate(spans):
        where = locate(doc, span.start)
        meta: dict[str, Any] = {
            "start": span.start,
            "end": span.end,
            "index": i,
            "page": where["page"],
            "section": where["section"],
            "section_path": where["section_path"],
            "chunker": type(chunker).__name__,
        }
        text = span.text
        if heading_context:
            trail = [t for t in [doc.title, *where["section_path"]] if t]
            trail = [t for j, t in enumerate(trail) if t not in trail[:j]]
            if trail:
                text = " › ".join(trail) + "\n" + text
        if windows is not None:
            meta["window"] = prep(windows[i].text)
            meta["window_start"], meta["window_end"] = windows[i].start, windows[i].end
        chunks.append(Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=prep(text), metadata=meta))
    return chunks


__all__ = [
    "ABBREVIATIONS",
    "Chunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceWindowChunker",
    "Span",
    "chunk_document",
    "estimate_tokens",
    "is_abbreviation",
    "pack",
    "split_headings",
    "split_lines",
    "split_paragraphs",
    "split_sentence_spans",
    "split_sentences",
    "split_words",
]

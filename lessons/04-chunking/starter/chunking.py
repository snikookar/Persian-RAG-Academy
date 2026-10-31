"""Lesson 04 starter — chunking Persian text.

Fill in every function marked TODO, then check your work:

    python -m grader run 04

Each docstring is the specification. Helpers such as Span, the splitters
(split_headings, split_paragraphs, ...) and the classes' constructors are
written for you. Reference solution: src/parsrag/chunking.py (try not to
peek until you have tried!).
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
    """Is the ``.`` at index ``dot`` part of an abbreviation, an initial or a list number?

    1. Find the last whitespace-free token before ``dot``:
       ``_LAST_TOKEN_RE.search(text, 0, dot)`` (None -> False); strip leading «("' from it.
    2. True if the token is in ABBREVIATIONS              («ق.ظ»، «Dr»، ...)
    3. True if it is a single letter (``len == 1 and isalpha()``)  («م. رضایی»)
    4. True if it is all digits AND only whitespace precedes it on its line  («1. مورد»)
    5. Otherwise False.
    """
    # TODO: implement
    raise NotImplementedError("TODO: is_abbreviation")


def split_sentences(text: str) -> list[Span]:
    """Split Persian (or mixed) text into sentences, keeping character offsets.

    Boundaries are ``.`` ``!`` ``?`` ``؟`` ``؛`` ``…`` (and runs like ``?!`` or
    ``...``) followed by whitespace, and newlines. A dot does **not** end a
    sentence inside numbers (``2.5``), web addresses (``www.kahroba.ir``),
    after abbreviations (``ق.ظ.``، ``Dr.``), initials (``م. رضایی``) or list
    numbers at the start of a line (``1.``). Persian decimal separators
    (``۹٫۶``) are not dots at all.

    Steps: walk over ``_BOUNDARY_RE.finditer(text)`` keeping ``start`` (where the
    current sentence began).
      * a match that is exactly one "." (after removing closing quotes/brackets)
        and is_abbreviation(text, m.start()) -> not a boundary, continue;
      * the sentence ends at m.end() (or at m.start() for a "\n" match);
      * add _span(text, start, end) if it is not None; then start = m.end().
    Finally add the tail _span(text, start, len(text)).
    """
    # TODO: implement
    raise NotImplementedError("TODO: split_sentences")


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

    Steps: ValueError if bytes_per_token <= 0. For every match of _PRETOKEN_RE:
    max(1, math.ceil(len(piece.encode("utf-8")) / bytes_per_token)); return the sum.
    """
    # TODO: implement
    raise NotImplementedError("TODO: estimate_tokens")


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
        """Cut ``text`` into windows.

        start = 0; loop while start < len(text):
          * skip whitespace at ``start`` (stop if you reach the end)
          * end = self._snap(text, min(start + size, len(text)), start)
          * add _span(text, start, end) if not None; stop if end >= len(text)
          * next = self._snap(text, max(end - overlap, start + 1), start)
          * start = next if next > start else end      (always make progress!)
        """
        # TODO: implement
        raise NotImplementedError("TODO: FixedSizeChunker.split")


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

    i = 0; while i < len(units):
      * j = i; grow j while length(source[units[i].start : units[j + 1].end]) <= size
      * emit Span(source[units[i].start : units[j].end], units[i].start, units[j].end)
      * stop if j is the last unit
      * k = j + 1; move k back while k - 1 > i and
        length(source[units[k - 1].start : units[j].end]) <= overlap
      * i = k
    """
    # TODO: implement
    raise NotImplementedError("TODO: pack")


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
        """Break ``span`` into pieces that each fit in ``size`` (recursively).

        * if self.length(span.text) <= size: return [span]
        * if level == len(self.splitters): hard-cut into pieces of ``size``
          characters (use _span on each slice, skip None)
        * parts = self.splitters[level](source, span)
        * if there is <= 1 part, this splitter did not help: try level + 1 on span
        * otherwise atomize every part with level + 1 and concatenate the results
        """
        # TODO: implement
        raise NotImplementedError("TODO: RecursiveChunker._atomize")

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
        """One context span per sentence: from sentence i - window to i + window.

        Use ``sentences`` if given, else self.split(text). Clamp at both ends;
        the span is Span(text[a.start : b.end], a.start, b.end).
        """
        # TODO: implement
        raise NotImplementedError("TODO: SentenceWindowChunker.windows")


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

    For every span i of chunker.split(doc.text):
      * where = locate(doc, span.start)   (import it: from parsrag.loaders.base import locate)
      * metadata = {"start", "end", "index": i, "page", "section", "section_path"
        (from where), "chunker": type(chunker).__name__}
      * text = span.text; with heading_context prefix it with
        " › ".join(trail) + "\n" where trail = [doc.title, *section_path] without
        empty or repeated titles
      * for a SentenceWindowChunker also store metadata["window"] (preprocessed
        window text), "window_start" and "window_end"
      * Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=preprocess(text), metadata=...)
    """
    # TODO: implement
    raise NotImplementedError("TODO: chunk_document")


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

# Imported for you — you will need them.
_ = (math, Any)

"""Lesson 09 starter — grounded generation with citations and «نمی‌دانم».

Fill in every function marked TODO, then check your work:

    python -m grader run 09

The docstrings are the specification. The prompts, the data models and the
constructors are written for you. Reference solution: src/parsrag/generate.py
(try not to peek until you have tried!).

Suggested order: block_header → build_context → build_messages →
parse_citations → is_refusal → validate_citations → to_citation →
GroundedGenerator.answer → AbstentionStats → abstention_stats → calibrate_threshold.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from parsrag.llm import LLM, Message
from parsrag.normalize import normalize
from parsrag.retrieval import Retriever
from parsrag.types import Chunk, SearchResult

# ----------------------------------------------------------------- prompts

REFUSAL = "نمی‌دانم"

SYSTEM_PROMPT = """تو دستیار پاسخ‌گویی هستی که فقط بر اساس متن‌های شماره‌دارِ داده‌شده جواب می‌دهی.
قانون‌ها:
۱. فقط از اطلاعات همین متن‌ها استفاده کن، نه دانسته‌های خودت.
۲. بعد از هر جمله شمارهٔ متنِ منبع را در کروشه بیاور، مثل [1] یا [2][3].
۳. اگر پاسخ در متن‌ها نیست، فقط و دقیقاً بنویس: نمی‌دانم
۴. کوتاه، دقیق و به فارسی روان جواب بده."""

USER_TEMPLATE = """متن‌ها:
{context}

پرسش: {question}"""


def block_header(number: int, chunk: Chunk) -> str:
    """``[n] doc_id › section (صفحهٔ p)`` — what the model (and a human) sees above a chunk."""
    # TODO: implement
    raise NotImplementedError("TODO: block_header")


def chunk_context(chunk: Chunk) -> str:
    """The text shown to the model: the sentence window if there is one (lesson 04)."""
    window = chunk.metadata.get("window")
    return window if isinstance(window, str) else chunk.text


def build_context(results: Sequence[SearchResult]) -> str:
    """Numbered context blocks, one per retrieved chunk, separated by blank lines."""
    # TODO: implement
    raise NotImplementedError("TODO: build_context")


def build_messages(
    question: str,
    results: Sequence[SearchResult],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    template: str = USER_TEMPLATE,
) -> list[Message]:
    """System instructions + numbered context + the question."""
    # TODO: implement
    raise NotImplementedError("TODO: build_messages")


# --------------------------------------------------------------- citations

_CITE = re.compile(r"\[\s*([0-9۰-۹٠-٩]+(?:\s*[,،]\s*[0-9۰-۹٠-٩]+)*)\s*\]")
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def parse_citations(text: str) -> list[int]:
    """All cited block numbers in order of first appearance: ``[1]``, ``[۲]``, ``[1، 3]``."""
    # TODO: implement
    raise NotImplementedError("TODO: parse_citations")


def strip_citations(text: str) -> str:
    """The answer without ``[n]`` markers (for display or for metrics)."""
    return re.sub(r"\s{2,}", " ", _CITE.sub("", text)).strip()


class CitationCheck(BaseModel):
    """Result of :func:`validate_citations`."""

    cited: list[int]
    valid: list[int]
    invalid: list[int]  # numbers that were not in the context
    uncited: bool  # a non-refusal answer without any citation

    @property
    def ok(self) -> bool:
        return not self.invalid and not self.uncited


_REFUSALS = (
    "نمی‌دانم",
    "نمیدانم",
    "نمی دانم",
    "نمی‌دونم",
    "اطلاعاتی در این مورد",
    "در متن‌ها نیامده",
    "در متن‌ها نیست",
    "i don't know",
)


def is_refusal(text: str) -> bool:
    """True if the reply says «نمی‌دانم» (or a close variant) instead of answering."""
    # TODO: implement
    raise NotImplementedError("TODO: is_refusal")


def validate_citations(answer: str, n_context: int) -> CitationCheck:
    """Find cited numbers outside ``1..n_context`` and answers that cite nothing."""
    # TODO: implement
    raise NotImplementedError("TODO: validate_citations")


class Citation(BaseModel):
    """A cited context block mapped back to its source, ready for a UI link."""

    number: int
    chunk_id: str
    doc_id: str
    section: str | None = None
    page: int | None = None
    start: int | None = None
    end: int | None = None
    score: float | None = None


def to_citation(number: int, result: SearchResult) -> Citation:
    # TODO: implement
    raise NotImplementedError("TODO: to_citation")


AbstainReason = Literal["no_results", "low_score", "model"]


class GroundedAnswer(BaseModel):
    """Everything a caller (or a UI, or lesson 10's evaluator) needs about one answer."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: AbstainReason | None = None
    top_score: float | None = None
    invalid_citations: list[int] = Field(default_factory=list)
    uncited: bool = False
    sources: list[SearchResult] = Field(default_factory=list)
    raw: str = ""

    @property
    def text(self) -> str:
        """The answer without citation markers."""
        return strip_citations(self.answer)


# --------------------------------------------------------------- generator


Confidence = Callable[[str, Sequence[SearchResult]], float]


def top_score(question: str, results: Sequence[SearchResult]) -> float:
    """Default confidence: the score of the best retrieved result."""
    return results[0].score if results else float("-inf")


def retriever_confidence(retriever: Retriever) -> Confidence:
    """Confidence = the top score of *another* retriever (e.g. BM25) for the same question.

    Useful when the ranking retriever's scores are not comparable across
    questions (RRF ranks, reranker coverage), but a raw score is.
    """

    def confidence(question: str, results: Sequence[SearchResult]) -> float:
        best = retriever.search(question, 1)
        return best[0].score if best else 0.0

    return confidence


class GroundedGenerator:
    """Retrieve → (maybe abstain) → prompt → generate → check citations.

    ``min_score`` is a confidence threshold: if ``confidence(question,
    results)`` (default: the top result's score) is below it, the generator
    answers «نمی‌دانم» without calling the LLM. The right value depends on the
    score's scale — calibrate it with :func:`calibrate_threshold` on questions
    with known answerability.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        *,
        k: int = 4,
        min_score: float | None = None,
        confidence: Confidence = top_score,
        system_prompt: str = SYSTEM_PROMPT,
        template: str = USER_TEMPLATE,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.min_score = min_score
        self.confidence = confidence
        self.system_prompt = system_prompt
        self.template = template

    def answer(self, question: str) -> GroundedAnswer:
        # TODO: implement
        raise NotImplementedError("TODO: GroundedGenerator.answer")


# ------------------------------------------------------------- calibration


class AbstentionStats(BaseModel):
    """Abstention as a classifier: positive = "should say «نمی‌دانم»" (unanswerable)."""

    tp: int  # abstained on an unanswerable question
    fp: int  # abstained although the answer was there
    fn: int  # answered an unanswerable question
    tn: int  # answered an answerable question

    @property
    def precision(self) -> float:
        # TODO: implement
        raise NotImplementedError("TODO: AbstentionStats.precision")

    @property
    def recall(self) -> float:
        # TODO: implement
        raise NotImplementedError("TODO: AbstentionStats.recall")

    @property
    def f1(self) -> float:
        # TODO: implement
        raise NotImplementedError("TODO: AbstentionStats.f1")

    @property
    def accuracy(self) -> float:
        # TODO: implement
        raise NotImplementedError("TODO: AbstentionStats.accuracy")


def abstention_stats(abstained: Sequence[bool], answerable: Sequence[bool]) -> AbstentionStats:
    """Confusion counts for abstention decisions against known answerability."""
    # TODO: implement
    raise NotImplementedError("TODO: abstention_stats")


def calibrate_threshold(
    scores: Sequence[float], answerable: Sequence[bool]
) -> tuple[float, AbstentionStats]:
    """Pick the ``min_score`` that maximises abstention F1 (ties: the lower threshold).

    Candidates are the midpoints between consecutive observed scores (abstain
    if ``score < threshold``) plus "never" and "always". Returns the threshold
    and its stats on the given data — which are optimistic: report them on a
    *different* question set.
    """
    # TODO: implement
    raise NotImplementedError("TODO: calibrate_threshold")


__all__ = [
    "REFUSAL",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "AbstainReason",
    "AbstentionStats",
    "Citation",
    "CitationCheck",
    "Confidence",
    "GroundedAnswer",
    "GroundedGenerator",
    "abstention_stats",
    "block_header",
    "build_context",
    "build_messages",
    "calibrate_threshold",
    "chunk_context",
    "is_refusal",
    "parse_citations",
    "retriever_confidence",
    "strip_citations",
    "to_citation",
    "top_score",
    "validate_citations",
]

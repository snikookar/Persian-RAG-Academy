"""Grounded generation with citations and an honest «نمی‌دانم» (lesson 09).

The pipeline::

    retriever.search(question)          # lessons 07–08
      └─ top score < min_score?  → abstain («نمی‌دانم»), no LLM call
    build_messages(question, results)   # numbered context blocks [1] [2] ...
    llm.generate(messages)              # instructed to cite [n] or say «نمی‌دانم»
      └─ is_refusal(reply)?      → abstain
    parse_citations + validate          # [n] → chunk metadata (doc, section, page, offsets)

Every answer is a :class:`GroundedAnswer`; its :class:`Citation` objects carry
``doc_id``/``section``/``page``/``start``/``end`` so a UI can link to the exact
source span.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from parsrag.llm import LLM, Message
from parsrag.normalize import normalize
from parsrag.retrieval import Retriever
from parsrag.types import Chunk, SearchResult

if TYPE_CHECKING:
    from parsrag.security import InjectionGuard

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
    parts = [f"[{number}] {chunk.doc_id}"]
    section = chunk.metadata.get("section")
    if section:
        parts.append(f" › {section}")
    page = chunk.metadata.get("page")
    if page:
        parts.append(f" (صفحهٔ {page})")
    return "".join(parts)


def chunk_context(chunk: Chunk) -> str:
    """The text shown to the model: the sentence window if there is one (lesson 04)."""
    window = chunk.metadata.get("window")
    return window if isinstance(window, str) else chunk.text


def build_context(results: Sequence[SearchResult]) -> str:
    """Numbered context blocks, one per retrieved chunk, separated by blank lines."""
    return "\n\n".join(
        f"{block_header(i, r.chunk)}\n{chunk_context(r.chunk)}" for i, r in enumerate(results, 1)
    )


def build_messages(
    question: str,
    results: Sequence[SearchResult],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    template: str = USER_TEMPLATE,
) -> list[Message]:
    """System instructions + numbered context + the question."""
    user = template.format(context=build_context(results), question=question)
    return [Message(role="system", content=system_prompt), Message(role="user", content=user)]


# --------------------------------------------------------------- citations

_CITE = re.compile(r"\[\s*([0-9۰-۹٠-٩]+(?:\s*[,،]\s*[0-9۰-۹٠-٩]+)*)\s*\]")
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def parse_citations(text: str) -> list[int]:
    """All cited block numbers in order of first appearance: ``[1]``, ``[۲]``, ``[1، 3]``."""
    out: list[int] = []
    for m in _CITE.finditer(text):
        for part in re.split(r"[,،]", m.group(1)):
            n = int(part.strip().translate(_DIGITS))
            if n not in out:
                out.append(n)
    return out


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
    t = normalize(text).lower()
    return any(normalize(r).lower() in t for r in _REFUSALS)


def validate_citations(answer: str, n_context: int) -> CitationCheck:
    """Find cited numbers outside ``1..n_context`` and answers that cite nothing."""
    cited = parse_citations(answer)
    valid = [n for n in cited if 1 <= n <= n_context]
    invalid = [n for n in cited if not 1 <= n <= n_context]
    return CitationCheck(
        cited=cited, valid=valid, invalid=invalid, uncited=not valid and not is_refusal(answer)
    )


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
    meta = result.chunk.metadata
    return Citation(
        number=number,
        chunk_id=result.chunk.id,
        doc_id=result.chunk.doc_id,
        section=meta.get("section"),
        page=meta.get("page"),
        start=meta.get("start"),
        end=meta.get("end"),
        score=result.score,
    )


#: Why an answer is «نمی‌دانم»: lesson 09's generator uses the first three;
#: the agent (lesson 11) adds the next four; ``unsafe`` comes from lesson 14's guard.
AbstainReason = Literal[
    "no_results",
    "low_score",
    "model",
    "clarify",
    "out_of_scope",
    "not_relevant",
    "unfaithful",
    "unsafe",
]


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
    #: Problems found by lesson 14's output checks (e.g. ``link_removed``, ``followed``).
    security: list[str] = Field(default_factory=list)

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

    ``guard`` (lesson 14) switches the prompt to the sandboxed, sanitised one
    and checks the reply before it is returned; retrieval-time screening is
    done by wrapping the retriever (:meth:`InjectionGuard.wrap`).
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
        guard: InjectionGuard | None = None,
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
        self.guard = guard

    def answer(self, question: str) -> GroundedAnswer:
        return self.answer_from(question, self.retriever.search(question, self.k))

    def answer_from(self, question: str, results: Sequence[SearchResult]) -> GroundedAnswer:
        """Everything after retrieval: (maybe abstain) → prompt → generate → check citations.

        Split out of :meth:`answer` so callers can time, cache or replace the
        retrieval step (lessons 11–12) and still get the same answer.
        """
        results = list(results)
        early, top = self.precheck(question, results)
        if early is not None:
            return early
        raw = self.llm.generate(self.messages(question, results))
        return self.finalize(question, results, raw, top)

    def precheck(
        self, question: str, results: Sequence[SearchResult]
    ) -> tuple[GroundedAnswer | None, float | None]:
        """Abstain before calling the LLM? Returns ``(answer or None, confidence)``."""
        if not results:
            return GroundedAnswer(
                question=question, answer=REFUSAL, abstained=True, abstain_reason="no_results"
            ), None
        top = self.confidence(question, results)
        if self.min_score is not None and top < self.min_score:
            return GroundedAnswer(
                question=question,
                answer=REFUSAL,
                abstained=True,
                abstain_reason="low_score",
                top_score=top,
                sources=list(results),
            ), top
        return None, top

    def messages(self, question: str, results: Sequence[SearchResult]) -> list[Message]:
        """The prompt this generator sends for ``question`` and ``results``."""
        if self.guard is not None:
            return self.guard.messages(question, results)
        return build_messages(
            question, results, system_prompt=self.system_prompt, template=self.template
        )

    def finalize(
        self, question: str, results: Sequence[SearchResult], raw: str, top: float | None
    ) -> GroundedAnswer:
        """Turn the model's reply into a :class:`GroundedAnswer` (refusal, citations)."""
        if is_refusal(raw):
            return GroundedAnswer(
                question=question,
                answer=REFUSAL,
                abstained=True,
                abstain_reason="model",
                top_score=top,
                sources=list(results),
                raw=raw,
            )
        answer, security = raw, []
        if self.guard is not None and self.guard.check_output:
            verdict = self.guard.check_answer(raw, results)
            security = list(verdict.problems)
            if verdict.refuse:
                return GroundedAnswer(
                    question=question,
                    answer=REFUSAL,
                    abstained=True,
                    abstain_reason="unsafe",
                    top_score=top,
                    sources=list(results),
                    raw=raw,
                    security=security,
                )
            answer = verdict.answer
        check = validate_citations(answer, len(results))
        return GroundedAnswer(
            question=question,
            answer=answer.strip(),
            citations=[to_citation(n, results[n - 1]) for n in check.valid],
            top_score=top,
            invalid_citations=check.invalid,
            uncited=check.uncited,
            sources=list(results),
            raw=raw,
            security=security,
        )


# ------------------------------------------------------------- calibration


class AbstentionStats(BaseModel):
    """Abstention as a classifier: positive = "should say «نمی‌دانم»" (unanswerable)."""

    tp: int  # abstained on an unanswerable question
    fp: int  # abstained although the answer was there
    fn: int  # answered an unanswerable question
    tn: int  # answered an answerable question

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0


def abstention_stats(abstained: Sequence[bool], answerable: Sequence[bool]) -> AbstentionStats:
    """Confusion counts for abstention decisions against known answerability."""
    if len(abstained) != len(answerable):
        raise ValueError("need one answerability label per decision")
    pairs = list(zip(abstained, answerable, strict=True))
    return AbstentionStats(
        tp=sum(a and not ans for a, ans in pairs),
        fp=sum(a and ans for a, ans in pairs),
        fn=sum(not a and not ans for a, ans in pairs),
        tn=sum(not a and ans for a, ans in pairs),
    )


def calibrate_threshold(
    scores: Sequence[float], answerable: Sequence[bool]
) -> tuple[float, AbstentionStats]:
    """Pick the ``min_score`` that maximises abstention F1 (ties: the lower threshold).

    Candidates are the midpoints between consecutive observed scores (abstain
    if ``score < threshold``) plus "never" and "always". Returns the threshold
    and its stats on the given data — which are optimistic: report them on a
    *different* question set.
    """
    if len(scores) != len(answerable):
        raise ValueError("need one answerability label per score")
    distinct = sorted(set(scores))
    mids = [(a + b) / 2 for a, b in itertools.pairwise(distinct)]
    candidates = [float("-inf"), *mids, float("inf")]
    best_t, best = candidates[0], abstention_stats([False] * len(scores), answerable)
    for t in candidates:
        stats = abstention_stats([s < t for s in scores], answerable)
        if stats.f1 > best.f1:
            best_t, best = t, stats
    return best_t, best


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

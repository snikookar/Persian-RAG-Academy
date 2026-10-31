"""Lesson 10 starter (part 1 of 3) — generation metrics.

Fill in every function marked TODO, then check your work:

    python -m grader run 10

The docstrings are the specification; prompts and data classes are written
for you. Reference solution: src/parsrag/eval/generation.py (try not to peek!).
Parts 2 and 3 are stats.py and gate.py in this folder.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from parsrag.eval.dataset import QAItem
from parsrag.eval.retrieval import contains_evidence
from parsrag.llm import LLM, Message
from parsrag.normalize import normalize
from parsrag.types import Chunk

# ------------------------------------------------------------------- tokens

_CITATION = re.compile(r"\[\s*[0-9۰-۹٠-٩]+(?:\s*[,،]\s*[0-9۰-۹٠-٩]+)*\s*\]")
_WORD = re.compile(r"[\w‌]+")
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")

#: Function words ignored when comparing answers (kept small on purpose).
ANSWER_STOPWORDS: frozenset[str] = frozenset(
    ["و", "را", "به", "از", "در", "که", "است", "این", "آن", "با", "برای", "یا", "هم",
     "می", "ای", "ها", "های", "ی", "هست", "شود", "می‌شود", "بر"]
)  # fmt: skip


def answer_tokens(text: str, stopwords: frozenset[str] = ANSWER_STOPWORDS) -> list[str]:
    """Normalised lowercase tokens without citation markers and function words.

    ZWNJ is treated as a separator so «می‌شود» and «می شود» compare equal, and
    thousands separators are dropped so «۱۰٬۴۲۰» matches «10420».
    """
    # TODO: implement
    raise NotImplementedError("TODO: answer_tokens")


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall (multiset overlap); 0.0 if either is empty."""
    # TODO: implement
    raise NotImplementedError("TODO: token_f1")


# ---------------------------------------------------------------- citations


def citation_precision(cited: Sequence[Chunk], item: QAItem) -> float | None:
    """Share of cited chunks that contain the item's evidence span.

    ``None`` when nothing was cited or the question has no evidence (unanswerable).
    """
    # TODO: implement
    raise NotImplementedError("TODO: citation_precision")


def citation_hit(cited: Sequence[Chunk], item: QAItem) -> bool:
    """True if at least one cited chunk contains the evidence."""
    # TODO: implement
    raise NotImplementedError("TODO: citation_hit")


# ------------------------------------------------------------- faithfulness

_CLAIM_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n+|(?<=[.!?؟])$")


def split_claims(answer: str) -> list[str]:
    """Sentences of the answer (citation markers removed) — our unit of "claim"."""
    # TODO: implement
    raise NotImplementedError("TODO: split_claims")


@dataclass
class ClaimCheck:
    claim: str
    support: float  # best share of content words found in a single chunk
    numbers_ok: bool  # every number in the claim appears in that chunk
    supported: bool


@dataclass
class FaithfulnessResult:
    score: float  # supported claims / claims (1.0 when there are no claims)
    claims: list[ClaimCheck] = field(default_factory=list)


def check_claim(claim: str, contexts: Sequence[str], threshold: float = 0.8) -> ClaimCheck:
    """Is ``claim`` supported by at least one of ``contexts``?"""
    # TODO: implement
    raise NotImplementedError("TODO: check_claim")


def faithfulness(
    answer: str, contexts: Sequence[str], threshold: float = 0.8
) -> FaithfulnessResult:
    """Share of the answer's claims supported by the retrieved contexts."""
    # TODO: implement
    raise NotImplementedError("TODO: faithfulness")


# ------------------------------------------------------------- LLM as judge

JUDGE_SYSTEM = (
    "تو یک داور بی‌طرف و سخت‌گیر برای ارزیابی سیستم‌های پرسش و پاسخ فارسی هستی. "
    "فقط بر اساس شواهد داده‌شده قضاوت کن، نه دانسته‌های خودت. طول پاسخ یا لحن مطمئن آن "
    "نباید روی امتیازت اثر بگذارد."
)

FAITHFULNESS_TEMPLATE = """متن‌های منبع:
{context}

پرسش: {question}
پاسخ سیستم: {answer}

آیا **همهٔ** ادعاهای پاسخ سیستم در متن‌های منبع آمده است؟ ادعایی که در متن‌ها نیست،
حتی اگر درست باشد، پشتیبانی‌نشده حساب می‌شود.
دقیقاً در دو سطر جواب بده:
امتیاز: <عددی از ۰ (هیچ ادعایی پشتیبانی نمی‌شود) تا ۱۰ (همه پشتیبانی می‌شوند)>
دلیل: <یک جملهٔ کوتاه>"""

CORRECTNESS_TEMPLATE = """پرسش: {question}
پاسخ مرجع: {reference}
پاسخ سیستم: {answer}

پاسخ سیستم چقدر با پاسخ مرجع از نظر **معنا** یکی است؟ تفاوت در واژه‌ها، ارقام فارسی
یا لاتین و طول جمله مهم نیست؛ فقط درستی و کامل بودن مهم است.
دقیقاً در دو سطر جواب بده:
امتیاز: <عددی از ۰ (غلط) تا ۱۰ (کاملاً درست و کامل)>
دلیل: <یک جملهٔ کوتاه>"""

_REASON = re.compile(r"(?:دلیل|reason)\s*[:：]\s*(.+)", re.I)


@dataclass
class JudgeResult:
    score: float | None  # in [0, 1]; None if the reply could not be parsed
    reason: str
    raw: str


class LLMJudge:
    """LLM-as-judge with Persian prompts. Scores are mapped from 0–10 to 0–1.

    Known biases to keep in mind (lesson 10): judges prefer longer answers,
    answers in the first position, and text written by their own model family;
    they are also sensitive to prompt wording. Always check a judge against
    a small set of human labels before trusting it.
    """

    def __init__(self, llm: LLM, *, system_prompt: str = JUDGE_SYSTEM) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def _ask(self, user: str) -> JudgeResult:
        # TODO: implement
        raise NotImplementedError("TODO: LLMJudge._ask")

    def faithfulness(self, question: str, answer: str, contexts: Sequence[str]) -> JudgeResult:
        # TODO: implement
        raise NotImplementedError("TODO: LLMJudge.faithfulness")

    def correctness(self, question: str, answer: str, reference: str) -> JudgeResult:
        # TODO: implement
        raise NotImplementedError("TODO: LLMJudge.correctness")


__all__ = [
    "ANSWER_STOPWORDS",
    "CORRECTNESS_TEMPLATE",
    "FAITHFULNESS_TEMPLATE",
    "JUDGE_SYSTEM",
    "ClaimCheck",
    "FaithfulnessResult",
    "JudgeResult",
    "LLMJudge",
    "answer_tokens",
    "check_claim",
    "citation_hit",
    "citation_precision",
    "faithfulness",
    "split_claims",
    "token_f1",
]

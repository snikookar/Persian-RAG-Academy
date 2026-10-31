"""Generation metrics (lesson 10): correctness, citations, abstention, faithfulness.

All metrics work on plain strings and :class:`~parsrag.types.Chunk` objects,
so they can score any RAG system, not only parsrag's.

* :func:`token_f1` — SQuAD-style token overlap between an answer and the
  reference, after Persian normalisation (digits, ی/ک, ZWNJ, citations removed).
* :func:`citation_precision` — share of cited chunks that contain the evidence.
* :func:`faithfulness` — a transparent heuristic: split the answer into
  claims; a claim is *supported* if (almost) all of its content words and
  *all* of its numbers appear in one retrieved chunk.
* :class:`LLMJudge` — the same questions asked to an LLM with a Persian judge
  prompt (``FakeLLM`` in tests; real models only on your own machine).
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
    text = normalize(_CITATION.sub(" ", text)).lower()
    text = re.sub(r"(?<=\d)[,٬](?=\d{3})", "", text).replace("‌", " ")
    return [t for t in _WORD.findall(text) if t not in stopwords]


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall (multiset overlap); 0.0 if either is empty."""
    pred, ref = answer_tokens(prediction), answer_tokens(reference)
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- citations


def citation_precision(cited: Sequence[Chunk], item: QAItem) -> float | None:
    """Share of cited chunks that contain the item's evidence span.

    ``None`` when nothing was cited or the question has no evidence (unanswerable).
    """
    if not cited or item.evidence is None:
        return None
    return sum(contains_evidence(c, item) for c in cited) / len(cited)


def citation_hit(cited: Sequence[Chunk], item: QAItem) -> bool:
    """True if at least one cited chunk contains the evidence."""
    return any(contains_evidence(c, item) for c in cited)


# ------------------------------------------------------------- faithfulness

_CLAIM_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n+|(?<=[.!?؟])$")


def split_claims(answer: str) -> list[str]:
    """Sentences of the answer (citation markers removed) — our unit of "claim"."""
    text = _CITATION.sub("", answer)
    return [c.strip() for c in _CLAIM_SPLIT.split(text) if c and len(answer_tokens(c)) > 0]


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
    words = set(answer_tokens(claim))
    numbers = {w for w in words if _NUMBER.match(w)}
    best, best_numbers_ok = 0.0, not numbers
    for ctx in contexts:
        ctx_words = set(answer_tokens(ctx))
        share = len(words & ctx_words) / len(words) if words else 0.0
        numbers_ok = numbers <= ctx_words
        if (numbers_ok, share) > (best_numbers_ok, best):
            best, best_numbers_ok = share, numbers_ok
    return ClaimCheck(claim, best, best_numbers_ok, best >= threshold and best_numbers_ok)


def faithfulness(
    answer: str, contexts: Sequence[str], threshold: float = 0.8
) -> FaithfulnessResult:
    """Share of the answer's claims supported by the retrieved contexts."""
    claims = [check_claim(c, contexts, threshold) for c in split_claims(answer)]
    if not claims:
        return FaithfulnessResult(1.0, [])
    return FaithfulnessResult(sum(c.supported for c in claims) / len(claims), claims)


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
        from parsrag.rerank import parse_score  # shared "امتیاز: N" parser

        raw = self.llm.generate(
            [Message(role="system", content=self.system_prompt), Message(role="user", content=user)]
        )
        score = parse_score(raw)
        m = _REASON.search(raw)
        return JudgeResult(
            None if score is None else score / 10, m.group(1).strip() if m else "", raw
        )

    def faithfulness(self, question: str, answer: str, contexts: Sequence[str]) -> JudgeResult:
        context = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        return self._ask(
            FAITHFULNESS_TEMPLATE.format(context=context, question=question, answer=answer)
        )

    def correctness(self, question: str, answer: str, reference: str) -> JudgeResult:
        return self._ask(
            CORRECTNESS_TEMPLATE.format(question=question, reference=reference, answer=answer)
        )


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

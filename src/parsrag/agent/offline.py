"""An offline, deterministic "LLM" for every step of the agent (lesson 11).

:class:`OfflineAgentLLM` recognises which agent prompt it received (by its
system message) and answers in the format that prompt asks for:

======== ==================================================================
step     offline rule
======== ==================================================================
analyse  :func:`~parsrag.agent.rewrite.colloquial_to_standard`,
         :func:`~parsrag.agent.rewrite.split_questions`; «ابهام» when only
         pointer words are left, «خارج» when no content word is in the index
grade    :class:`~parsrag.rerank.LexicalReranker` coverage ≥ ``grade_threshold``
         (0.4: best answer F1 on ``qa.jsonl`` among 0.3–0.7, lesson 11)
rewrite  content words only; on the next try, also split at ZWNJ
generate the extractive, citing :class:`~parsrag.llm.FakeLLM` of lesson 09
check    the faithfulness heuristic of lesson 10, as «امتیاز: n»
======== ==================================================================

It goes through exactly the same prompts and parsers as a real model, so the
agent's code paths, call counts and loop logic are tested for real. What it
cannot do is *understand* — synonyms, intent, world knowledge. Measure the
agent with a real model (``examples/agentic_rag.py --llm``) before deciding.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence

from parsrag.agent.prompts import (
    ANALYZE_SYSTEM,
    DEFAULT_CLARIFICATION,
    GRADE_SYSTEM,
    REWRITE_SYSTEM,
)
from parsrag.agent.rewrite import (
    colloquial_to_standard,
    content_words,
    is_ambiguous,
    keyword_query,
    split_questions,
)
from parsrag.eval.generation import JUDGE_SYSTEM, faithfulness
from parsrag.llm import LLM, FakeLLM, Message, MessageLike, to_messages
from parsrag.rerank import LexicalReranker
from parsrag.retrieval import PERSIAN_STOPWORDS, PersianTokenizer

Handler = Callable[[list[Message]], str]


def task_of(messages: Sequence[MessageLike]) -> str:
    """Which agent step a prompt belongs to: analyze, grade, rewrite, check or generate."""
    msgs = to_messages(messages)
    system = msgs[0].content if msgs and msgs[0].role == "system" else ""
    return {
        ANALYZE_SYSTEM: "analyze",
        GRADE_SYSTEM: "grade",
        REWRITE_SYSTEM: "rewrite",
        JUDGE_SYSTEM: "check",
    }.get(system, "generate")


def _after(label: str, text: str) -> str:
    m = re.search(rf"^{label}\s*:\s*(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _blocks(text: str) -> list[str]:
    """Bodies of ``[n] ...`` blocks (the rest of the ``[n]`` line included)."""
    marks = list(re.finditer(r"^\[(\d+)\]\s?", text, re.MULTILINE))
    out = []
    for m, nxt in zip(marks, [*marks[1:], None], strict=True):
        out.append(text[m.end() : nxt.start() if nxt else len(text)].strip())
    return out


class OfflineAgentLLM:
    """Answer the agent's prompts with transparent rules (see the module docstring).

    ``overrides`` replaces one step, e.g. ``{"generate": lambda msgs: "..."}``
    to simulate a model that hallucinates; :attr:`calls` records
    ``(step, messages)`` for every call.
    """

    def __init__(
        self,
        *,
        vocabulary: Iterable[str] = (),
        grade_threshold: float = 0.4,
        colloquial: bool = True,
        answerer: LLM | None = None,
        overrides: Mapping[str, Handler] | None = None,
    ) -> None:
        self.vocabulary = frozenset(vocabulary)
        self.grade_threshold = grade_threshold
        self.colloquial = colloquial
        self.answerer = answerer or FakeLLM(mode="extractive", cite=True, ignore=PERSIAN_STOPWORDS)
        self.overrides = dict(overrides or {})
        self.reranker = LexicalReranker()
        self._tokenize = PersianTokenizer(stem=True)
        self.calls: list[tuple[str, list[Message]]] = []

    def generate(self, messages: Sequence[MessageLike]) -> str:
        msgs = to_messages(messages)
        task = task_of(msgs)
        self.calls.append((task, msgs))
        if task in self.overrides:
            return self.overrides[task](msgs)
        user = next((m.content for m in reversed(msgs) if m.role == "user"), "")
        if task == "analyze":
            return self.analyze(_after("پرسش کاربر", user))
        if task == "grade":
            question, _, context = user.partition("\n\nمتن‌ها:\n")
            question = question.removeprefix("پرسش:").strip()
            context = context.split("\n\nبرای هر متن")[0]
            return self.grade(question, _blocks(context))
        if task == "rewrite":
            previous = re.findall(r"^- (.+)$", user, re.MULTILINE)
            return self.rewrite(_after("پرسش اصلی", user), previous)
        if task == "check":
            context = user.split("متن‌های منبع:\n", 1)[-1].split("\n\nپرسش:", 1)[0]
            return self.check(_after("پاسخ سیستم", user), _blocks(context))
        return self.answerer.generate(msgs)

    # ------------------------------------------------------------- the rules

    def in_scope(self, question: str) -> bool:
        if not self.vocabulary:
            return True
        terms = self._tokenize(" ".join(content_words(question)))
        return not terms or any(t in self.vocabulary for t in terms)

    def standard(self, question: str) -> str:
        return colloquial_to_standard(question) if self.colloquial else question.strip()

    def analyze(self, question: str) -> str:
        standard = self.standard(question)
        if is_ambiguous(standard):
            route, clarify = "ابهام", DEFAULT_CLARIFICATION
        elif not self.in_scope(standard):
            route, clarify = "خارج", ""
        else:
            route, clarify = "اسناد", ""
        lines = [f"مسیر: {route}", f"بازنویسی: {standard}"]
        lines += [f"زیرپرسش: {q}" for q in split_questions(standard)]
        if clarify:
            lines.append(f"پرسش روشن‌کننده: {clarify}")
        return "\n".join(lines)

    def grade(self, question: str, passages: Sequence[str]) -> str:
        return "\n".join(
            f"[{i}] {'بله' if self.reranker.score(question, p) >= self.grade_threshold else 'خیر'}"
            for i, p in enumerate(passages, 1)
        )

    def rewrite(self, question: str, previous: Sequence[str]) -> str:
        standard = self.standard(question)
        query = keyword_query(standard, split_zwnj=len(previous) > 1)
        return query or standard

    def check(self, answer: str, contexts: Sequence[str]) -> str:
        result = faithfulness(answer, contexts)
        unsupported = [c.claim for c in result.claims if not c.supported]
        reason = "همهٔ ادعاها در متن‌ها آمده است." if not unsupported else (
            f"ادعای پشتیبانی‌نشده: {unsupported[0]}"
        )  # fmt: skip
        return f"امتیاز: {round(result.score * 10)}\nدلیل: {reason}"


__all__ = ["OfflineAgentLLM", "task_of"]

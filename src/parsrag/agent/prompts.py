"""Persian prompts of the agent's LLM steps and the parsers for their replies (lesson 11).

Every step asks for a small, fixed reply format so a parser — not another
LLM — can read it. Parsers are forgiving (Persian or Latin digits, extra
spaces, «بله/خیر» or yes/no) and fall back to a safe default when a reply
cannot be read.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from parsrag.generate import chunk_context
from parsrag.normalize import normalize
from parsrag.types import SearchResult

Route = Literal["documents", "clarify", "out_of_scope"]

# ------------------------------------------------------------------ analyse

ANALYZE_SYSTEM = (
    "تو مرحلهٔ تحلیلِ پرسش در یک سامانهٔ پرسش و پاسخ فارسی هستی. "
    "به پرسش جواب نده؛ فقط آن را برای جست‌وجو در اسناد آماده کن."
)

ANALYZE_TEMPLATE = """موضوع اسناد: {scope}

پرسش کاربر: {question}

۱. اگر پرسش محاوره‌ای یا شکسته است، آن را به فارسی معیار بازنویسی کن؛ معنا را عوض نکن.
۲. اگر پرسش چند بخش مستقل دارد، هر بخش را یک زیرپرسش جدا بنویس.
۳. مسیر را انتخاب کن: «اسناد» اگر پاسخ ممکن است در اسناد باشد، «ابهام» اگر پرسش بدون
توضیح بیشتر قابل فهم نیست، «خارج» اگر به موضوع اسناد ربطی ندارد.
دقیقاً با این قالب جواب بده:
مسیر: <اسناد | ابهام | خارج>
بازنویسی: <پرسش به فارسی معیار>
زیرپرسش: <زیرپرسش اول>
زیرپرسش: <زیرپرسش دوم، اگر هست>
پرسش روشن‌کننده: <فقط اگر مسیر «ابهام» است>"""

_ROUTES: dict[str, Route] = {
    "اسناد": "documents",
    "سند": "documents",
    "documents": "documents",
    "ابهام": "clarify",
    "مبهم": "clarify",
    "clarify": "clarify",
    "خارج": "out_of_scope",
    "out_of_scope": "out_of_scope",
}

DEFAULT_CLARIFICATION = "پرسش‌تان را کمی دقیق‌تر بگویید: دربارهٔ چه چیزی می‌پرسید؟"


@dataclass
class Analysis:
    """The parsed reply of the analyse step."""

    route: Route
    query: str
    sub_questions: list[str] = field(default_factory=list)
    clarification: str | None = None


def _field(line: str, *names: str) -> str | None:
    for name in names:
        m = re.match(rf"^\s*[-*]?\s*{name}\s*[:：]\s*(.*)$", line)
        if m:
            return m.group(1).strip()
    return None


def parse_analysis(reply: str, question: str) -> Analysis:
    """Read «مسیر / بازنویسی / زیرپرسش / پرسش روشن‌کننده» lines.

    Missing or unreadable fields fall back to: route «documents», the original
    question, no sub-questions besides the query itself.
    """
    route: Route = "documents"
    query, clarification = "", None
    subs: list[str] = []
    for line in normalize(reply).splitlines():
        if (v := _field(line, "مسیر", "route")) is not None:
            route = _ROUTES.get(v.strip("«» .").split()[0] if v.strip("«» .") else "", route)
        elif (v := _field(line, "بازنویسی", "rewrite")) is not None:
            query = v
        elif (v := _field(line, "زیرپرسش", "sub")) is not None:
            if v:
                subs.append(v)
        elif (v := _field(line, "پرسش روشن‌کننده", "پرسش روشن کننده", "clarify")) is not None:
            clarification = v or None
    query = query or question
    subs = subs or [query]
    if route == "clarify" and not clarification:
        clarification = DEFAULT_CLARIFICATION
    return Analysis(route, query, subs, clarification)


# -------------------------------------------------------------------- grade

GRADE_SYSTEM = (
    "تو یک ارزیاب دقیق برای موتور جست‌وجو هستی. برای هر متن فقط بگو آیا برای پاسخ دادن "
    "به پرسش مفید است یا نه."
)

GRADE_TEMPLATE = """پرسش: {question}

متن‌ها:
{context}

برای هر متن بگو آیا اطلاعاتی دارد که برای پاسخ دادن به پرسش **لازم** است.
دقیقاً برای هر متن یک سطر بنویس، مثل:
[1] بله
[2] خیر"""

_GRADE = re.compile(r"\[\s*([0-9۰-۹]+)\s*\]\s*[:：-]?\s*(بله|آره|خیر|نه|yes|no)", re.I)
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def grade_context(results: Sequence[SearchResult]) -> str:
    return "\n\n".join(f"[{i}] {chunk_context(r.chunk)}" for i, r in enumerate(results, 1))


def parse_grades(reply: str, n: int) -> list[bool]:
    """``[n] بله/خیر`` lines → one bool per passage; passages without a verdict count as no."""
    grades = [False] * n
    for m in _GRADE.finditer(reply):
        i = int(m.group(1).translate(_DIGITS))
        if 1 <= i <= n:
            grades[i - 1] = m.group(2).lower() in ("بله", "آره", "yes")
    return grades


# ------------------------------------------------------------------ rewrite

REWRITE_SYSTEM = (
    "تو به یک موتور جست‌وجوی فارسی کمک می‌کنی. پرسش را طوری بازنویسی کن که با واژه‌های "
    "اسناد رسمی بهتر پیدا شود."
)

REWRITE_TEMPLATE = """پرسش اصلی: {question}

جست‌وجوهای قبلی که نتیجهٔ خوبی نداشتند:
{previous}

پرسش را با واژه‌های دیگر (هم‌معنا، رسمی‌تر یا کلیدی‌تر) دوباره بنویس. معنا را عوض نکن.
فقط پرسش بازنویسی‌شده را در یک سطر بنویس."""


def parse_rewrite(reply: str, fallback: str) -> str:
    """First non-empty line, without a «پرسش:» style label; ``fallback`` if empty."""
    for line in reply.splitlines():
        line = re.sub(r"^\s*(?:[-*]|پرسش( بازنویسی‌شده)?\s*[:：])\s*", "", line).strip()
        if line:
            return line
    return fallback


__all__ = [
    "ANALYZE_SYSTEM",
    "ANALYZE_TEMPLATE",
    "DEFAULT_CLARIFICATION",
    "GRADE_SYSTEM",
    "GRADE_TEMPLATE",
    "REWRITE_SYSTEM",
    "REWRITE_TEMPLATE",
    "Analysis",
    "Route",
    "grade_context",
    "parse_analysis",
    "parse_grades",
    "parse_rewrite",
]

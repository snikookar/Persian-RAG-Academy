"""Rule-based question analysis for the offline agent (lesson 11).

A real LLM rewrites colloquial Persian into standard Persian and splits
multi-part questions far better than these rules. They exist so the agent's
tests and measurements run offline, and they are deliberately *generic*:
spoken-language morphology (``میشه`` → ``می‌شود``, ``کتابخونه`` →
``کتابخانه``), not a synonym list written for our question sets — such a list
would leak the test set into the system (lesson 10).
"""

from __future__ import annotations

import re

from parsrag.normalize import ZWNJ, normalize
from parsrag.retrieval import PERSIAN_STOPWORDS

#: Common spoken forms → standard written forms (whole words, after normalisation).
COLLOQUIAL_WORDS: dict[str, str] = {
    "میشه": "می‌شود",
    "نمیشه": "نمی‌شود",
    "بشه": "بشود",
    "نشه": "نشود",
    "اگه": "اگر",
    "رو": "را",
    "یه": "یک",
    "چیه": "چیست",
    "کیه": "کیست",
    "چنده": "چند است",
    "چقدره": "چقدر است",
    "چقد": "چقدر",
    "چطوری": "چطور",
    "اینو": "این را",
    "اونو": "آن را",
    "اون": "آن",
    "اونا": "آن‌ها",
    "اینا": "این‌ها",
    "همون": "همان",
    "همین‌جوری": "همین‌طور",
    "داره": "دارد",
    "نداره": "ندارد",
    "باشه": "باشد",
    "نباشه": "نباشد",
    "کنه": "کند",
    "نکنه": "نکند",
    "بدم": "بدهم",
    "بدیم": "بدهیم",
    "میخوام": "می‌خواهم",
    "می‌خوام": "می‌خواهم",
    "میخوای": "می‌خواهی",
    "می‌خوای": "می‌خواهی",
    "بخوام": "بخواهم",
    "میتونم": "می‌توانم",
    "می‌تونم": "می‌توانم",
    "نمی‌تونم": "نمی‌توانم",
    "میتونه": "می‌تواند",
    "می‌تونه": "می‌تواند",
    "میاد": "می‌آید",
    "میره": "می‌رود",
    "خونه": "خانه",
    "کتابخونه": "کتابخانه",
    "دانشگاهو": "دانشگاه را",
    "بیشتره": "بیشتر است",
    "کمتره": "کمتر است",
    "منو": "من را",
    "تو": "در",
    "دیگه": "دیگر",
    "بیاد": "بیاید",
    "بیاره": "بیاورد",
    "بیارن": "بیاورند",
    "درمیاد": "درمی‌آید",
    "برگرده": "برگردد",
    "بازه": "باز است",
}

#: Spoken filler words that carry no meaning for retrieval.
FILLERS = frozenset(["خب", "آخه", "حالا", "لطفا", "لطفاً", "ببخشید", "سلام"])

#: Question words: used to split multi-part questions and to find content words.
QUESTION_WORDS = frozenset(
    [
        "چقدر",
        "چند",
        "چه",
        "چی",
        "چیست",
        "چرا",
        "کی",
        "کیست",
        "کجا",
        "کجاست",
        "چطور",
        "چگونه",
        "آیا",
        "کدام",
        "کدوم",
    ]
)

#: Words that only point at something said earlier: a question made of these is ambiguous.
DEICTIC = frozenset(["این", "آن", "اون", "همین", "همان", "همون", "اینو", "اونو", "اینا", "اونا"])

_WORD = re.compile(r"[\w‌]+")
_MI_VERB_2PL = re.compile(rf"^(ن?می{ZWNJ}?)(\w{{2,}})ین$")  # می‌فرستین → می‌فرستید
_MI_VERB_3SG = re.compile(rf"^(ن?می{ZWNJ}?)(\w{{2,}})ه$")  # می‌رسه → می‌رسد
_MI_VERB_3PL = re.compile(rf"^(ن?می{ZWNJ}?)(\w{{2,}})ن$")  # می‌سوزن → می‌سوزند
_POSSESSIVE_OBJ = re.compile(r"^(\w{3,})(م|ت|ش)و$")  # سفارشمو → سفارشم را
_ZWNJ_AM = re.compile(rf"^(\w+){ZWNJ}م$")  # بسته‌م → بسته‌ام


def _verb(prefix: str, stem: str, ending: str) -> str:
    """``می`` + stem + ending, with ZWNJ, and spoken «ون» stems → «ان» (می‌مونه → می‌ماند)."""
    prefix = prefix if prefix.endswith(ZWNJ) else prefix + ZWNJ
    if stem.endswith("ون"):
        stem = stem[:-2] + "ان"
    return f"{prefix}{stem}{ending}"


def _standard_word(word: str) -> str:
    if word in COLLOQUIAL_WORDS:
        return COLLOQUIAL_WORDS[word]
    for pattern, ending in ((_MI_VERB_2PL, "ید"), (_MI_VERB_3SG, "د"), (_MI_VERB_3PL, "ند")):
        if m := pattern.match(word):
            return _verb(m.group(1), m.group(2), ending)
    if m := _POSSESSIVE_OBJ.match(word):
        return f"{m.group(1)}{m.group(2)} را"
    if m := _ZWNJ_AM.match(word):
        return f"{m.group(1)}{ZWNJ}ام"
    return word


def colloquial_to_standard(question: str) -> str:
    """Rewrite common spoken forms into standard Persian, word by word.

    >>> colloquial_to_standard("اگه کتابو دیر بدم چی میشه؟")
    'اگر کتابو دیر بدهم چی می‌شود؟'
    """
    text = normalize(question)
    out = []
    for token in re.split(r"(\s+)", text):
        m = re.fullmatch(r"([\w‌]+)(\W*)", token)
        if not m:
            out.append(token)
            continue
        word, tail = m.groups()
        if word in FILLERS:
            out.append(tail.strip())
            continue
        out.append(_standard_word(word) + tail)
    return re.sub(r"\s{2,}", " ", "".join(out)).strip(" ،,")


_SPLIT_AND = re.compile(r"\s+و\s+(?:همچنین\s+)?")


def split_questions(question: str) -> list[str]:
    """Split a multi-part question into its parts; one part → ``[question]``.

    Splits at question marks, and at «و» / «؛» when *both* sides contain a
    question word («هزینهٔ پیک چقدر است و مرجوعی تا کی ممکن است؟»).
    """
    parts = [p.strip() for p in re.split(r"(?<=[؟?])\s*", question) if p.strip(" ؟?،")]
    out: list[str] = []
    for part in parts:
        pieces = [part]
        for sep in (re.compile(r"\s*؛\s*"), _SPLIT_AND):
            split = []
            for piece in pieces:
                sides = sep.split(piece)
                if len(sides) > 1 and all(_has_question_word(s) for s in sides):
                    split.extend(sides)
                else:
                    split.append(piece)
            pieces = split
        out.extend(pieces)
    out = [p if p.endswith(("؟", "?")) else p.rstrip(" .،") + "؟" for p in out]
    return out or [question]


def _has_question_word(text: str) -> bool:
    return any(w in QUESTION_WORDS for w in _WORD.findall(normalize(text)))


def content_words(question: str) -> list[str]:
    """Words of the question that are not stop words, question words or fillers."""
    skip = PERSIAN_STOPWORDS | QUESTION_WORDS | FILLERS
    return [w for w in _WORD.findall(normalize(question)) if w not in skip and len(w) > 1]


def is_ambiguous(question: str) -> bool:
    """True if the question has no content word except pointers like «این» / «اون»."""
    return all(w in DEICTIC for w in content_words(colloquial_to_standard(question)))


def keyword_query(question: str, *, split_zwnj: bool = False) -> str:
    """The question reduced to its content words — a classic retry query."""
    words = content_words(question)
    if split_zwnj:
        words = [p for w in words for p in w.split(ZWNJ) if len(p) > 1]
    return " ".join(dict.fromkeys(words))


__all__ = [
    "COLLOQUIAL_WORDS",
    "DEICTIC",
    "FILLERS",
    "QUESTION_WORDS",
    "colloquial_to_standard",
    "content_words",
    "is_ambiguous",
    "keyword_query",
    "split_questions",
]

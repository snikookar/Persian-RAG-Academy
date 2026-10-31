"""Lesson 00 starter — an environment "doctor".

Fill in the three TODOs, then run:

    python -m grader run 00
    python lessons/00-setup/starter/doctor.py

Reference solution: src/parsrag/doctor.py
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence

from parsrag.embeddings import HashingEmbedding  # noqa: F401  (you will need these)
from parsrag.llm import FakeLLM, Message  # noqa: F401
from parsrag.stores.memory import InMemoryVectorStore  # noqa: F401
from parsrag.types import Chunk  # noqa: F401

MIN_PYTHON: tuple[int, int] = (3, 10)

EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "openai": ("httpx",),
    "ollama": ("ollama",),
    "st": ("sentence_transformers",),
    "faiss": ("faiss",),
    "pgvector": ("pgvector", "psycopg"),
    "agent": ("langgraph", "langchain_core"),
    "api": ("fastapi", "uvicorn"),
    "loaders": ("pypdf", "docx", "bs4"),
}

SMOKE_DOCS = {
    "tea": "چای ایرانی معمولاً با قند یا نبات سرو می‌شود.",
    "rain": "در شمال ایران باران زیاد می‌بارد و هوا مرطوب است.",
    "metro": "مترو تهران هر روز صبح ساعت شش کار خود را شروع می‌کند.",
}
SMOKE_QUESTION = "مترو تهران ساعت چند شروع به کار می‌کند؟"


def python_version_ok(
    min_version: tuple[int, int] = MIN_PYTHON,
    version_info: Sequence[int] | None = None,
) -> bool:
    """Return True if the running Python (or `version_info`, when given) is >= min_version.

    Only (major, minor) matter: (3, 10, 0) and (3, 10, 9) are both "3.10".
    """
    # TODO: implement (hint: sys.version_info, tuple slicing, tuple comparison)
    raise NotImplementedError("TODO: python_version_ok")


def module_available(name: str) -> bool:
    """True if `name` can be imported (without actually importing it). Done for you."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def extras_status() -> dict[str, bool]:
    """Return {extra_name: installed?} for every extra in EXTRA_MODULES.

    An extra counts as installed only if ALL of its modules are available.
    Use module_available().
    """
    # TODO: implement
    raise NotImplementedError("TODO: extras_status")


def smoke_test() -> str:
    """A tiny offline RAG loop. Return the FakeLLM's answer.

    1. Create an InMemoryVectorStore(HashingEmbedding()).
    2. Add one Chunk per SMOKE_DOCS item (id and doc_id = the dict key).
    3. Search for SMOKE_QUESTION with k=1.
    4. Ask FakeLLM(mode="extractive") with a single user Message whose content is
       f"{best_chunk_text}\n\nپرسش: {SMOKE_QUESTION}" and return its reply.
    """
    # TODO: implement
    raise NotImplementedError("TODO: smoke_test")


def main() -> int:
    ok = python_version_ok()
    print(f"Python {'.'.join(map(str, sys.version_info[:3]))}: {'OK' if ok else 'too old'}")
    for extra, installed in extras_status().items():
        print(f"  {'✔' if installed else '·'} {extra}")
    print(f"Offline smoke test: {smoke_test()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

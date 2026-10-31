"""Environment check (lesson 00): ``python -m parsrag.doctor``.

Reports the Python version, which optional extras are installed, and runs an
offline smoke test with :class:`~parsrag.llm.FakeLLM` and
:class:`~parsrag.embeddings.HashingEmbedding`.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence

from parsrag.embeddings import HashingEmbedding
from parsrag.llm import FakeLLM, Message
from parsrag.stores.memory import InMemoryVectorStore
from parsrag.types import Chunk

MIN_PYTHON: tuple[int, int] = (3, 10)

# extra name -> importable modules that must all be present
EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "openai": ("httpx",),
    "ollama": ("ollama",),
    "st": ("sentence_transformers",),
    "faiss": ("faiss",),
    "pgvector": ("pgvector", "psycopg"),
    "agent": ("langgraph", "langchain_core"),
    "api": ("fastapi", "uvicorn"),
    "loaders": ("pypdf", "docx", "bs4"),
    "dev": ("matplotlib", "fpdf"),
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
    """Return True if the running (or given) Python version is at least ``min_version``."""
    current = tuple(version_info if version_info is not None else sys.version_info)[:2]
    return current >= tuple(min_version)


def module_available(name: str) -> bool:
    """True if ``name`` can be imported (without importing it)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def extras_status() -> dict[str, bool]:
    """Map every optional extra to whether all of its modules are importable."""
    return {
        extra: all(module_available(m) for m in modules) for extra, modules in EXTRA_MODULES.items()
    }


def smoke_test() -> str:
    """Run a tiny offline retrieve-then-generate loop and return the answer.

    Indexes :data:`SMOKE_DOCS`, retrieves the best chunk for
    :data:`SMOKE_QUESTION` and lets an extractive :class:`FakeLLM` answer.
    """
    store = InMemoryVectorStore(HashingEmbedding())
    store.add(Chunk(id=key, doc_id=key, text=text) for key, text in SMOKE_DOCS.items())
    best = store.search(SMOKE_QUESTION, k=1)[0]
    llm = FakeLLM(mode="extractive")
    prompt = f"{best.chunk.text}\n\nپرسش: {SMOKE_QUESTION}"
    return llm.generate([Message(role="user", content=prompt)])


def main() -> int:
    ok = python_version_ok()
    version = ".".join(map(str, sys.version_info[:3]))
    print(f"Python {version}: {'OK' if ok else 'too old, need >= 3.10'}")
    print("Optional extras:")
    for extra, installed in extras_status().items():
        print(f"  {'✔' if installed else '·'} {extra}")
    answer = smoke_test()
    print(f"Offline smoke test: {answer}")
    return 0 if ok and answer else 1


if __name__ == "__main__":
    raise SystemExit(main())

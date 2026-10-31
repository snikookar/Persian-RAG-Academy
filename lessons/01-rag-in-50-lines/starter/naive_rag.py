"""Lesson 01 starter — the whole RAG loop in ~50 lines.

load → chunk → embed → retrieve → generate

Fill in the TODOs, then run:

    python -m grader run 01

Reference solution: src/parsrag/naive_rag.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from parsrag.embeddings import Embedding
from parsrag.llm import LLM, Message
from parsrag.stores.memory import InMemoryVectorStore
from parsrag.types import Chunk, Document, SearchResult

SYSTEM_PROMPT = (
    "تو دستیاری هستی که فقط بر اساس «متن‌های بازیابی‌شده» به پرسش‌ها پاسخ می‌دهد. "
    "اگر پاسخ در متن‌ها نیست، بگو «نمی‌دانم»."
)


class Answer(BaseModel):
    text: str
    sources: list[SearchResult]


def load_documents(directory: str | Path, pattern: str = "*.md") -> list[Document]:
    """Load every file as a Document (id = file stem, title = first '# ' line). Done for you."""
    docs = []
    for path in sorted(Path(directory).glob(pattern)):
        text = path.read_text(encoding="utf-8").strip()
        first, _, rest = text.partition("\n")
        title = first[2:].strip() if first.startswith("# ") else ""
        docs.append(Document(id=path.stem, title=title, text=rest.strip() if title else text))
    return docs


def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """Split `text` (stripped) into windows of `size` characters overlapping by `overlap`.

    - raise ValueError unless size > 0 and 0 <= overlap < size
    - empty / whitespace-only text -> []
    - "abcdefghij", size=4, overlap=1 -> ["abcd", "defg", "ghij"]
      (no extra tiny chunk that is entirely inside the previous one)
    """
    # TODO: implement
    raise NotImplementedError("TODO: chunk_text")


def build_prompt(question: str, results: list[SearchResult]) -> list[Message]:
    """Return [system message, user message].

    system: SYSTEM_PROMPT
    user:   "متن‌های بازیابی‌شده:\n" + one block per result, numbered from 1:
                "[1] (doc_id) chunk text"   (blocks separated by a blank line)
            then a blank line and finally  "پرسش: <question>"
    """
    # TODO: implement
    raise NotImplementedError("TODO: build_prompt")


class NaiveRAG:
    def __init__(
        self,
        embedding: Embedding,
        llm: LLM,
        *,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        k: int = 3,
        preprocess: Callable[[str], str] | None = None,
    ) -> None:
        self.llm = llm
        self.store = InMemoryVectorStore(embedding)
        self.chunk_size, self.chunk_overlap, self.k = chunk_size, chunk_overlap, k
        self.preprocess = preprocess or (lambda s: s)

    def index(self, documents: list[Document]) -> int:
        """Chunk every document (after self.preprocess), add the chunks to self.store.

        Chunk ids: f"{doc.id}#{i}". Return the number of chunks added.
        """
        # TODO: implement
        _ = Chunk  # you will need this
        raise NotImplementedError("TODO: NaiveRAG.index")

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        """Search the store for the preprocessed question (default k = self.k)."""
        # TODO: implement
        raise NotImplementedError("TODO: NaiveRAG.retrieve")

    def answer(self, question: str) -> Answer:
        """retrieve -> build_prompt -> self.llm.generate -> Answer(text, sources)."""
        # TODO: implement
        raise NotImplementedError("TODO: NaiveRAG.answer")

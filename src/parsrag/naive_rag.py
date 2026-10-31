"""The whole RAG loop in ~50 lines (lesson 01).

load → chunk (naively, by characters) → embed → retrieve → generate.

Later lessons replace each step with something better (Persian-aware
chunking, hybrid search, reranking, grounded generation with citations),
but the shape of the pipeline stays exactly the same.
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
    """The generated answer plus the chunks it was based on."""

    text: str
    sources: list[SearchResult]


def load_documents(directory: str | Path, pattern: str = "*.md") -> list[Document]:
    """Load every file matching ``pattern`` as a :class:`Document` (id = file stem).

    A first line starting with ``# `` becomes the title.
    """
    docs = []
    for path in sorted(Path(directory).glob(pattern)):
        text = path.read_text(encoding="utf-8").strip()
        first, _, rest = text.partition("\n")
        title = first[2:].strip() if first.startswith("# ") else ""
        docs.append(Document(id=path.stem, title=title, text=rest.strip() if title else text))
    return docs


def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """Split ``text`` into windows of ``size`` characters overlapping by ``overlap``."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    text = text.strip()
    step = size - overlap
    return [text[i : i + size] for i in range(0, max(len(text) - overlap, 1), step) if text]


def build_prompt(question: str, results: list[SearchResult]) -> list[Message]:
    """Put retrieved chunks and the question into chat messages."""
    context = "\n\n".join(
        f"[{i}] ({r.chunk.doc_id}) {r.chunk.text}" for i, r in enumerate(results, start=1)
    )
    user = f"متن‌های بازیابی‌شده:\n{context}\n\nپرسش: {question}"
    return [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=user)]


class NaiveRAG:
    """Minimal retrieval-augmented generation over an in-memory vector store."""

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
        """Chunk, embed and store documents. Returns the number of chunks added."""
        chunks = [
            Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=piece)
            for doc in documents
            for i, piece in enumerate(
                chunk_text(self.preprocess(doc.text), self.chunk_size, self.chunk_overlap)
            )
        ]
        self.store.add(chunks)
        return len(chunks)

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        """Return the ``k`` chunks most similar to ``question``."""
        return self.store.search(self.preprocess(question), k or self.k)

    def answer(self, question: str) -> Answer:
        """Retrieve, build the prompt, and ask the LLM."""
        results = self.retrieve(question)
        text = self.llm.generate(build_prompt(question, results))
        return Answer(text=text, sources=results)

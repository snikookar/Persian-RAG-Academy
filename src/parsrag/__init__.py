"""parsrag — Retrieval-Augmented Generation for Persian text.

The core package only depends on ``pydantic`` and ``numpy``. Real model
providers live behind optional extras and are imported lazily.
"""

from __future__ import annotations

from parsrag.embeddings import Embedding, HashingEmbedding
from parsrag.llm import LLM, FakeLLM, Message
from parsrag.stores.memory import InMemoryVectorStore
from parsrag.types import Chunk, Document, SearchResult

__version__ = "0.1.0.dev0"

__all__ = [
    "LLM",
    "Chunk",
    "Document",
    "Embedding",
    "FakeLLM",
    "HashingEmbedding",
    "InMemoryVectorStore",
    "Message",
    "SearchResult",
    "__version__",
]

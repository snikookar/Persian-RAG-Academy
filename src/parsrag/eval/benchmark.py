"""Benchmark any :class:`~parsrag.embeddings.Embedding` on a corpus + QA set (lesson 05).

>>> from parsrag import HashingEmbedding
>>> from parsrag.eval.benchmark import benchmark_embedding, load_fixture_benchmark
>>> docs, qa = load_fixture_benchmark()
>>> reports = benchmark_embedding(HashingEmbedding(), docs, qa)
>>> round(reports["passage"].mrr, 3)   # doctest: +SKIP

The pipeline is the one chosen in lesson 04 — recursive chunks of 400
characters with heading context, normalised text — so the *only* thing that
changes between runs is the embedding. Models with a ``fit`` method
(:class:`~parsrag.embeddings.TfidfEmbedding`) are fitted on the indexed chunks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from parsrag.chunking import Chunker, RecursiveChunker, chunk_document
from parsrag.embeddings import Embedding
from parsrag.eval.dataset import QAItem, load_qa
from parsrag.eval.retrieval import RetrievalReport, evaluate_retrieval
from parsrag.normalize import normalize
from parsrag.stores.memory import InMemoryVectorStore
from parsrag.types import Document

Preprocess = Callable[[str], str] | None
SAME: Literal["same"] = "same"

FIXTURES = Path(__file__).resolve().parents[3] / "data" / "fixtures"


def load_fixture_benchmark(
    fixtures: str | Path = FIXTURES, qa_file: str = "qa.jsonl"
) -> tuple[list[Document], list[QAItem]]:
    """The course corpus (Markdown, headings kept) and one of its QA sets.

    ``qa_file="qa_paraphrase.jsonl"`` loads the paraphrased questions (lesson 07+).
    """
    from parsrag.loaders import load_directory

    root = Path(fixtures)
    return load_directory(root / "corpus", "*.md"), load_qa(root / qa_file)


def index_documents(
    docs: Sequence[Document],
    embedding: Embedding,
    chunker: Chunker | None = None,
    *,
    preprocess: Preprocess = normalize,
    heading_context: bool = True,
) -> InMemoryVectorStore:
    """Chunk, (fit,) embed and store ``docs``."""
    chunker = chunker or RecursiveChunker(400, 50)
    chunks = [
        chunk
        for doc in docs
        for chunk in chunk_document(
            doc, chunker, preprocess=preprocess, heading_context=heading_context
        )
    ]
    fit = getattr(embedding, "fit", None)
    if callable(fit):
        fit([c.text for c in chunks])
    store = InMemoryVectorStore(embedding)
    store.add(chunks)
    return store


def benchmark_embedding(
    embedding: Embedding,
    docs: Sequence[Document],
    qa: Sequence[QAItem],
    *,
    chunker: Chunker | None = None,
    preprocess: Preprocess = normalize,
    query_preprocess: Preprocess | Literal["same"] = SAME,
    heading_context: bool = True,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, RetrievalReport]:
    """Return ``{"doc": report, "passage": report}`` for ``embedding``.

    ``query_preprocess`` defaults to the same function as ``preprocess``;
    set it differently to measure what happens when index and query are
    normalised differently.
    """
    store = index_documents(
        docs, embedding, chunker, preprocess=preprocess, heading_context=heading_context
    )
    prep_q = preprocess if query_preprocess == SAME else query_preprocess
    prep = prep_q if callable(prep_q) else (lambda s: s)

    def retrieve(question: str, k: int):  # type: ignore[no-untyped-def]
        return store.search(prep(question), k)

    return {
        level: evaluate_retrieval(qa, retrieve, ks, level=level)  # type: ignore[arg-type]
        for level in ("doc", "passage")
    }


__all__ = ["benchmark_embedding", "index_documents", "load_fixture_benchmark"]

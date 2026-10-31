"""Lesson 13 starter — your own ``PersianRAG``: the whole course behind one object.

Fill in every function marked TODO, then check your work:

    python -m grader run 13

What is given (inherited from ``parsrag.rag.PersianRAG``): the constructor,
``from_directory`` / ``from_config`` / ``course_default``, ``add_documents``
(which calls your ``_index``), ``search``, ``stream``, ``evaluate``,
``save`` / ``load``. What you write is the core: indexing, the retrieval
chain, answering, source passages, calibration — and a checker for your own
question file. Reference solution: src/parsrag/rag.py and
src/parsrag/eval/dataset.py (try not to peek!).

Suggested order: _index → hybrid → retriever → ask → passage → calibrate → validate_qa.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence

from parsrag.chunking import chunk_document
from parsrag.eval.dataset import QAItem, load_qa
from parsrag.eval.retrieval import match_key
from parsrag.generate import (
    Citation,
    GroundedAnswer,
    calibrate_threshold,
    retriever_confidence,
)
from parsrag.rag import (
    AnswerPayload,
    EmptyIndexError,
    QASource,
    StreamEvent,
    Timings,
    _as_items,
    answer_payload,
)
from parsrag.rag import PersianRAG as ReferenceRAG
from parsrag.rerank import RerankingRetriever
from parsrag.retrieval import DenseRetriever, HybridRetriever
from parsrag.stores.memory import InMemoryVectorStore
from parsrag.types import Document

__all__ = [
    "AnswerPayload",
    "EmptyIndexError",
    "PersianRAG",
    "StreamEvent",
    "Timings",
    "answer_payload",
    "load_qa",
    "validate_qa",
]


class PersianRAG(ReferenceRAG):
    """The course pipeline with your indexing, retrieval, answering and calibration."""

    def _index(self, docs: Sequence[Document]) -> int:
        """Chunk and index a non-empty batch of new documents; return the number of chunks.

        1. Reject ids already in ``self.documents`` or repeated in the batch:
           ``ValueError("documents already indexed: <ids>")``.
        2. ``chunk_document(d, self.chunker, preprocess=self.preprocess,
           heading_context=self.heading_context)`` for every document (lesson 04).
        3. Record the documents (``self.documents[d.id] = d``) and the chunks
           (``self._chunks``).
        4. Vectors: if ``self._fit_pending`` (TF-IDF-like embeddings) call
           ``self.embedding.fit`` on *all* chunk texts and rebuild
           ``self.store = InMemoryVectorStore(self.embedding)`` with all chunks;
           otherwise ``self.store.add(new chunks)``.
        5. ``self.bm25.add(new chunks)``; ``self.index_version += 1``.
        """
        # TODO: implement
        raise NotImplementedError("TODO: _index")

    @property
    def hybrid(self) -> HybridRetriever:
        """RRF over ``self.bm25`` and ``DenseRetriever(self.store, preprocess=self.preprocess)``."""
        # TODO: implement
        raise NotImplementedError("TODO: hybrid")

    @property
    def retriever(self) -> RerankingRetriever | HybridRetriever:
        """``self.hybrid`` wrapped in a reranker (lesson 08), unless ``self.reranker`` is None.

        ``RerankingRetriever(hybrid, self.reranker, fetch_k=self.fetch_k)``.
        """
        # TODO: implement
        raise NotImplementedError("TODO: retriever")

    def ask(self, question: str, *, timings: Timings | None = None) -> GroundedAnswer:
        """Answer with citations, or «نمی‌دانم».

        Raise ``EmptyIndexError`` without documents (``self._check_index()``).
        ``gen = self.generator``; retrieve ``gen.retriever.search(question, gen.k)``,
        then ``gen.answer_from(question, results)``. If ``timings`` is given,
        fill ``retrieve_ms``, ``generate_ms``, ``total_ms`` (``time.perf_counter``)
        and ``llm_calls`` (0 if the answer abstained for "no_results" or
        "low_score", else 1).
        """
        # TODO: implement
        raise NotImplementedError("TODO: ask")

    def passage(self, citation: Citation) -> str:
        """The exact source text of a citation: ``doc.text[start:end]`` of the original document.

        Unknown document → ``""``. Without offsets, the cited chunk's text
        (find it in ``self._chunks`` by ``citation.chunk_id``), or ``""``.
        """
        # TODO: implement
        raise NotImplementedError("TODO: passage")

    def calibrate(self, qa: QASource) -> float:
        """Set ``self.min_score`` to the BM25 threshold that maximises abstention F1 on ``qa``.

        ``qa`` is a JSONL path or a list of ``QAItem`` (``_as_items``). Confidence
        = ``retriever_confidence(self.bm25)(question, [])``; then lesson 09's
        ``calibrate_threshold(scores, answerable)``. Return the threshold.
        """
        # TODO: implement
        raise NotImplementedError("TODO: calibrate")


def validate_qa(
    items: Iterable[QAItem], documents: Mapping[str, Document] | Iterable[Document]
) -> list[str]:
    """Problems in a hand-written QA set, in order of the items. Messages (exactly):

    * ``"<id>: duplicate id"`` (from the second occurrence on)
    * ``"<id>: empty question"`` (only whitespace)
    * ``"<id>: unanswerable (no doc_id) but has evidence"``
    * ``"<id>: unknown doc_id '<doc_id>'"``
    * ``"<id>: answerable but has no evidence span"``
    * ``"<id>: evidence not found in '<doc_id>': '<evidence>'"`` — compare
      ``match_key(evidence)`` with ``match_key(document text)`` (lesson 05).

    Check them in this order for each item; an unanswerable item gets no
    doc_id/evidence checks after its own.
    """
    # TODO: implement
    raise NotImplementedError("TODO: validate_qa")

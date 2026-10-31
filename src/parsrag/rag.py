"""``PersianRAG`` — the whole course pipeline behind one object (lesson 13).

>>> from parsrag import PersianRAG
>>> rag = PersianRAG.from_directory("data/fixtures/corpus")       # doctest: +SKIP
>>> answer = rag.ask("هزینهٔ پیک در تهران چقدر است؟")              # doctest: +SKIP
>>> answer.text, [c.doc_id for c in answer.citations]             # doctest: +SKIP

Defaults are the choices measured in lessons 02–09:

* normalise (lesson 02); recursive chunks of 400 characters, overlap 50, with
  heading context (lesson 04);
* ``HashingEmbedding`` in an :class:`~parsrag.stores.InMemoryVectorStore`
  (lessons 05–06) — swap in a real model for real use;
* RRF over BM25 with light stemming + dense (lesson 07), lexical rerank of 20
  (lesson 08);
* :class:`~parsrag.generate.GroundedGenerator` with ``k=4``, citations and a
  BM25-score abstention threshold set by :meth:`PersianRAG.calibrate` (lesson 09);
* :meth:`PersianRAG.evaluate` = lesson 10's report.

Every part can be replaced with a keyword argument (``embedding``, ``llm``,
``chunker``, ``store``, ``reranker``). The default LLM is the offline
extractive :class:`~parsrag.llm.FakeLLM` — it copies a sentence from the
context, which is enough for tests and demos, not for users.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from parsrag.chunking import Chunker, RecursiveChunker, chunk_document
from parsrag.config import RAGConfig, load_config, make_embedding, make_llm
from parsrag.embeddings import Embedding, HashingEmbedding
from parsrag.eval.dataset import QAItem, load_qa
from parsrag.generate import (
    Citation,
    GroundedAnswer,
    GroundedGenerator,
    calibrate_threshold,
    retriever_confidence,
)
from parsrag.llm import LLM, stream_text
from parsrag.normalize import normalize
from parsrag.rerank import LexicalReranker, Reranker, RerankingRetriever
from parsrag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, PersianTokenizer
from parsrag.stores.base import Filter, VectorStore
from parsrag.stores.memory import InMemoryVectorStore
from parsrag.types import Chunk, Document, SearchResult

DEFAULT = object()  # "use the course default" (``None`` means "turn it off")


class EmptyIndexError(RuntimeError):
    """Raised when asking or searching before any document was added."""


class StreamEvent(BaseModel):
    """One event of :meth:`PersianRAG.stream`: ``sources`` → ``token``\\* → ``done``."""

    type: str  # "sources" | "token" | "done"
    data: Any = None


class Timings(BaseModel):
    """Per-stage latency of one :meth:`PersianRAG.ask` call, in milliseconds."""

    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0
    llm_calls: int = 0


QASource = str | Path | Sequence[QAItem]


def _as_items(qa: QASource) -> list[QAItem]:
    return load_qa(qa) if isinstance(qa, (str, Path)) else list(qa)


class PersianRAG:
    """Load → chunk → index → retrieve → rerank → answer with citations, with course defaults."""

    def __init__(
        self,
        documents: Iterable[Document | str | Path] = (),
        *,
        embedding: Embedding | None = None,
        llm: LLM | None = None,
        chunker: Chunker | None = None,
        store: VectorStore | None = None,
        reranker: Reranker | object | None = DEFAULT,
        k: int = 4,
        fetch_k: int = 20,
        stem: bool = True,
        heading_context: bool = True,
        normalize_text: bool = True,
        min_score: float | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.embedding = embedding or HashingEmbedding()
        self.llm = llm if llm is not None else make_llm(self.config.llm)
        self.chunker = chunker or RecursiveChunker(400, 50)
        self._fit_pending = callable(getattr(self.embedding, "fit", None))
        if store is None and not self._fit_pending:
            store = InMemoryVectorStore(self.embedding)
        # TF-IDF style models get their store after the first fit (see ``_index``).
        self._store: VectorStore | None = store
        self.reranker: Reranker | None = (
            LexicalReranker() if reranker is DEFAULT else reranker  # type: ignore[assignment]
        )
        self.k = k
        self.fetch_k = fetch_k
        self.heading_context = heading_context
        self.preprocess: Callable[[str], str] | None = normalize if normalize_text else None
        self.min_score = min_score
        self.bm25 = BM25Retriever(tokenizer=PersianTokenizer(stem=stem))
        self.documents: dict[str, Document] = {}
        self.index_version = 0
        self._chunks: list[Chunk] = []
        self.add_documents(documents)

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_directory(
        cls, directory: str | Path, pattern: str = "*", *, recursive: bool = True, **kwargs: Any
    ) -> PersianRAG:
        """Index every supported file (TXT, MD, HTML, DOCX, PDF — lesson 03) in ``directory``."""
        from parsrag.loaders import load_directory

        return cls(load_directory(directory, pattern, recursive=recursive), **kwargs)

    @classmethod
    def from_config(
        cls, source: str | Path | Mapping[str, Any] | RAGConfig, *, load_documents: bool = True,
        **overrides: Any,
    ) -> PersianRAG:  # fmt: skip
        """Build from a ``parsrag.toml`` (see :mod:`parsrag.config`); ``overrides`` win."""
        cfg = load_config(source)
        rr = cfg.retrieval
        kwargs: dict[str, Any] = {
            "embedding": make_embedding(cfg.embedding),
            "chunker": RecursiveChunker(cfg.chunking.size, cfg.chunking.overlap),
            "k": rr.k,
            "fetch_k": rr.fetch_k,
            "stem": rr.stem,
            "heading_context": cfg.chunking.heading_context,
            "normalize_text": cfg.chunking.normalize,
            "min_score": rr.min_score,
            "reranker": _make_reranker(rr.rerank, rr.cross_encoder),
            "config": cfg,
        }
        if "llm" not in overrides:
            kwargs["llm"] = make_llm(cfg.llm)
        kwargs.update(overrides)
        rag = cls(**kwargs)
        if load_documents:
            for path in cfg.documents.paths:
                rag.add_path(cfg.resolve(path), cfg.documents.pattern, cfg.documents.recursive)
        return rag

    @classmethod
    def course_default(
        cls, fixtures: str | Path | None = None, *, calibrate: bool = True, **kwargs: Any
    ) -> PersianRAG:
        """The lesson 04–09 pipeline on the course fixtures, calibrated on ``qa.jsonl``.

        Offline and deterministic: this is what ``python -m parsrag.eval`` gates on.
        """
        from parsrag.eval.benchmark import FIXTURES, load_fixture_benchmark

        docs, qa = load_fixture_benchmark(fixtures or FIXTURES)
        rag = cls(docs, **kwargs)
        if calibrate:
            rag.calibrate(qa)
        return rag

    # ---------------------------------------------------------------- indexing

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            raise EmptyIndexError("the vector store is created with the first documents")
        return self._store

    @store.setter
    def store(self, value: VectorStore) -> None:
        self._store = value

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add_path(self, path: str | Path, pattern: str = "*", recursive: bool = True) -> int:
        """Add one file or every supported file in a folder. Returns the number of new chunks."""
        from parsrag.loaders import load_directory, load_document

        p = Path(path)
        if p.is_dir():
            return self.add_documents(load_directory(p, pattern, recursive=recursive))
        if p.is_file():
            return self.add_documents([load_document(p)])
        raise FileNotFoundError(f"no such file or folder: {p}")

    def add_documents(self, documents: Iterable[Document | str | Path]) -> int:
        """Chunk and index documents; strings and paths are files or folders to load.

        Document ids must be new (re-adding a document would duplicate its
        chunks). Returns the number of chunks added.
        """
        added = 0
        batch: list[Document] = []
        for item in documents:
            if isinstance(item, Document):
                batch.append(item)
            else:
                added += self._index(batch)
                batch = []
                added += self.add_path(item)
        return added + self._index(batch)

    def add_texts(self, texts: Sequence[str], ids: Sequence[str] | None = None) -> int:
        """Index raw strings as documents (ids default to ``text-<n>``)."""
        start = len(self.documents)
        ids = list(ids) if ids is not None else [f"text-{start + i}" for i in range(len(texts))]
        if len(ids) != len(texts):
            raise ValueError("need one id per text")
        return self.add_documents([Document(id=i, text=t) for i, t in zip(ids, texts, strict=True)])

    def _index(self, docs: Sequence[Document]) -> int:
        if not docs:
            return 0
        ids = [d.id for d in docs]
        dupes = sorted({i for i in ids if i in self.documents or ids.count(i) > 1})
        if dupes:
            raise ValueError(f"documents already indexed: {', '.join(dupes)}")
        chunks = [
            c
            for d in docs
            for c in chunk_document(
                d, self.chunker, preprocess=self.preprocess, heading_context=self.heading_context
            )
        ]
        for d in docs:
            self.documents[d.id] = d
        self._chunks.extend(chunks)
        if self._fit_pending:
            # TF-IDF style models must see the whole collection: refit and re-embed.
            self.embedding.fit([c.text for c in self._chunks])  # type: ignore[attr-defined]
            self.store = InMemoryVectorStore(self.embedding)
            self.store.add(self._chunks)
        else:
            self.store.add(chunks)
        self.bm25.add(chunks)
        self.index_version += 1
        return len(chunks)

    # --------------------------------------------------------------- pipeline

    @property
    def retriever(self) -> RerankingRetriever | HybridRetriever:
        """RRF(BM25, dense), reranked when a reranker is set."""
        hybrid = HybridRetriever(
            [self.bm25, DenseRetriever(self.store, preprocess=self.preprocess)]
        )
        if self.reranker is None:
            return hybrid
        return RerankingRetriever(hybrid, self.reranker, fetch_k=self.fetch_k)

    @property
    def generator(self) -> GroundedGenerator:
        """The lesson 09 generator over :attr:`retriever` (built on demand)."""
        return GroundedGenerator(
            self.retriever,
            self.llm,
            k=self.k,
            min_score=self.min_score,
            confidence=retriever_confidence(self.bm25),
        )

    def _check_index(self) -> None:
        if not self._chunks:
            raise EmptyIndexError(
                "the index is empty — add documents first / هنوز سندی اضافه نشده است"
            )

    def search(
        self, query: str, k: int | None = None, *, filter: Filter | None = None
    ) -> list[SearchResult]:
        """Hybrid + reranked search."""
        self._check_index()
        return self.retriever.search(query, k or self.k, filter=filter)

    def ask(self, question: str, *, timings: Timings | None = None) -> GroundedAnswer:
        """Answer with citations, or «نمی‌دانم». Pass ``timings`` to get per-stage latency."""
        self._check_index()
        t0 = time.perf_counter()
        gen = self.generator
        results = gen.retriever.search(question, gen.k)
        t1 = time.perf_counter()
        answer = gen.answer_from(question, results)
        t2 = time.perf_counter()
        if timings is not None:
            timings.retrieve_ms = (t1 - t0) * 1000
            timings.generate_ms = (t2 - t1) * 1000
            timings.total_ms = (t2 - t0) * 1000
            timings.llm_calls = 0 if answer.abstain_reason in ("no_results", "low_score") else 1
        return answer

    def stream(self, question: str) -> Iterator[StreamEvent]:
        """Like :meth:`ask`, but yields ``sources``, then answer ``token``\\s, then ``done``.

        ``done`` carries the final :class:`GroundedAnswer` (citations checked
        on the full text). An early abstention yields ``sources`` and ``done`` only.
        """
        self._check_index()
        gen = self.generator
        results = gen.retriever.search(question, gen.k)
        yield StreamEvent(type="sources", data=results)
        early, top = gen.precheck(question, results)
        if early is not None:
            yield StreamEvent(type="done", data=early)
            return
        pieces: list[str] = []
        for piece in stream_text(self.llm, gen.messages(question, results)):
            pieces.append(piece)
            yield StreamEvent(type="token", data=piece)
        yield StreamEvent(type="done", data=gen.finalize(question, results, "".join(pieces), top))

    def passage(self, citation: Citation) -> str:
        """The exact source span of a citation (original text, before normalisation)."""
        doc = self.documents.get(citation.doc_id)
        if doc is None:
            return ""
        if citation.start is None or citation.end is None:
            return next((c.text for c in self._chunks if c.id == citation.chunk_id), "")
        return doc.text[citation.start : citation.end]

    # ------------------------------------------------------------- evaluation

    def calibrate(self, qa: QASource) -> float:
        """Choose the BM25 abstention threshold that maximises abstention F1 on ``qa``."""
        self._check_index()
        items = _as_items(qa)
        confidence = retriever_confidence(self.bm25)
        threshold, _ = calibrate_threshold(
            [confidence(q.question, []) for q in items], [q.answerable for q in items]
        )
        self.min_score = threshold
        return threshold

    def evaluate(self, qa: QASource, *, name: str = "pipeline", **kwargs: Any) -> Any:
        """Lesson 10's :class:`~parsrag.eval.pipeline.PipelineReport` on a QA set (JSONL path)."""
        from parsrag.eval.pipeline import evaluate_pipeline

        self._check_index()
        return evaluate_pipeline(self.generator, _as_items(qa), name=name, **kwargs)

    # ------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        """Write documents, options and (for in-memory stores) the vectors to ``directory``."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "documents.jsonl").open("w", encoding="utf-8") as f:
            for doc in self.documents.values():
                f.write(doc.model_dump_json() + "\n")
        options = {
            "k": self.k,
            "fetch_k": self.fetch_k,
            "min_score": self.min_score,
            "heading_context": self.heading_context,
            "normalize_text": self.preprocess is not None,
            "embedding": type(self.embedding).__name__,
            "n_chunks": len(self._chunks),
        }
        (path / "rag.json").write_text(json.dumps(options, indent=2) + "\n", encoding="utf-8")
        if isinstance(self._store, InMemoryVectorStore):
            self._store.save(path / "store")

    @classmethod
    def load(cls, directory: str | Path, **kwargs: Any) -> PersianRAG:
        """Load an index written by :meth:`save`.

        Pass the same ``embedding``/``chunker`` as when saving (the defaults if
        you used them). Chunks are rebuilt from the documents (cheap and
        deterministic); saved vectors are reused when they match.
        """
        path = Path(directory)
        options = json.loads((path / "rag.json").read_text("utf-8"))
        for key in ("k", "fetch_k", "min_score", "heading_context", "normalize_text"):
            kwargs.setdefault(key, options[key])
        rag = cls(**kwargs)
        with (path / "documents.jsonl").open(encoding="utf-8") as f:
            docs = [Document.model_validate_json(line) for line in f if line.strip()]
        store_dir = path / "store"
        reusable = (
            store_dir.is_dir()
            and not rag._fit_pending
            and isinstance(rag.store, InMemoryVectorStore)
            and options.get("embedding") == type(rag.embedding).__name__
        )
        if not reusable:
            rag.add_documents(docs)
            return rag
        saved = InMemoryVectorStore.load(store_dir, rag.embedding)
        chunks = [
            c
            for d in docs
            for c in chunk_document(
                d, rag.chunker, preprocess=rag.preprocess, heading_context=rag.heading_context
            )
        ]
        if [c.id for c in chunks] != [c.id for c in saved.chunks]:
            rag.add_documents(docs)  # chunker changed since saving: rebuild
            return rag
        rag.documents = {d.id: d for d in docs}
        rag._chunks = chunks
        rag.store = saved
        rag.bm25.add(chunks)
        rag.index_version += 1
        return rag


def _make_reranker(kind: str, model: str | None) -> Reranker | None:
    kind = kind.lower()
    if kind in ("none", "off", ""):
        return None
    if kind == "lexical":
        return LexicalReranker()
    if kind in ("cross-encoder", "cross_encoder"):
        from parsrag.rerank import CrossEncoderReranker

        return CrossEncoderReranker(model) if model else CrossEncoderReranker()
    raise ValueError(f"unknown reranker {kind!r} (lexical | cross-encoder | none)")


class AnswerPayload(BaseModel):
    """A JSON-friendly view of a :class:`GroundedAnswer` (used by the CLI and the API)."""

    question: str
    answer: str
    text: str
    abstained: bool
    abstain_reason: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)


def answer_payload(rag: PersianRAG, answer: GroundedAnswer) -> AnswerPayload:
    """Answer + citations, each with the exact source passage for display."""
    return AnswerPayload(
        question=answer.question,
        answer=answer.answer,
        text=answer.text,
        abstained=answer.abstained,
        abstain_reason=answer.abstain_reason,
        citations=[{**c.model_dump(), "passage": rag.passage(c)} for c in answer.citations],
    )


__all__ = [
    "AnswerPayload",
    "EmptyIndexError",
    "PersianRAG",
    "StreamEvent",
    "Timings",
    "answer_payload",
]

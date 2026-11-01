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
* :meth:`PersianRAG.evaluate` = lesson 10's report;
* no prompt-injection guard unless you opt in with ``guard=True`` or an
  :class:`~parsrag.security.InjectionGuard` (lesson 14).

Working on real data:

* :meth:`PersianRAG.save` / :meth:`PersianRAG.load` persist documents, chunks,
  vectors, the BM25 index and the configuration (a versioned folder format,
  :data:`INDEX_FORMAT_VERSION`), so a large collection is embedded once;
* :meth:`PersianRAG.upsert_documents` adds new documents, re-indexes changed
  ones (content hash) and skips unchanged ones; :meth:`PersianRAG.remove_documents`
  deletes by id;
* ``rag.ask(question, filters={"lang": "fa"})`` restricts retrieval by document
  or chunk metadata (:meth:`PersianRAG.resolve_filter`).

Every part can be replaced with a keyword argument (``embedding``, ``llm``,
``chunker``, ``store``, ``reranker``). The default LLM is the offline
extractive :class:`~parsrag.llm.FakeLLM` — it copies a sentence from the
context, which is enough for tests and demos, not for users.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from parsrag.chunking import Chunker, RecursiveChunker, chunk_document
from parsrag.config import RAGConfig, load_config, make_embedding, make_llm
from parsrag.embeddings import Embedding, HashingEmbedding
from parsrag.eval.dataset import QAItem, load_qa, validate_qa
from parsrag.generate import (
    Citation,
    GroundedAnswer,
    GroundedGenerator,
    calibrate_threshold,
    retriever_confidence,
)
from parsrag.llm import LLM, split_for_streaming, stream_text
from parsrag.normalize import normalize
from parsrag.rerank import LexicalReranker, Reranker, RerankingRetriever
from parsrag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    PersianTokenizer,
    Retriever,
)
from parsrag.security import InjectionGuard
from parsrag.stores.base import Filter, VectorStore
from parsrag.stores.memory import InMemoryVectorStore
from parsrag.types import Chunk, Document, SearchResult

DEFAULT = object()  # "use the course default" (``None`` means "turn it off")

#: Version of the folder written by :meth:`PersianRAG.save`. Version 1 (0.1.0)
#: had no number and stored only documents, options and in-memory vectors;
#: :meth:`PersianRAG.load` still reads it.
INDEX_FORMAT_VERSION = 2

#: Filter keys that refer to chunks (everything else refers to document metadata).
CHUNK_FILTER_KEYS = frozenset(
    {"id", "doc_id", "start", "end", "index", "page", "section", "section_path", "chunker",
     "window", "window_start", "window_end"}
)  # fmt: skip


def document_hash(doc: Document) -> str:
    """SHA-256 of a document's text, title and metadata (``source`` path excluded).

    Used by :meth:`PersianRAG.upsert_documents` to skip unchanged documents. The
    file path is left out so that indexing the same folder from another working
    directory does not count as a change.
    """
    meta = {k: v for k, v in doc.metadata.items() if k != "source"}
    payload = json.dumps(
        {"text": doc.text, "title": doc.title, "metadata": meta},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_set(value: Any) -> set[Any]:
    return set(value) if isinstance(value, (list, tuple, set, frozenset)) else {value}


def _document_matches(doc: Document, conditions: Mapping[str, Any]) -> bool:
    """Equality / any-of on ``doc.title`` or ``doc.metadata``; list values match by overlap."""
    for key, wanted in conditions.items():
        value = doc.title if key == "title" else doc.metadata.get(key)
        allowed = _as_set(wanted)
        if isinstance(value, (list, tuple, set, frozenset)):
            if not allowed & set(value):
                return False
        elif value not in allowed:
            return False
    return True


class IndexUpdate(BaseModel):
    """What :meth:`PersianRAG.upsert_documents` (or ``parsrag ingest``) changed."""

    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    chunks_added: int = 0
    chunks_removed: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def summary(self) -> str:
        return (
            f"added {len(self.added)}, updated {len(self.updated)}, "
            f"unchanged {len(self.unchanged)}, removed {len(self.removed)} documents "
            f"(+{self.chunks_added} / -{self.chunks_removed} chunks)"
        )


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
        guard: InjectionGuard | bool | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.guard: InjectionGuard | None = InjectionGuard() if guard is True else (guard or None)
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
        self.stem = stem
        self.bm25 = BM25Retriever(tokenizer=PersianTokenizer(stem=stem))
        self.documents: dict[str, Document] = {}
        self.index_version = 0
        self._chunks: list[Chunk] = []
        self._hashes: dict[str, str] = {}
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
        kwargs = _config_kwargs(cfg, skip=overrides)
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
                added += self._index(batch) if batch else 0
                batch = []
                added += self.add_path(item)
        return added + (self._index(batch) if batch else 0)

    @staticmethod
    def _resolve_documents(documents: Iterable[Document | str | Path]) -> list[Document]:
        """Documents as given; strings and paths are loaded (files or folders, lesson 03)."""
        from parsrag.loaders import load_directory, load_document

        docs: list[Document] = []
        for item in documents:
            if isinstance(item, Document):
                docs.append(item)
                continue
            p = Path(item)
            if p.is_dir():
                docs.extend(load_directory(p))
            elif p.is_file():
                docs.append(load_document(p))
            else:
                raise FileNotFoundError(f"no such file or folder: {p}")
        return docs

    def upsert_documents(
        self, documents: Iterable[Document | str | Path], *, prune: bool = False
    ) -> IndexUpdate:
        """Bring the index up to date with ``documents`` without redoing unchanged work.

        * a new id is chunked, embedded and indexed;
        * a known id whose content hash (:func:`document_hash`) changed has its
          old chunks removed and is indexed again;
        * a known id with the same hash is skipped — nothing is re-embedded;
        * with ``prune=True`` indexed documents that are *not* in ``documents``
          are removed (a folder sync).

        Strings and paths are loaded like in :meth:`add_documents`. TF-IDF style
        embeddings (with ``fit``) still refit on the whole collection when
        something changed — that is how they work.
        """
        docs = self._resolve_documents(documents)
        ids = [d.id for d in docs]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate document ids in the batch: {', '.join(dupes)}")
        report = IndexUpdate()
        changed: list[Document] = []
        for doc in docs:
            if doc.id not in self.documents:
                report.added.append(doc.id)
                changed.append(doc)
            elif self._hashes.get(doc.id) == document_hash(doc):
                report.unchanged.append(doc.id)
            else:
                report.updated.append(doc.id)
                changed.append(doc)
        stale = [i for i in self.documents if i not in set(ids)] if prune else []
        if report.updated or stale:
            self._check_can_delete()
        if stale:
            report.removed = stale
        report.chunks_removed = self._remove(report.updated + stale)
        report.chunks_added = self._index(changed) if changed else 0
        if not changed and report.chunks_removed:
            self.index_version += 1
        return report

    def remove_documents(self, ids: Iterable[str], *, missing_ok: bool = False) -> int:
        """Delete documents (and their chunks and vectors) by id. Returns the chunks removed.

        Unknown ids raise ``KeyError`` unless ``missing_ok``.
        """
        wanted = list(dict.fromkeys(ids))
        missing = [i for i in wanted if i not in self.documents]
        if missing and not missing_ok:
            raise KeyError(f"documents not in the index: {', '.join(missing)}")
        present = [i for i in wanted if i in self.documents]
        if not present:
            return 0
        self._check_can_delete()
        removed = self._remove(present)
        self.index_version += 1
        return removed

    def remove_document(self, doc_id: str) -> int:
        """Delete one document by id (``KeyError`` if it is not indexed)."""
        return self.remove_documents([doc_id])

    def _check_can_delete(self) -> None:
        store = self._store
        if store is not None and not self._fit_pending and not hasattr(store, "delete"):
            raise NotImplementedError(
                f"{type(self._store).__name__} cannot delete chunks; rebuild the index instead"
            )

    def _remove(self, doc_ids: Sequence[str]) -> int:
        """Drop documents and their chunks from every index (no version bump)."""
        if not doc_ids:
            return 0
        gone_docs = set(doc_ids)
        gone = [c.id for c in self._chunks if c.doc_id in gone_docs]
        self._chunks = [c for c in self._chunks if c.doc_id not in gone_docs]
        for doc_id in gone_docs:
            self.documents.pop(doc_id, None)
            self._hashes.pop(doc_id, None)
        self.bm25.remove(gone)
        if self._fit_pending:
            if self._chunks:
                self.embedding.fit([c.text for c in self._chunks])  # type: ignore[attr-defined]
                self.store = InMemoryVectorStore(self.embedding)
                self.store.add(self._chunks)
            else:
                self._store = None
        elif self._store is not None and gone:
            self._store.delete(gone)  # type: ignore[attr-defined]
        return len(gone)

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
            self._hashes[d.id] = document_hash(d)
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
    def hybrid(self) -> HybridRetriever:
        """RRF over BM25 (light stemming) and dense retrieval — the first stage."""
        return HybridRetriever([self.bm25, DenseRetriever(self.store, preprocess=self.preprocess)])

    @property
    def retriever(self) -> Retriever:
        """:attr:`hybrid`, reranked when a reranker is set, screened when a guard is set."""
        retriever: Retriever = self.hybrid
        if self.reranker is not None:
            retriever = RerankingRetriever(retriever, self.reranker, fetch_k=self.fetch_k)
        if self.guard is not None and self.guard.detect:
            retriever = self.guard.wrap(retriever)
        return retriever

    @property
    def generator(self) -> GroundedGenerator:
        """The lesson 09 generator over :attr:`retriever` (built on demand)."""
        return GroundedGenerator(
            self.retriever,
            self.llm,
            k=self.k,
            min_score=self.min_score,
            confidence=retriever_confidence(self.bm25),
            guard=self.guard,
        )

    def _check_index(self) -> None:
        if not self._chunks:
            raise EmptyIndexError(
                "the index is empty — add documents first / هنوز سندی اضافه نشده است"
            )

    def resolve_filter(self, filters: Filter | None) -> Filter | None:
        """Turn a metadata filter into one every store understands.

        A dict may mix chunk keys (:data:`CHUNK_FILTER_KEYS`: ``doc_id``,
        ``section``, ``page``, ...) with document keys (``title`` or anything in
        ``Document.metadata``, e.g. ``format`` or your own ``lang``/``tags``).
        Document conditions are evaluated here, on :attr:`documents`, and become a
        ``doc_id`` list, so they also work with database stores. Values are
        equality, a list means "any of these", and a list-valued metadata field
        (``tags``) matches when it shares a value. Callables pass through unchanged.
        """
        if filters is None or callable(filters):
            return filters
        chunk_part = {k: v for k, v in filters.items() if k in CHUNK_FILTER_KEYS}
        doc_part = {k: v for k, v in filters.items() if k not in CHUNK_FILTER_KEYS}
        if not doc_part:
            return chunk_part or None
        allowed = [d.id for d in self.documents.values() if _document_matches(d, doc_part)]
        if "doc_id" in chunk_part:
            wanted = _as_set(chunk_part["doc_id"])
            allowed = [i for i in allowed if i in wanted]
        chunk_part["doc_id"] = allowed
        return chunk_part

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        filter: Filter | None = None,
        filters: Filter | None = None,
    ) -> list[SearchResult]:
        """Hybrid + reranked search, optionally restricted by metadata (:meth:`resolve_filter`)."""
        self._check_index()
        flt = self.resolve_filter(filters if filters is not None else filter)
        return self.retriever.search(query, k or self.k, filter=flt)

    def ask(
        self,
        question: str,
        *,
        timings: Timings | None = None,
        filters: Filter | None = None,
    ) -> GroundedAnswer:
        """Answer with citations, or «نمی‌دانم». Pass ``timings`` to get per-stage latency.

        ``filters`` restricts retrieval to matching documents or chunks, e.g.
        ``{"doc_id": ["faq", "terms"]}`` or ``{"lang": "fa", "tags": "billing"}``
        (:meth:`resolve_filter`). If nothing matches, the answer abstains.
        """
        self._check_index()
        t0 = time.perf_counter()
        gen = self.generator
        results = gen.retriever.search(question, gen.k, filter=self.resolve_filter(filters))
        t1 = time.perf_counter()
        answer = gen.answer_from(question, results)
        t2 = time.perf_counter()
        if timings is not None:
            timings.retrieve_ms = (t1 - t0) * 1000
            timings.generate_ms = (t2 - t1) * 1000
            timings.total_ms = (t2 - t0) * 1000
            timings.llm_calls = 0 if answer.abstain_reason in ("no_results", "low_score") else 1
        return answer

    def stream(self, question: str, *, filters: Filter | None = None) -> Iterator[StreamEvent]:
        """Like :meth:`ask`, but yields ``sources``, then answer ``token``\\s, then ``done``.

        ``done`` carries the final :class:`GroundedAnswer` (citations checked
        on the full text). An early abstention yields ``sources`` and ``done`` only.
        With a guard that checks output (lesson 14), the reply is generated in
        full and checked *before* any token is sent, so an unsafe reply never
        reaches the user (time to first token = whole generation).
        """
        self._check_index()
        gen = self.generator
        results = gen.retriever.search(question, gen.k, filter=self.resolve_filter(filters))
        yield StreamEvent(type="sources", data=results)
        early, top = gen.precheck(question, results)
        if early is not None:
            yield StreamEvent(type="done", data=early)
            return
        if self.guard is not None and self.guard.check_output:
            raw = "".join(stream_text(self.llm, gen.messages(question, results)))
            final = gen.finalize(question, results, raw, top)
            for piece in split_for_streaming(final.answer):
                yield StreamEvent(type="token", data=piece)
            yield StreamEvent(type="done", data=final)
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
        """Write the whole index to ``directory`` (format :data:`INDEX_FORMAT_VERSION`).

        ============== ======================================================
        ``rag.json``   format version, options, content hashes, configuration
        ``documents``  ``documents.jsonl`` — the original documents
        ``chunks``     ``chunks.jsonl`` — chunks exactly as indexed
        ``bm25.json``  the tokenised BM25 index (no re-tokenising on load)
        ``store/``     vectors of an in-memory or FAISS store
        ============== ======================================================

        The folder is written next to the target and swapped in at the end, so
        an interrupted save never leaves a half-written index behind.
        """
        from parsrag import __version__

        target = Path(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex[:8]}")
        tmp.mkdir()
        try:
            with (tmp / "documents.jsonl").open("w", encoding="utf-8") as f:
                for doc in self.documents.values():
                    f.write(doc.model_dump_json() + "\n")
            with (tmp / "chunks.jsonl").open("w", encoding="utf-8") as f:
                for chunk in self._chunks:
                    f.write(chunk.model_dump_json() + "\n")
            (tmp / "bm25.json").write_text(
                json.dumps(self.bm25.state(), ensure_ascii=False), encoding="utf-8"
            )
            store_kind = None
            if self._store is not None and _store_kind(self._store) is not None:
                self._store.save(tmp / "store")  # type: ignore[attr-defined]
                store_kind = _store_kind(self._store)
            options = {
                "format_version": INDEX_FORMAT_VERSION,
                "parsrag_version": __version__,
                "k": self.k,
                "fetch_k": self.fetch_k,
                "min_score": self.min_score,
                "heading_context": self.heading_context,
                "normalize_text": self.preprocess is not None,
                "stem": self.stem,
                "embedding": type(self.embedding).__name__,
                "store": store_kind,
                "n_chunks": len(self._chunks),
                "index_version": self.index_version,
                "hashes": dict(self._hashes),
                "config": self.config.model_dump(mode="json"),
            }
            (tmp / "rag.json").write_text(
                json.dumps(options, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            old = None
            if target.exists():
                old = target.with_name(f".{target.name}.old-{uuid.uuid4().hex[:8]}")
                target.rename(old)
            tmp.rename(target)
            if old is not None:
                shutil.rmtree(old, ignore_errors=True)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def index_info(directory: str | Path) -> dict[str, Any]:
        """The ``rag.json`` of a saved index (``format_version`` is 1 for 0.1.0 indexes)."""
        info = json.loads((Path(directory) / "rag.json").read_text("utf-8"))
        info.setdefault("format_version", 1)
        return dict(info)

    @classmethod
    def load(cls, directory: str | Path, **kwargs: Any) -> PersianRAG:
        """Load an index written by :meth:`save` — nothing is re-chunked or re-embedded.

        Without keyword arguments the saved configuration rebuilds every part
        (embedding, reranker, LLM, guard, chunker for documents added later), and
        the saved options (``k``, ``min_score``, ...) are restored. Keyword
        arguments override both, e.g. ``llm=`` for another model. The embedding
        must be the one the index was built with; a different embedding class
        raises ``ValueError``, and saved vectors whose dimension does not match
        are recomputed from the saved chunks.

        Passing ``chunker``, ``heading_context`` or ``normalize_text`` re-chunks the
        documents; if that gives other chunks, the index is rebuilt (re-embedded).
        Folders from parsrag 0.1.0 (format 1) still load; they are re-chunked
        from the documents with the given (or default) chunker.
        """
        path = Path(directory)
        options = cls.index_info(path)
        version = int(options["format_version"])
        if version == 1:
            return cls._load_v1(path, options, kwargs)
        if version > INDEX_FORMAT_VERSION:
            raise ValueError(
                f"{path} uses index format {version}, newer than this parsrag supports "
                f"({INDEX_FORMAT_VERSION}); upgrade parsrag or rebuild the index"
            )
        if "config" not in kwargs and options.get("config") is not None:
            kwargs["config"] = RAGConfig.model_validate(options["config"])
        cfg: RAGConfig = kwargs.get("config") or RAGConfig()
        user_embedding = "embedding" in kwargs
        rechunk = any(k in kwargs for k in ("chunker", "heading_context", "normalize_text"))
        for key in ("k", "fetch_k", "min_score", "heading_context", "normalize_text", "stem"):
            if key in options:
                kwargs.setdefault(key, options[key])
        for key, value in _config_kwargs(cfg, skip=kwargs).items():
            kwargs.setdefault(key, value)
        saved_embedding = options.get("embedding")
        if not user_embedding and saved_embedding != type(kwargs["embedding"]).__name__:
            raise ValueError(
                f"{path} was built with {saved_embedding}, but its configuration makes "
                f"{type(kwargs['embedding']).__name__}; pass embedding=... to load it"
            )
        user_store = kwargs.get("store")
        rag = cls(**kwargs)
        with (path / "documents.jsonl").open(encoding="utf-8") as f:
            docs = [Document.model_validate_json(line) for line in f if line.strip()]
        with (path / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
        if rechunk:  # the caller changed how documents are chunked: rebuild if it matters
            fresh = [
                c
                for d in docs
                for c in chunk_document(
                    d, rag.chunker, preprocess=rag.preprocess, heading_context=rag.heading_context
                )
            ]
            if [(c.id, c.text) for c in fresh] != [(c.id, c.text) for c in chunks]:
                rag.add_documents(docs)
                return rag
        rag.documents = {d.id: d for d in docs}
        saved_hashes = options.get("hashes") or {}
        rag._hashes = {d.id: saved_hashes.get(d.id) or document_hash(d) for d in docs}
        rag._chunks = chunks
        bm25_file = path / "bm25.json"
        if bm25_file.is_file() and options.get("stem") == rag.stem:
            rag.bm25.load_state(json.loads(bm25_file.read_text("utf-8")), chunks)
        else:
            rag.bm25.add(chunks)
        if chunks:
            rag._restore_vectors(path / "store", options.get("store"), chunks, user_store)
        rag.index_version = int(options.get("index_version") or 1)
        return rag

    def _restore_vectors(
        self, store_dir: Path, kind: str | None, chunks: list[Chunk], user_store: Any
    ) -> None:
        if self._fit_pending:  # TF-IDF style: refit on the saved chunks (deterministic)
            self.embedding.fit([c.text for c in chunks])  # type: ignore[attr-defined]
            self.store = InMemoryVectorStore(self.embedding)
            self.store.add(chunks)
            return
        store = self.store
        if len(store):
            if len(store) != len(chunks):
                raise ValueError(
                    f"the given store holds {len(store)} chunks, the index {len(chunks)}"
                )
            return  # a persistent store (e.g. a database) that is already filled
        loader = getattr(type(store), "load", None)
        if store_dir.is_dir() and kind is not None and kind == _store_kind(store) and loader:
            try:
                saved = loader(store_dir, self.embedding)
            except ValueError:  # e.g. another dimension: fall through and re-embed
                saved = None
            if saved is not None and [c.id for c in saved.chunks] == [c.id for c in chunks]:
                self.store = saved
                return
        store.add(chunks)

    @classmethod
    def _load_v1(cls, path: Path, options: dict[str, Any], kwargs: dict[str, Any]) -> PersianRAG:
        """Indexes saved by parsrag 0.1.0 (no format number, no chunks or BM25 state)."""
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
        rag._hashes = {d.id: document_hash(d) for d in docs}
        rag._chunks = chunks
        rag.store = saved
        rag.bm25.add(chunks)
        rag.index_version += 1
        return rag


def _store_kind(store: Any) -> str | None:
    """``"memory"`` / ``"faiss"`` for stores whose vectors live in the index folder."""
    if isinstance(store, InMemoryVectorStore):
        return "memory"
    if type(store).__name__ == "FaissVectorStore":
        return "faiss"
    return None


def _config_kwargs(cfg: RAGConfig, *, skip: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``PersianRAG`` keyword arguments described by ``cfg`` (parts in ``skip`` are not built)."""
    rr = cfg.retrieval
    kwargs: dict[str, Any] = {
        "k": rr.k,
        "fetch_k": rr.fetch_k,
        "stem": rr.stem,
        "heading_context": cfg.chunking.heading_context,
        "normalize_text": cfg.chunking.normalize,
        "min_score": rr.min_score,
        "config": cfg,
    }
    builders: dict[str, Callable[[], Any]] = {
        "embedding": lambda: make_embedding(cfg.embedding),
        "chunker": lambda: RecursiveChunker(cfg.chunking.size, cfg.chunking.overlap),
        "reranker": lambda: _make_reranker(rr.rerank, rr.cross_encoder),
        "guard": lambda: cfg.security.make_guard(),
        "llm": lambda: make_llm(cfg.llm),
    }
    for key, build in builders.items():
        if key not in (skip or {}):
            kwargs[key] = build()
    return kwargs


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
    security: list[str] = Field(default_factory=list)


def answer_payload(rag: PersianRAG, answer: GroundedAnswer) -> AnswerPayload:
    """Answer + citations, each with the exact source passage for display."""
    return AnswerPayload(
        question=answer.question,
        answer=answer.answer,
        text=answer.text,
        abstained=answer.abstained,
        abstain_reason=answer.abstain_reason,
        citations=[{**c.model_dump(), "passage": rag.passage(c)} for c in answer.citations],
        security=list(answer.security),
    )


__all__ = [
    "CHUNK_FILTER_KEYS",
    "INDEX_FORMAT_VERSION",
    "AnswerPayload",
    "EmptyIndexError",
    "IndexUpdate",
    "PersianRAG",
    "StreamEvent",
    "Timings",
    "answer_payload",
    "document_hash",
    "validate_qa",
]

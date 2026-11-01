"""The FastAPI application (lesson 12; ``pip install 'parsrag[api]'``).

========================= =====================================================
``GET  /health``          status, index size, cache statistics
``POST /ingest``          add documents given as JSON
``POST /ingest/files``    add uploaded files (TXT, MD, HTML, DOCX, PDF)
``GET  /documents``       list indexed documents; ``/documents/{id}`` = full text
``DELETE /documents/{id}`` remove a document (its chunks and vectors) from the index
``POST /ask``             answer + citations; optional ``filters`` on metadata
``POST /ask/stream``      the same, as Server-Sent Events: sources → token… → done
``GET  /``                the web UI of lesson 13
``GET  /static/fonts/…``   the bundled Vazirmatn web fonts (SIL OFL 1.1) for the UI
========================= =====================================================

Run it with ``parsrag serve`` or ``uvicorn parsrag.api.app:app``; the module
attribute ``app`` is created on first access from :class:`~parsrag.api.settings.Settings`.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from parsrag import __version__
from parsrag.api.cache import AnswerCache, cache_key
from parsrag.api.logs import get_logger, log_event, question_fingerprint
from parsrag.api.ratelimit import RateLimiter
from parsrag.api.settings import Settings
from parsrag.generate import GroundedAnswer
from parsrag.llm import split_for_streaming, stream_text
from parsrag.rag import EmptyIndexError, PersianRAG, answer_payload
from parsrag.types import Document

# The web fonts shipped next to the UI, so it renders the same offline.
FONT_FILES = {
    "Vazirmatn-Regular.woff2": "font/woff2",
    "Vazirmatn-Medium.woff2": "font/woff2",
    "Vazirmatn-Bold.woff2": "font/woff2",
    "OFL.txt": "text/plain; charset=utf-8",
}

# ------------------------------------------------------------------- models


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)
    #: ``True``: re-index changed documents and skip unchanged ones instead of 409.
    upsert: bool = False


class IngestResponse(BaseModel):
    added_documents: list[str]
    added_chunks: int
    documents: int
    chunks: int
    index_version: int
    updated_documents: list[str] = Field(default_factory=list)
    unchanged_documents: list[str] = Field(default_factory=list)


FilterValue = str | int | float | bool | None | list[str | int | float | bool | None]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    #: Metadata filter, e.g. ``{"doc_id": ["faq"]}`` or ``{"format": "pdf"}`` — see
    #: :meth:`parsrag.PersianRAG.resolve_filter`.
    filters: dict[str, FilterValue] | None = Field(default=None, max_length=20)


class StageTimings(BaseModel):
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


class Citation(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    section: str | None = None
    page: int | None = None
    start: int | None = None
    end: int | None = None
    score: float | None = None
    passage: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    text: str
    abstained: bool
    abstain_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    security: list[str] = Field(default_factory=list)
    cached: bool = False
    timings: StageTimings = Field(default_factory=StageTimings)


class ErrorResponse(BaseModel):
    error: str
    message: str


class APIError(Exception):
    def __init__(self, status: int, error: str, message: str, **headers: str) -> None:
        self.status, self.error, self.message, self.headers = status, error, message, headers


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "Timeout" in type(exc).__name__


# ---------------------------------------------------------------- the service


class RAGService:
    """The parts of the API that do not depend on HTTP: locking, caching, timeouts.

    Retrieval runs under a lock (indexing mutates the BM25 index and the
    store); the LLM call does not, so slow generations do not block each other.
    """

    def __init__(
        self,
        rag: PersianRAG,
        settings: Settings,
        *,
        cache: AnswerCache | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.rag = rag
        self.settings = settings
        self.cache = (
            cache if cache is not None else AnswerCache(settings.cache_size, settings.cache_ttl)
        )
        self.limiter = limiter if limiter is not None else RateLimiter(settings.rate_limit)
        self.lock = threading.RLock()
        self.logger = get_logger(settings.log_level)

    def ingest(self, docs: list[Document], *, upsert: bool = False) -> IngestResponse:
        with self.lock:
            try:
                if upsert:
                    report = self.rag.upsert_documents(docs)
                    added_ids, added = report.added, report.chunks_added
                    updated, unchanged = report.updated, report.unchanged
                else:
                    added = self.rag.add_documents(docs)
                    added_ids, updated, unchanged = [d.id for d in docs], [], []
            except ValueError as exc:
                raise APIError(409, "duplicate_document", str(exc)) from exc
            return IngestResponse(
                added_documents=added_ids,
                added_chunks=added,
                documents=len(self.rag.documents),
                chunks=len(self.rag),
                index_version=self.rag.index_version,
                updated_documents=updated,
                unchanged_documents=unchanged,
            )

    def remove(self, doc_id: str) -> dict[str, Any]:
        with self.lock:
            if doc_id not in self.rag.documents:
                raise APIError(404, "not_found", f"no document {doc_id!r}")
            try:
                removed = self.rag.remove_document(doc_id)
            except NotImplementedError as exc:
                raise APIError(409, "not_supported", str(exc)) from exc
            return {
                "removed_document": doc_id,
                "removed_chunks": removed,
                "documents": len(self.rag.documents),
                "chunks": len(self.rag),
                "index_version": self.rag.index_version,
            }

    def retrieve(
        self, question: str, filters: dict[str, Any] | None = None
    ) -> tuple[Any, list[Any]]:
        with self.lock:
            try:
                self.rag._check_index()
            except EmptyIndexError as exc:
                raise APIError(409, "empty_index", str(exc)) from exc
            gen = self.rag.generator
            flt = self.rag.resolve_filter(filters)
            return gen, gen.retriever.search(question, gen.k, filter=flt)

    def _key(self, question: str, filters: dict[str, Any] | None) -> str:
        version: int | str = self.rag.index_version
        if filters:
            version = f"{version}|{json.dumps(filters, sort_keys=True, ensure_ascii=False)}"
        return cache_key(question, version)

    def to_response(self, answer: GroundedAnswer, timings: StageTimings) -> AskResponse:
        payload = answer_payload(self.rag, answer)
        return AskResponse(**payload.model_dump(), timings=timings)

    async def ask(self, question: str, filters: dict[str, Any] | None = None) -> AskResponse:
        key = self._key(question, filters)
        start = time.perf_counter()
        hit = self.cache.get(key)
        if hit is not None:
            total = (time.perf_counter() - start) * 1000
            response = hit.model_copy(
                update={"cached": True, "timings": StageTimings(total_ms=total)}
            )
            self._log(question, response)
            return response
        gen, results = await asyncio.to_thread(self.retrieve, question, filters)
        t1 = time.perf_counter()
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(gen.answer_from, question, results), self.settings.llm_timeout
            )
        except Exception as exc:
            raise self._llm_error(exc) from exc
        t2 = time.perf_counter()
        timings = StageTimings(
            retrieve_ms=(t1 - start) * 1000,
            generate_ms=(t2 - t1) * 1000,
            total_ms=(t2 - start) * 1000,
        )
        response = self.to_response(answer, timings)
        self.cache.set(key, response)
        self._log(question, response)
        return response

    def _llm_error(self, exc: BaseException) -> APIError:
        if isinstance(exc, APIError):
            return exc
        if _is_timeout(exc):
            return APIError(
                504,
                "llm_timeout",
                f"the language model did not answer within {self.settings.llm_timeout:g} s"
                " / مدل زبانی به‌موقع پاسخ نداد",
            )
        return APIError(502, "llm_error", f"the language model failed: {type(exc).__name__}")

    def _log(self, question: str, response: AskResponse, *, stream: bool = False) -> None:
        log_event(
            self.logger,
            "ask",
            question_id=question_fingerprint(question),
            question_chars=len(question),
            stream=stream,
            cached=response.cached,
            abstained=response.abstained,
            abstain_reason=response.abstain_reason,
            citations=len(response.citations),
            **{k: round(v, 2) for k, v in response.timings.model_dump().items()},
        )

    async def stream(
        self, question: str, filters: dict[str, Any] | None = None
    ) -> AsyncIterator[str]:
        """Server-Sent Events: ``sources``, then ``token`` events, then ``done`` (or ``error``)."""
        start = time.perf_counter()
        key = self._key(question, filters)
        hit = self.cache.get(key)
        if hit is not None:
            yield sse("sources", [c.model_dump() for c in hit.citations])
            for piece in split_for_streaming(hit.answer):
                yield sse("token", {"text": piece})
            total = (time.perf_counter() - start) * 1000
            done = hit.model_copy(update={"cached": True, "timings": StageTimings(total_ms=total)})
            self._log(question, done, stream=True)
            yield sse("done", done.model_dump())
            return
        try:
            gen, results = await asyncio.to_thread(self.retrieve, question, filters)
        except APIError as exc:
            yield sse("error", {"error": exc.error, "message": exc.message})
            return
        t1 = time.perf_counter()
        yield sse(
            "sources",
            [
                {"number": i, "chunk_id": r.chunk.id, "doc_id": r.chunk.doc_id, "score": r.score}
                for i, r in enumerate(results, 1)
            ],
        )
        early, top = gen.precheck(question, results)
        guarded = gen.guard is not None and gen.guard.check_output
        if early is not None:
            answer = early
        elif guarded:
            # Lesson 14: check the whole reply before sending any of it.
            try:
                answer = await asyncio.wait_for(
                    asyncio.to_thread(gen.answer_from, question, results),
                    self.settings.llm_timeout,
                )
            except Exception as exc:
                err = self._llm_error(exc)
                yield sse("error", {"error": err.error, "message": err.message})
                return
            for piece in split_for_streaming(answer.answer):
                yield sse("token", {"text": piece})
        else:
            pieces: list[str] = []
            try:
                async for piece in _iterate_in_thread(
                    stream_text(gen.llm, gen.messages(question, results)), self.settings.llm_timeout
                ):
                    pieces.append(piece)
                    yield sse("token", {"text": piece})
            except Exception as exc:
                err = self._llm_error(exc)
                yield sse("error", {"error": err.error, "message": err.message})
                return
            answer = gen.finalize(question, results, "".join(pieces), top)
        t2 = time.perf_counter()
        timings = StageTimings(
            retrieve_ms=(t1 - start) * 1000,
            generate_ms=(t2 - t1) * 1000,
            total_ms=(t2 - start) * 1000,
        )
        response = self.to_response(answer, timings)
        self.cache.set(key, response)
        self._log(question, response, stream=True)
        yield sse("done", response.model_dump())


async def _iterate_in_thread(iterator: Any, timeout: float) -> AsyncIterator[str]:
    """Pull a blocking iterator in a worker thread, with a timeout per piece."""
    sentinel = object()
    while True:
        item = await asyncio.wait_for(asyncio.to_thread(next, iterator, sentinel), timeout)
        if item is sentinel:
            return
        yield item


def sse(event: str, data: Any) -> str:
    """One Server-Sent Event: ``event: <name>`` + a JSON ``data:`` line + a blank line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------ the app


def build_rag(settings: Settings) -> PersianRAG:
    """The RAG system described by the settings (config file, saved index, docs, database)."""
    from parsrag.config import load_config, make_embedding

    cfg = load_config(settings.config) if settings.config else load_config()
    if settings.llm_provider:
        cfg.llm.provider = settings.llm_provider
    if settings.llm_model:
        cfg.llm.model = settings.llm_model
    if settings.llm_base_url:
        cfg.llm.base_url = settings.llm_base_url
    cfg.llm.timeout = settings.llm_timeout
    if settings.security is not None:
        cfg.security.enabled = settings.security
    if settings.index and Path(settings.index, "rag.json").is_file():
        from parsrag.config import make_llm

        return PersianRAG.load(
            settings.index,
            embedding=make_embedding(cfg.embedding),
            llm=make_llm(cfg.llm),
            guard=cfg.security.make_guard(),
        )
    kwargs: dict[str, Any] = {}
    if settings.database_url:
        from parsrag.stores.pgvector_store import PgVectorStore

        embedding = make_embedding(cfg.embedding)
        store = PgVectorStore(embedding, settings.database_url, create=False)
        store.drop()  # the folder is the source of truth: rebuild the table on startup
        store.create_table()
        kwargs = {"embedding": embedding, "store": store}
    rag = PersianRAG.from_config(cfg, **kwargs)
    if settings.docs:
        rag.add_path(settings.docs)
    return rag


def create_app(
    rag: PersianRAG | None = None,
    settings: Settings | None = None,
    *,
    cache: AnswerCache | None = None,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    """Build the FastAPI app around ``rag`` (default: built from ``settings`` / the environment).

    ``cache`` and ``limiter`` replace the ones the settings describe (any
    objects with the same methods — lesson 12 plugs in your own).
    """
    settings = settings or Settings.from_env()
    rag = rag if rag is not None else build_rag(settings)
    service = RAGService(rag, settings, cache=cache, limiter=limiter)
    app = FastAPI(
        title="parsrag",
        version=__version__,
        description="Persian RAG with citations — Persian RAG Academy, lesson 12.",
    )
    app.state.service = service

    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            ErrorResponse(error=exc.error, message=exc.message).model_dump(),
            status_code=exc.status,
            headers=exc.headers,
        )

    @app.middleware("http")
    async def access_log_and_rate_limit(request: Request, call_next: Any) -> Any:
        start = time.perf_counter()
        client = request.headers.get("x-api-key") or (
            request.client.host if request.client else "unknown"
        )
        limited = request.method in ("POST", "DELETE")
        if limited:
            decision = service.limiter.check(client)
            if not decision.allowed:
                retry = max(1, round(decision.retry_after))
                log_event(service.logger, "rate_limited", path=request.url.path, retry_after=retry)
                return JSONResponse(
                    ErrorResponse(
                        error="rate_limited",
                        message=f"too many requests, retry in {retry} s / درخواست‌ها زیاد است",
                    ).model_dump(),
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                )
        response = await call_next(request)
        log_event(
            service.logger,
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        return response

    error_responses: dict[int | str, dict[str, Any]] = {
        409: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    }

    @app.get("/health")
    def health() -> dict[str, Any]:
        rag = service.rag
        return {
            "status": "ok",
            "version": __version__,
            "documents": len(rag.documents),
            "chunks": len(rag),
            "index_version": rag.index_version,
            "llm": type(rag.llm).__name__,
            "cache": service.cache.stats(),
        }

    @app.post("/ingest", response_model=IngestResponse, responses=error_responses)
    async def ingest(body: IngestRequest) -> IngestResponse:
        docs = [Document(**d.model_dump()) for d in body.documents]
        return await asyncio.to_thread(service.ingest, docs, upsert=body.upsert)

    @app.post("/ingest/files", response_model=IngestResponse, responses=error_responses)
    async def ingest_files(files: list[UploadFile] = File(...)) -> IngestResponse:  # noqa: B008
        from parsrag.loaders import LOADERS, load_document

        limit = int(settings.max_upload_mb * 1024 * 1024)
        docs = []
        with tempfile.TemporaryDirectory() as tmp:
            for upload in files:
                name = Path(upload.filename or "upload.txt").name
                if Path(name).suffix.lower() not in LOADERS:
                    supported = ", ".join(sorted(LOADERS))
                    raise APIError(415, "unsupported_file", f"{name}: supported types {supported}")
                data = await upload.read(limit + 1)
                if len(data) > limit:
                    raise APIError(413, "file_too_large", f"{name} is larger than {limit} bytes")
                path = Path(tmp) / name
                path.write_bytes(data)
                try:
                    docs.append(await asyncio.to_thread(load_document, path))
                except ImportError as exc:
                    raise APIError(415, "missing_extra", str(exc)) from exc
        return await asyncio.to_thread(service.ingest, docs)

    @app.get("/documents")
    def documents() -> list[dict[str, Any]]:
        return [
            {"id": d.id, "title": d.title, "chars": len(d.text)}
            for d in service.rag.documents.values()
        ]

    @app.get("/documents/{doc_id}", responses={404: {"model": ErrorResponse}})
    def document(doc_id: str) -> dict[str, Any]:
        doc = service.rag.documents.get(doc_id)
        if doc is None:
            raise APIError(404, "not_found", f"no document {doc_id!r}")
        return {"id": doc.id, "title": doc.title, "text": doc.text}

    @app.delete("/documents/{doc_id}", responses={404: {"model": ErrorResponse}, **error_responses})
    async def delete_document(doc_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(service.remove, doc_id)

    @app.post("/ask", response_model=AskResponse, responses=error_responses)
    async def ask(body: AskRequest) -> AskResponse:
        return await service.ask(body.question, body.filters)

    @app.post("/ask/stream", responses={200: {"content": {"text/event-stream": {}}}})
    async def ask_stream(body: AskRequest) -> StreamingResponse:
        return StreamingResponse(
            service.stream(body.question, body.filters),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return resources.files("parsrag.api").joinpath("static/index.html").read_text("utf-8")

    @app.get(
        "/static/fonts/{name}", include_in_schema=False, responses={404: {"model": ErrorResponse}}
    )
    def font(name: str) -> Response:
        if name not in FONT_FILES:  # a fixed allow-list: no path traversal possible
            raise APIError(404, "not_found", f"no font {name!r}")
        data = resources.files("parsrag.api").joinpath("static/fonts").joinpath(name).read_bytes()
        return Response(
            data,
            media_type=FONT_FILES[name],
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    log_event(
        service.logger,
        "startup",
        documents=len(service.rag.documents),
        chunks=len(service.rag),
        llm=type(service.rag.llm).__name__,
    )
    return app


def __getattr__(name: str) -> Any:
    """``uvicorn parsrag.api.app:app`` — build the app from the environment on first access."""
    if name == "app":
        app = create_app()
        globals()["app"] = app
        return app
    raise AttributeError(name)


__all__ = [
    "AskRequest",
    "AskResponse",
    "IngestRequest",
    "IngestResponse",
    "RAGService",
    "Settings",
    "build_rag",
    "create_app",
    "sse",
]

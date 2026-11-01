"""The FastAPI app (lesson 12) with TestClient and FakeLLM — no network, no keys."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="needs the api extra")
pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient

from parsrag import PersianRAG
from parsrag.api import RateLimiter, Settings, create_app
from parsrag.api.app import build_rag, sse

PIK = "هزینهٔ پیک در تهران چقدر است؟"
OFF = Settings(rate_limit=0)


@pytest.fixture(scope="module")
def course() -> PersianRAG:
    return PersianRAG.course_default()


@pytest.fixture()
def client(course: PersianRAG) -> TestClient:
    return TestClient(create_app(course, OFF))


def events(body: str) -> list[tuple[str, Any]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_health(client: TestClient) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok" and data["documents"] == 25 and data["chunks"] > 100
    assert data["llm"] == "FakeLLM"


def test_ask_returns_answer_citations_and_timings(client: TestClient) -> None:
    r = client.post("/ask", json={"question": PIK})
    assert r.status_code == 200
    data = r.json()
    assert "200" in data["text"] and not data["abstained"] and not data["cached"]
    cite = data["citations"][0]
    assert cite["doc_id"] == "kahroba-shipping" and "پیک" in cite["passage"]
    t = data["timings"]
    assert t["total_ms"] >= t["retrieve_ms"] > 0


def test_cache_hit_on_normalised_question(client: TestClient) -> None:
    first = client.post("/ask", json={"question": PIK}).json()
    second = client.post("/ask", json={"question": "  هزينه پيک در تهران چقدر است"}).json()
    assert second["cached"] and second["text"] == first["text"]
    assert second["timings"]["retrieve_ms"] == 0
    assert client.get("/health").json()["cache"]["hits"] >= 1


def test_ingest_invalidates_cache_and_rejects_duplicates() -> None:
    app = create_app(PersianRAG(), OFF)
    c = TestClient(app)
    doc = {"id": "zoo", "text": "باغ‌وحش شهر از ساعت ۸ صبح تا ۶ عصر باز است.", "title": "باغ‌وحش"}
    r = c.post("/ingest", json={"documents": [doc]})
    assert r.status_code == 200 and r.json()["added_documents"] == ["zoo"]
    q = {"question": "باغ‌وحش تا چه ساعتی باز است؟"}
    assert not c.post("/ask", json=q).json()["cached"]
    assert c.post("/ask", json=q).json()["cached"]
    c.post("/ingest", json={"documents": [{"id": "z2", "text": "پارک تا ۱۰ شب باز است."}]})
    assert not c.post("/ask", json=q).json()["cached"]  # index_version changed
    dup = c.post("/ingest", json={"documents": [doc]})
    assert dup.status_code == 409 and dup.json()["error"] == "duplicate_document"
    assert c.get("/documents").json()[0]["id"] == "zoo"
    assert "باغ‌وحش" in c.get("/documents/zoo").json()["text"]
    assert c.get("/documents/nope").status_code == 404


def test_ingest_files(tmp_path: Path) -> None:
    c = TestClient(create_app(PersianRAG(), OFF))
    files = [
        ("files", ("menu.md", "# منو\nقیمت چای ۲۰ هزار تومان است.".encode(), "text/markdown")),
        ("files", ("note.txt", "کافه جمعه‌ها تعطیل است.".encode(), "text/plain")),
    ]
    r = c.post("/ingest/files", files=files)
    assert r.status_code == 200 and sorted(r.json()["added_documents"]) == ["menu", "note"]
    bad = c.post("/ingest/files", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 415 and bad.json()["error"] == "unsupported_file"
    small = TestClient(create_app(PersianRAG(), Settings(rate_limit=0, max_upload_mb=0.00001)))
    big = small.post(
        "/ingest/files", files=[("files", ("big.txt", "آ".encode() * 100, "text/plain"))]
    )
    assert big.status_code == 413


def test_empty_index_and_validation_errors() -> None:
    c = TestClient(create_app(PersianRAG(), OFF))
    r = c.post("/ask", json={"question": PIK})
    assert r.status_code == 409 and r.json()["error"] == "empty_index"
    assert c.post("/ask", json={"question": ""}).status_code == 422
    stream = c.post("/ask/stream", json={"question": PIK}).text
    assert events(stream) == [
        ("error", {"error": "empty_index", "message": events(stream)[0][1]["message"]})
    ]


class SlowLLM:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def generate(self, messages: Sequence[Any]) -> str:
        time.sleep(self.seconds)
        return "دیر [1]"

    def stream(self, messages: Sequence[Any]) -> Iterator[str]:
        time.sleep(self.seconds)
        yield "دیر"


class BrokenLLM:
    def generate(self, messages: Sequence[Any]) -> str:
        raise ConnectionError("provider down")


def test_provider_timeout_and_failure() -> None:
    rag = PersianRAG.course_default(llm=SlowLLM(0.5))
    c = TestClient(create_app(rag, Settings(rate_limit=0, llm_timeout=0.05)))
    r = c.post("/ask", json={"question": PIK})
    assert r.status_code == 504 and r.json()["error"] == "llm_timeout"
    stream = events(c.post("/ask/stream", json={"question": PIK}).text)
    assert stream[0][0] == "sources" and stream[-1] == ("error", stream[-1][1])
    assert stream[-1][1]["error"] == "llm_timeout"
    broken = TestClient(create_app(PersianRAG.course_default(llm=BrokenLLM()), OFF))
    r = broken.post("/ask", json={"question": PIK})
    assert r.status_code == 502 and r.json()["error"] == "llm_error"


def test_rate_limit_per_client(course: PersianRAG) -> None:
    c = TestClient(create_app(course, Settings(rate_limit=2)))
    q = {"question": PIK}
    codes = [c.post("/ask", json=q, headers={"X-API-Key": "a"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = c.post("/ask", json=q, headers={"X-API-Key": "a"})
    assert r.json()["error"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1
    assert c.post("/ask", json=q, headers={"X-API-Key": "b"}).status_code == 200
    assert c.get("/health").status_code == 200  # GETs are not limited


def test_injected_limiter(course: PersianRAG) -> None:
    c = TestClient(create_app(course, OFF, limiter=RateLimiter(1)))
    assert [c.post("/ask", json={"question": PIK}).status_code for _ in range(2)] == [200, 429]


def test_stream_events(client: TestClient) -> None:
    body = client.post("/ask/stream", json={"question": "جریمهٔ دیرکرد کتاب چقدر است؟"}).text
    evs = events(body)
    kinds = [k for k, _ in evs]
    assert kinds[0] == "sources" and kinds[-1] == "done" and kinds.count("token") >= 3
    text = "".join(d["text"] for k, d in evs if k == "token")
    done = evs[-1][1]
    assert done["answer"] == text.strip() and done["citations"]
    again = events(client.post("/ask/stream", json={"question": "جریمهٔ دیرکرد کتاب چقدر است"}).text)
    assert (
        again[-1][1]["cached"]
        and "".join(d["text"] for k, d in again if k == "token") == done["answer"]
    )


def test_stream_abstains_without_tokens(client: TestClient) -> None:
    evs = events(
        client.post("/ask/stream", json={"question": "قیمت بلیت هواپیما به استانبول؟"}).text
    )
    assert [k for k, _ in evs] == ["sources", "done"] and evs[-1][1]["abstained"]


def test_sse_format() -> None:
    assert sse("token", {"text": "سلام"}) == 'event: token\ndata: {"text": "سلام"}\n\n'


def test_ui_is_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and 'dir="rtl"' in r.text
    # Persian file picker instead of the browser's English "Choose Files / No file chosen"
    assert "انتخاب فایل" in r.text and "فایلی انتخاب نشده" in r.text
    assert 'class="visually-hidden" type="file"' in r.text  # still a real, focusable input


def test_ui_fonts_are_bundled(client: TestClient) -> None:
    html = client.get("/").text
    for name in ("Vazirmatn-Regular.woff2", "Vazirmatn-Medium.woff2", "Vazirmatn-Bold.woff2"):
        assert f"static/fonts/{name}" in html
        r = client.get(f"/static/fonts/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "font/woff2"
        assert r.content[:4] == b"wOF2"
    licence = client.get("/static/fonts/OFL.txt")
    assert licence.status_code == 200 and "SIL Open Font License" in licence.text
    assert "https://" not in html.split("<script>")[0]  # no external stylesheet or font
    assert client.get("/static/fonts/../app.py").status_code == 404
    assert client.get("/static/fonts/index.html").status_code == 404


def test_ui_translates_every_api_error_code(client: TestClient) -> None:
    """The API may answer in English; the UI maps every error code to Persian."""
    import re

    import parsrag.api.app as app_module

    source = Path(app_module.__file__).read_text(encoding="utf-8")
    codes = set(re.findall(r'APIError\(\s*\d+,\s*"(\w+)"', source))
    codes |= set(re.findall(r'error="(\w+)"', source))
    assert {"empty_index", "rate_limited", "unsupported_file"} <= codes
    html = client.get("/").text
    table = html[html.index("const ERRORS = {") : html.index("const STATUS_ERRORS")]
    for code in codes:
        assert f"  {code}: " in table, code
    # the header no longer shows the index status line
    assert 'id="health"' not in html


def test_logs_are_json_with_stage_latency(client: TestClient) -> None:
    import io
    import logging

    from parsrag.api.logs import LOGGER, JsonFormatter

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    logging.getLogger(LOGGER).addHandler(handler)
    try:
        client.post("/ask", json={"question": "ساعت کاری کتابخانه چیست؟"})
    finally:
        logging.getLogger(LOGGER).removeHandler(handler)
    lines = [json.loads(line) for line in buffer.getvalue().splitlines()]
    ask = next(e for e in lines if e["event"] == "ask")
    assert {"retrieve_ms", "generate_ms", "total_ms", "cached", "question_id"} <= set(ask)
    assert "question" not in ask  # the text itself is not logged
    assert any(e["event"] == "request" and e["path"] == "/ask" for e in lines)


def test_build_rag_from_settings(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# الف\nکتابخانه ساعت ۸ صبح باز می‌شود.", encoding="utf-8")
    rag = build_rag(Settings(docs=str(docs)))
    assert set(rag.documents) == {"a"}
    rag.save(tmp_path / "idx")
    loaded = build_rag(Settings(index=str(tmp_path / "idx"), llm_provider="fake"))
    assert len(loaded) == len(rag)
    real = build_rag(Settings(docs=str(docs), llm_provider="ollama", llm_model="qwen2.5:7b"))
    assert type(real.llm).__name__ == "OpenAICompatibleLLM"


def test_ask_with_filters_and_cache_keys(course: PersianRAG) -> None:
    client = TestClient(create_app(course, OFF))
    r = client.post("/ask", json={"question": PIK, "filters": {"doc_id": "mehregan-rules"}})
    assert r.status_code == 200
    data = r.json()
    assert all(c["doc_id"] == "mehregan-rules" for c in data["citations"])
    assert not data["cached"]
    plain = client.post("/ask", json={"question": PIK}).json()
    assert not plain["cached"]  # another filter is another cache entry
    assert plain["citations"][0]["doc_id"] == "kahroba-shipping"
    again = client.post("/ask", json={"question": PIK, "filters": {"doc_id": "mehregan-rules"}})
    assert again.json()["cached"]
    stream = client.post("/ask/stream", json={"question": PIK, "filters": {"doc_id": ["x"]}})
    kinds = events(stream.text)
    assert kinds[0] == ("sources", []) and kinds[-1][1]["abstained"]
    bad = client.post("/ask", json={"question": PIK, "filters": {"doc_id": {"nested": 1}}})
    assert bad.status_code == 422


def test_delete_document_and_upsert_ingest() -> None:
    rag = PersianRAG()
    client = TestClient(create_app(rag, OFF))
    doc = {"id": "pool", "text": "استخر محله دوشنبه‌ها تعطیل است.", "metadata": {"lang": "fa"}}
    assert client.post("/ingest", json={"documents": [doc]}).status_code == 200
    assert client.post("/ingest", json={"documents": [doc]}).status_code == 409
    r = client.post("/ingest", json={"documents": [doc], "upsert": True}).json()
    assert r["unchanged_documents"] == ["pool"] and r["added_chunks"] == 0
    changed = {**doc, "text": "استخر محله سه‌شنبه‌ها تعطیل است."}
    r = client.post("/ingest", json={"documents": [changed], "upsert": True}).json()
    assert r["updated_documents"] == ["pool"] and r["added_chunks"] == 1
    gone = client.delete("/documents/pool")
    assert gone.status_code == 200 and gone.json()["removed_chunks"] == 1
    assert client.delete("/documents/pool").status_code == 404
    assert client.post("/ask", json={"question": "استخر کی تعطیل است؟"}).status_code == 409

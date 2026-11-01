from __future__ import annotations

import sys
from typing import Any

import pytest

from parsrag.llm import LLM, FakeLLM, Message, OpenAICompatibleLLM, to_messages

USER = [{"role": "user", "content": "سلام"}]


def test_to_messages_accepts_dicts_and_models() -> None:
    msgs = to_messages([Message(role="system", content="s"), {"role": "user", "content": "u"}])
    assert [m.role for m in msgs] == ["system", "user"]


def test_to_messages_rejects_bad_role() -> None:
    with pytest.raises(ValueError):
        to_messages([{"role": "robot", "content": "x"}])


def test_fake_llm_satisfies_protocol() -> None:
    assert isinstance(FakeLLM(), LLM)
    assert isinstance(OpenAICompatibleLLM(model="m", client=object()), LLM)


def test_fake_llm_echo() -> None:
    assert FakeLLM().generate(USER) == "[FakeLLM] سلام"
    assert FakeLLM(prefix="").generate(USER) == "سلام"


def test_fake_llm_canned_responses_cycle_and_record_calls() -> None:
    llm = FakeLLM(responses=["a", "b"])
    assert [llm.generate(USER) for _ in range(3)] == ["a", "b", "a"]
    assert len(llm.calls) == 3
    assert llm.calls[0][0].content == "سلام"


def test_fake_llm_rejects_empty_responses() -> None:
    with pytest.raises(ValueError):
        FakeLLM(responses=[])


def test_fake_llm_handler_wins() -> None:
    llm = FakeLLM(responses=["x"], handler=lambda msgs: msgs[-1].content.upper())
    assert llm.generate([{"role": "user", "content": "abc"}]) == "ABC"


def test_fake_llm_extractive_picks_best_sentence() -> None:
    prompt = (
        "کتابخانه جمعه‌ها تعطیل است. هزینهٔ پیک در تهران ۲۰۰ هزار تومان است.\n\n"
        "پرسش: هزینهٔ پیک چقدر است؟"
    )
    llm = FakeLLM(mode="extractive")
    assert llm.generate([{"role": "user", "content": prompt}]) == (
        "هزینهٔ پیک در تهران ۲۰۰ هزار تومان است."
    )


def test_fake_llm_extractive_falls_back_when_nothing_matches() -> None:
    llm = FakeLLM(mode="extractive", fallback="نمی‌دانم.")
    prompt = "باران می‌بارد.\n\nپرسش: قیمت طلا؟"
    assert llm.generate([{"role": "user", "content": prompt}]) == "نمی‌دانم."


# ------------------------------------------------------------ OpenAICompatibleLLM


class FakeResponse:
    def __init__(self, data: Any, status: int = 200) -> None:
        self._data, self.status = data, status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self) -> Any:
        return self._data


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})
        return self.response


def _ok(content: str) -> FakeResponse:
    return FakeResponse({"choices": [{"message": {"role": "assistant", "content": content}}]})


def test_openai_compatible_sends_chat_completion_request() -> None:
    client = FakeClient(_ok("پاسخ"))
    llm = OpenAICompatibleLLM(
        model="my-model",
        base_url="http://localhost:11434/v1/",
        api_key="secret",
        max_tokens=64,
        client=client,
    )
    assert llm.generate(USER) == "پاسخ"
    (req,) = client.requests
    assert req["url"] == "http://localhost:11434/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer secret"
    assert req["json"] == {
        "model": "my-model",
        "messages": [{"role": "user", "content": "سلام"}],
        "temperature": 0.0,
        "max_tokens": 64,
    }


def test_openai_compatible_without_key_sends_no_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = FakeClient(_ok("ok"))
    OpenAICompatibleLLM(model="m", client=client).generate(USER)
    assert "Authorization" not in client.requests[0]["headers"]


def test_openai_compatible_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert OpenAICompatibleLLM(model="m", client=object()).api_key == "from-env"


def test_openai_compatible_raises_on_http_error() -> None:
    llm = OpenAICompatibleLLM(model="m", client=FakeClient(FakeResponse({}, status=500)))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        llm.generate(USER)


def test_openai_compatible_raises_on_malformed_response() -> None:
    llm = OpenAICompatibleLLM(model="m", client=FakeClient(FakeResponse({"oops": 1})))
    with pytest.raises(RuntimeError, match="Unexpected response"):
        llm.generate(USER)


def test_openai_compatible_lazy_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "httpx", None)  # makes `import httpx` fail
    llm = OpenAICompatibleLLM(model="m")  # constructing never imports httpx
    with pytest.raises(ImportError, match=r"parsrag\[openai\]"):
        llm.generate(USER)


def test_package_import_does_not_pull_optional_deps() -> None:
    import subprocess

    code = (
        "import sys, parsrag, parsrag.llm, parsrag.embeddings, parsrag.normalize,"
        " parsrag.retrieval,"
        " parsrag.rerank, parsrag.generate, parsrag.eval, parsrag.rag, parsrag.config,"
        " parsrag.agent;"
        "bad = {'httpx', 'sentence_transformers', 'torch', 'faiss', 'openai', 'langgraph',"
        " 'langchain_core', 'fastapi', 'uvicorn'} & set(sys.modules);"
        "print(sorted(bad)); raise SystemExit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_fake_llm_extractive_cites_numbered_blocks() -> None:
    prompt = (
        "متن‌ها:\n[1] shop › ارسال\nهزینهٔ پیک ۲۰۰ هزار تومان است.\n\n"
        "[2] library\nکتابخانه تا ۸ شب باز است.\n\nپرسش: کتابخانه تا کی باز است؟"
    )
    cited = FakeLLM(mode="extractive", cite=True)
    assert cited.generate([{"role": "user", "content": prompt}]) == "کتابخانه تا ۸ شب باز است. [2]"
    plain = FakeLLM(mode="extractive")
    assert plain.generate([{"role": "user", "content": prompt}]) == "کتابخانه تا ۸ شب باز است."


def test_fake_llm_extractive_min_overlap_and_ignore() -> None:
    prompt = "[1] d\nکتابخانه تا ۸ شب باز است.\n\nپرسش: پارکینگ کتابخانه کجا است؟"
    msgs = [{"role": "user", "content": prompt}]
    assert FakeLLM(mode="extractive").generate(msgs) == "کتابخانه تا ۸ شب باز است."
    strict = FakeLLM(mode="extractive", min_overlap=2)
    assert strict.generate(msgs) == "کتابخانه تا ۸ شب باز است."  # «کتابخانه» + «است»
    no_stop = FakeLLM(mode="extractive", min_overlap=2, ignore={"است"})
    assert no_stop.generate(msgs) == "نمی‌دانم."

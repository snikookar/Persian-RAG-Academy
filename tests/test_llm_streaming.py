"""Streaming helpers and CountingLLM (lessons 11–12)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from parsrag.llm import CountingLLM, FakeLLM, OpenAICompatibleLLM, split_for_streaming, stream_text

USER = [{"role": "user", "content": "سلام"}]


def test_split_for_streaming_round_trips() -> None:
    text = "هزینهٔ پیک  ۲۰۰ هزار تومان است [1].\nخط دوم"
    pieces = split_for_streaming(text)
    assert "".join(pieces) == text
    assert len(pieces) == 9
    assert split_for_streaming("") == []


def test_fake_llm_stream_matches_generate() -> None:
    llm = FakeLLM(["یک دو سه"])
    assert list(llm.stream(USER)) == ["یک ", "دو ", "سه"]
    assert len(llm.calls) == 1


def test_stream_text_falls_back_to_generate() -> None:
    class NoStream:
        def generate(self, messages: Any) -> str:
            return "الف ب"

    assert list(stream_text(NoStream(), USER)) == ["الف ", "ب"]


def test_counting_llm_counts_generate_and_stream() -> None:
    llm = CountingLLM(FakeLLM(["آری"]))
    llm.generate(USER)
    assert "".join(stream_text(llm, USER)) == "آری"
    assert llm.calls == 2 and llm.seconds >= 0
    llm.reset()
    assert llm.calls == 0


class StreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __enter__(self) -> StreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self) -> Iterator[str]:
        yield from self.lines


class StreamClient:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.requests: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, **kwargs: Any) -> StreamResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        return StreamResponse(self.lines)


def _chunk(text: str) -> str:
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]})


def test_openai_compatible_stream_parses_sse() -> None:
    lines = [": keep-alive", _chunk("پاسخ "), "", _chunk("کوتاه"), "data: [DONE]", _chunk("x")]
    client = StreamClient(lines)
    llm = OpenAICompatibleLLM(model="m", base_url="http://x/v1", api_key="k", client=client)
    assert list(llm.stream(USER)) == ["پاسخ ", "کوتاه"]
    (req,) = client.requests
    assert req["method"] == "POST" and req["json"]["stream"] is True


def test_openai_compatible_stream_rejects_garbage() -> None:
    llm = OpenAICompatibleLLM(model="m", client=StreamClient(["data: {not json"]))
    with pytest.raises(RuntimeError, match="Unexpected stream chunk"):
        list(llm.stream(USER))

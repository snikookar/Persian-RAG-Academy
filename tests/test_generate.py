from __future__ import annotations

from typing import Any

import pytest

from parsrag import FakeLLM
from parsrag.generate import (
    GroundedGenerator,
    calibrate_threshold,
    retriever_confidence,
    strip_citations,
    top_score,
)
from parsrag.llm import OpenAICompatibleLLM
from parsrag.retrieval import BM25Retriever
from parsrag.types import Chunk, SearchResult


def chunks() -> list[Chunk]:
    return [
        Chunk(id="a#0", doc_id="a", text="هزینهٔ پیک در تهران ۲۰۰ هزار تومان است.",
              metadata={"section": "ارسال", "start": 0, "end": 40}),
        Chunk(id="b#0", doc_id="b", text="کتابخانه تا ۸ شب باز است."),
    ]  # fmt: skip


def test_strip_citations_and_top_score() -> None:
    assert strip_citations("۲۰۰ هزار تومان [1] است [۲، 3].") == "۲۰۰ هزار تومان است ."
    assert top_score("q", []) == float("-inf")
    assert top_score("q", [SearchResult(chunk=chunks()[0], score=2.5)]) == 2.5


def test_generator_with_bm25_confidence() -> None:
    bm25 = BM25Retriever(chunks())
    gen = GroundedGenerator(
        bm25, FakeLLM(mode="extractive", cite=True), k=2, min_score=0.1,
        confidence=retriever_confidence(bm25),
    )  # fmt: skip
    ans = gen.answer("هزینهٔ پیک در تهران چقدر است؟")
    assert ans.citations[0].section == "ارسال" and ans.citations[0].end == 40
    assert gen.answer("پارکینگ").abstain_reason == "no_results"
    with pytest.raises(ValueError):
        GroundedGenerator(bm25, FakeLLM(), k=0)


def test_calibration_validates_lengths() -> None:
    with pytest.raises(ValueError):
        calibrate_threshold([1.0], [])


class FakeClient:
    """Stands in for httpx.Client so a 'real' model can be plugged in without network."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.payloads.append(kwargs["json"])

        class Response:
            def raise_for_status(self) -> None: ...

            def json(self) -> Any:
                return {"choices": [{"message": {"content": "۲۰۰ هزار تومان [1]"}}]}

        return Response()


def test_openai_compatible_model_plugs_in() -> None:
    client = FakeClient()
    llm = OpenAICompatibleLLM("any-model", base_url="http://localhost:11434/v1", client=client)
    ans = GroundedGenerator(BM25Retriever(chunks()), llm).answer("هزینهٔ پیک چقدر است؟")
    assert ans.citations[0].doc_id == "a" and ans.text == "۲۰۰ هزار تومان"
    system, user = client.payloads[0]["messages"]
    assert "نمی‌دانم" in system["content"] and user["content"].startswith("متن‌ها:\n[1] a")

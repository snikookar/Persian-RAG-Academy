"""API latency: cache misses vs hits, normalised cache keys, streaming (lesson 12).

    python examples/api_latency.py                       # offline, FakeLLM
    python examples/api_latency.py --token-delay 0.03    # + a *simulated* slow model

Requests go through the real FastAPI app with TestClient (in process, no
network), so the numbers are the server's own overhead on this machine:
retrieval, prompt building, JSON, logging. A real model adds its own latency
to every cache miss — measure it with your model (lesson 13).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import time
from collections.abc import Iterator, Sequence
from typing import Any

from fastapi.testclient import TestClient

from parsrag import FakeLLM, PersianRAG
from parsrag.api import AnswerCache, Settings, create_app
from parsrag.api.cache import normalize_question
from parsrag.api.logs import LOGGER
from parsrag.eval.benchmark import load_fixture_benchmark
from parsrag.retrieval import PERSIAN_STOPWORDS

ARABIC = str.maketrans({"ی": "ي", "ک": "ك"})


def variant(question: str) -> str:
    """The same question typed differently: Arabic ي/ك, no «؟», extra spaces."""
    return "  " + question.translate(ARABIC).replace("؟", "").replace(" ", "  ") + " "


class SlowStreamingLLM:
    """A *simulated* model: the extractive FakeLLM, ``delay`` seconds per word."""

    def __init__(self, delay: float) -> None:
        self.inner = FakeLLM(mode="extractive", cite=True, ignore=PERSIAN_STOPWORDS)
        self.delay = delay

    def generate(self, messages: Sequence[Any]) -> str:
        reply = self.inner.generate(messages)
        time.sleep(self.delay * len(reply.split()))
        return reply

    def stream(self, messages: Sequence[Any]) -> Iterator[str]:
        for piece in self.inner.stream(messages):
            time.sleep(self.delay)
            yield piece


def ms(values: list[float]) -> str:
    q = statistics.quantiles(values, n=20) if len(values) > 1 else values * 19
    return f"{statistics.median(values):6.2f} | {q[18]:6.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--token-delay", type=float, default=0.0, help="simulated s per word")
    args = parser.parse_args()
    logging.getLogger(LOGGER).disabled = True  # keep the output readable

    llm = SlowStreamingLLM(args.token_delay) if args.token_delay else None
    rag = PersianRAG.course_default(**({"llm": llm} if llm else {}))
    questions = [q.question for f in ("qa.jsonl", "qa_paraphrase.jsonl")
                 for q in load_fixture_benchmark(qa_file=f)[1]]  # fmt: skip
    client = TestClient(create_app(rag, Settings(rate_limit=0)))

    def timed(question: str) -> tuple[float, dict[str, Any]]:
        start = time.perf_counter()
        data = client.post("/ask", json={"question": question}).json()
        return (time.perf_counter() - start) * 1000, data

    miss, hit, server_miss = [], [], []
    for q in questions:
        t, data = timed(q)
        miss.append(t)
        server_miss.append(data["timings"]["retrieve_ms"])
        t, data = timed(q)
        assert data["cached"]
        hit.append(t)
    label = f"simulated model, {args.token_delay * 1000:.0f} ms/word" if llm else "FakeLLM"
    print(f"### /ask over {len(questions)} questions ({label})\n")
    print("| request | median ms | p95 ms |")
    print("|---|---|---|")
    print(f"| cache miss (HTTP round trip) | {ms(miss)} |")
    print(f"| … of which retrieval (server timing) | {ms(server_miss)} |")
    print(f"| cache hit (same question again) | {ms(hit)} |")

    # How often does the same question, typed differently, hit the cache?
    raw = AnswerCache(len(questions) * 2)
    for q in questions:
        raw.set(q, True)
    raw_hits = sum(raw.get(variant(q)) is not None for q in questions)
    norm_hits = sum(normalize_question(variant(q)) == normalize_question(q) for q in questions)
    print("\n### Variants (Arabic ي/ك, no «؟», extra spaces) of every question\n")
    print("| cache key | hits |")
    print("|---|---|")
    print(f"| raw question text | {raw_hits}/{len(questions)} |")
    print(f"| normalised question (`cache_key`) | {norm_hits}/{len(questions)} |")

    if not llm:
        return
    # TestClient buffers streamed bodies, so time the server's SSE generator directly.
    service = create_app(rag, Settings(rate_limit=0)).state.service  # fresh, empty cache

    async def stream_times(question: str) -> tuple[float | None, float]:
        start, first = time.perf_counter(), None
        async for event in service.stream(question):
            if first is None and event.startswith("event: token"):
                first = (time.perf_counter() - start) * 1000
        return first, (time.perf_counter() - start) * 1000

    first, total = [], []
    for q in questions[:20]:
        seen_token, whole = asyncio.run(stream_times(q))
        if seen_token is not None:
            first.append(seen_token)
            total.append(whole)
    print(f"\n### /ask/stream, {len(first)} answered questions ({label}, server side)\n")
    print("| moment | median ms | p95 ms |")
    print("|---|---|---|")
    print(f"| first token | {ms(first)} |")
    print(f"| whole answer | {ms(total)} |")
    print("\n(The «first token» row is what the user waits before seeing anything.)")


if __name__ == "__main__":
    main()

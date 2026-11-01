"""The whole RAG loop on the fixture corpus (lesson 01).

    python examples/rag_in_50_lines.py
    python examples/rag_in_50_lines.py "هزینهٔ پیک در تهران چقدر است؟"

Offline by default (HashingEmbedding + extractive FakeLLM). To use a real model,
set PARSRAG_BASE_URL / PARSRAG_MODEL (and OPENAI_API_KEY if your server needs one)
and install the extra: pip install "parsrag[openai]".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from parsrag import LLM, FakeLLM, HashingEmbedding
from parsrag.llm import OpenAICompatibleLLM
from parsrag.naive_rag import NaiveRAG, load_documents

CORPUS = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "corpus"
QUESTIONS = [
    "هزینهٔ پیک در تهران چقدر است؟",
    "مرکز خدمات پس از فروش کهربا در کدام شهر است؟",
    "جریمهٔ دیرکرد کتاب در کتابخانهٔ مهرگان چقدر است؟",
]


def make_llm() -> LLM:
    model = os.environ.get("PARSRAG_MODEL")
    if model:
        base_url = os.environ.get("PARSRAG_BASE_URL", "http://localhost:11434/v1")
        return OpenAICompatibleLLM(model=model, base_url=base_url)
    return FakeLLM(mode="extractive")


def main(questions: list[str]) -> None:
    rag = NaiveRAG(HashingEmbedding(), make_llm(), chunk_size=400, chunk_overlap=50, k=3)
    n = rag.index(load_documents(CORPUS))
    print(f"Indexed {n} chunks from {CORPUS}\n")
    for question in questions:
        answer = rag.answer(question)
        print(f"❓ {question}")
        print(f"💬 {answer.text}")
        for r in answer.sources:
            print(f"   📄 {r.chunk.id:<22} score={r.score:.3f}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or QUESTIONS)

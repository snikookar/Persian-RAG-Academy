"""The numbers quoted in lessons/05-embeddings/README.fa.md."""

from __future__ import annotations

import pytest

from parsrag import HashingEmbedding, TfidfEmbedding
from parsrag.eval import QAItem
from parsrag.eval.benchmark import benchmark_embedding, index_documents, load_fixture_benchmark
from parsrag.normalize import normalize

# same as examples/benchmark_embeddings.py
ARABIC_KEYBOARD = str.maketrans(
    {"ی": "ي", "ک": "ك", **dict(zip("۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩", strict=True))}
)


@pytest.fixture(scope="module")
def fixtures():  # type: ignore[no-untyped-def]
    return load_fixture_benchmark()


def _rounded(report) -> tuple[float, float, float]:  # type: ignore[no-untyped-def]
    return round(report.recall[1], 3), round(report.recall[3], 3), round(report.mrr, 3)


def test_offline_embedding_table(fixtures) -> None:  # type: ignore[no-untyped-def]
    docs, qa = fixtures
    assert len(docs) == 25 and sum(q.answerable for q in qa) == 59
    hashing = benchmark_embedding(HashingEmbedding(), docs, qa)
    assert _rounded(hashing["passage"]) == (0.763, 0.983, 0.858)
    word = benchmark_embedding(TfidfEmbedding("word"), docs, qa)
    assert _rounded(word["passage"]) == (0.729, 0.932, 0.836)
    char = benchmark_embedding(TfidfEmbedding("char"), docs, qa)
    assert _rounded(char["passage"]) == (0.797, 0.983, 0.886)


def test_normalisation_must_match(fixtures) -> None:  # type: ignore[no-untyped-def]
    docs, qa = fixtures
    messy = [
        QAItem(**{**q.model_dump(), "question": q.question.translate(ARABIC_KEYBOARD)}) for q in qa
    ]
    both = benchmark_embedding(TfidfEmbedding("word"), docs, messy)["passage"]
    index_only = benchmark_embedding(TfidfEmbedding("word"), docs, messy, query_preprocess=None)[
        "passage"
    ]
    assert round(both.mrr, 3) == 0.836
    assert round(index_only.mrr, 3) == 0.682


def test_index_documents_fits_tfidf(fixtures) -> None:  # type: ignore[no-untyped-def]
    docs, _ = fixtures
    emb = TfidfEmbedding()
    store = index_documents(docs[:3], emb, preprocess=normalize)
    assert emb.fitted and len(store) > 3
    assert store.chunks[0].text.startswith(docs[0].title)  # heading context on by default

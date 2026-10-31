from __future__ import annotations

import sys

import numpy as np
import pytest

from parsrag.embeddings import Embedding, HashingEmbedding, SentenceTransformerEmbedding


def test_satisfies_protocol() -> None:
    assert isinstance(HashingEmbedding(), Embedding)


def test_shape_dtype_and_unit_norm() -> None:
    emb = HashingEmbedding(dim=256)
    vecs = emb.embed(["کتاب", "فروشگاه اینترنتی", "hello world"])
    assert vecs.shape == (3, 256)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-5)


def test_deterministic_across_instances() -> None:
    a = HashingEmbedding().embed_query("بازیابی اطلاعات")
    b = HashingEmbedding().embed_query("بازیابی اطلاعات")
    np.testing.assert_array_equal(a, b)


def test_empty_inputs() -> None:
    emb = HashingEmbedding(dim=64)
    assert emb.embed([]).shape == (0, 64)
    np.testing.assert_array_equal(emb.embed_query(""), np.zeros(64, dtype=np.float32))


def test_rejects_single_string() -> None:
    with pytest.raises(TypeError):
        HashingEmbedding().embed("not a list")  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"dim": 0}, {"ngram_range": (3, 2)}, {"ngram_range": (0, 2)}])
def test_invalid_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HashingEmbedding(**kwargs)  # type: ignore[arg-type]


def test_similar_persian_words_are_closer_than_unrelated() -> None:
    emb = HashingEmbedding()
    base, plural, other = emb.embed(["کتاب", "کتاب‌ها", "باران"])
    assert base @ plural > base @ other + 0.2


def test_retrieves_related_sentence() -> None:
    emb = HashingEmbedding()
    docs = emb.embed(["هزینهٔ ارسال با پیک", "ساعت کاری کتابخانه", "گارانتی لوازم برقی"])
    q = emb.embed_query("کتابخانه چه ساعتی باز است؟")
    assert int(np.argmax(docs @ q)) == 1


def test_arabic_letters_hurt_similarity_without_normalization() -> None:
    """The motivation for lesson 02: different code points, different features."""
    emb = HashingEmbedding()
    persian, arabic = emb.embed(["کتابخانه کودک", "كتابخانه كودك"])
    assert persian @ arabic < 0.6


def test_features_include_words_and_char_ngrams() -> None:
    feats = HashingEmbedding(ngram_range=(2, 2)).features("آب")
    assert feats == {"w:آب": 1, "c:<آ": 1, "c:آب": 1, "c:ب>": 1}


def test_sentence_transformer_requires_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"parsrag\[st\]"):
        SentenceTransformerEmbedding()


def test_sentence_transformer_with_stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the wrapper with a fake module — no model download."""
    import types

    class StubModel:
        def __init__(self, name: str, device: str | None = None) -> None:
            self.seen: list[str] = []

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            self.seen.extend(texts)
            return np.ones((len(texts), 3), dtype=np.float32) / np.sqrt(3)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    emb = SentenceTransformerEmbedding("stub", query_prefix="query: ", document_prefix="passage: ")
    assert emb.dim == 3
    assert emb.embed(["الف", "ب"]).shape == (2, 3)
    assert emb.embed_query("پ").shape == (3,)
    assert emb._model.seen == ["passage: الف", "passage: ب", "query: پ"]

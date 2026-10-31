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


# ------------------------------------------------------------- lesson 05


def test_cosine_similarity() -> None:
    from parsrag.embeddings import cosine_similarity

    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 3]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1], [-1, -1]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_tfidf_fit_and_weights() -> None:
    from parsrag.embeddings import TfidfEmbedding

    emb = TfidfEmbedding()
    with pytest.raises(RuntimeError, match="fit"):
        _ = emb.dim
    emb.fit(["کتاب خوب است", "کتاب بد است", "باران می‌بارد"])
    assert isinstance(emb, Embedding)
    assert emb.dim == len(emb.vocabulary) == 7
    v = emb.embed_query("کتاب خوب")
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, rtol=1e-6)
    # the rare word «خوب» weighs more than the common «کتاب»
    assert v[emb.vocabulary["خوب"]] > v[emb.vocabulary["کتاب"]] > 0
    assert not emb.embed_query("واژهٔ ناشناخته").any()  # unseen terms are ignored
    with pytest.raises(TypeError):
        emb.fit("یک رشته")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TfidfEmbedding().fit(["", "  "])


def test_tfidf_char_ngrams_are_robust_to_suffixes() -> None:
    from parsrag.embeddings import TfidfEmbedding

    texts = ["کتاب‌ها روی میز است", "باران آمد", "کتابخانه بسته است"]
    word = TfidfEmbedding("word").fit(texts)
    char = TfidfEmbedding("char").fit(texts)
    # «کتاب» vs «کتابخانه»: no shared word, but many shared character n-grams
    assert word.embed_query("کتاب") @ word.embed(texts)[2] == 0.0
    assert char.embed_query("کتاب") @ char.embed(texts)[2] > 0.2
    assert TfidfEmbedding("char", ngram_range=(2, 2)).terms("آب") == ["<آ", "آب", "ب>"]
    for bad in ({"analyzer": "x"}, {"ngram_range": (3, 2)}, {"min_df": 0}):
        with pytest.raises(ValueError):
            TfidfEmbedding(**bad)  # type: ignore[arg-type]


class _FakeResponse:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return self._data


class _FakeClient:
    """Answers /embeddings with deterministic vectors, out of order on purpose."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, json: dict[str, object], **_: object) -> _FakeResponse:
        self.calls.append({"url": url, **json})
        texts = json["input"]
        assert isinstance(texts, list)
        rows = [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(texts)]
        return _FakeResponse({"data": list(reversed(rows))})


def test_openai_compatible_embedding_with_fake_client() -> None:
    from parsrag.embeddings import OpenAICompatibleEmbedding

    client = _FakeClient()
    emb = OpenAICompatibleEmbedding(
        "m", base_url="http://x/v1/", batch_size=2, query_prefix="q: ", document_prefix="d: ",
        client=client,
    )  # fmt: skip
    vecs = emb.embed(["الف", "بب", "پپپپپپپ"])
    assert vecs.shape == (3, 3) and emb.dim == 3
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-6)
    assert vecs[0, 0] < vecs[2, 0]  # order restored from "index"
    assert [c["input"] for c in client.calls] == [["d: الف", "d: بب"], ["d: پپپپپپپ"]]
    assert client.calls[0]["url"] == "http://x/v1/embeddings"
    emb.embed_query("س")
    assert client.calls[-1]["input"] == ["q: س"]
    assert isinstance(emb, Embedding)


def test_openai_embedding_requires_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from parsrag.embeddings import OpenAICompatibleEmbedding

    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError, match=r"parsrag\[openai\]"):
        OpenAICompatibleEmbedding("m", dim=3).embed(["x"])


def test_openai_embedding_bad_response() -> None:
    from parsrag.embeddings import OpenAICompatibleEmbedding

    class Bad(_FakeClient):
        def post(self, url: str, json: dict[str, object], **_: object) -> _FakeResponse:
            return _FakeResponse({"error": "nope"})

    with pytest.raises(RuntimeError, match="Unexpected"):
        OpenAICompatibleEmbedding("m", client=Bad()).embed(["x"])

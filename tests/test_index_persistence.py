"""Index persistence, incremental ingestion, metadata filters and presets (0.2.0)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from parsrag import Document, HashingEmbedding, PersianRAG, TfidfEmbedding
from parsrag.chunking import RecursiveChunker
from parsrag.config import (
    PRESETS,
    ProviderConfig,
    RAGConfig,
    config_template,
    load_config,
    make_embedding,
    make_llm,
    preset_config,
)
from parsrag.eval.dataset import QAItem
from parsrag.rag import INDEX_FORMAT_VERSION, IndexUpdate, document_hash
from parsrag.retrieval import BM25Retriever, PersianTokenizer
from parsrag.stores.memory import InMemoryVectorStore

# Original sentences written for these tests (a fictional bakery and a fictional clinic).
BAKERY = Document(
    id="bakery",
    title="نانوایی سپیده",
    text="# ساعت کار\nنانوایی سپیده هر روز از ساعت ۶ صبح باز است.\n\n"
    "# سفارش\nسفارش کیک تولد باید دو روز زودتر ثبت شود.",
    metadata={"lang": "fa", "tags": ["food", "shop"], "format": "md"},
)
CLINIC = Document(
    id="clinic",
    title="درمانگاه نارون",
    text="# نوبت\nنوبت دندانپزشکی درمانگاه نارون فقط تلفنی گرفته می‌شود.\n\n"
    "# هزینه\nویزیت عمومی ۳۰۰ هزار تومان است.",
    metadata={"lang": "fa", "tags": ["health"], "format": "pdf"},
)
NOTE = Document(
    id="note",
    text="Opening hours of the reading room: 9 to 17.",
    metadata={"lang": "en", "format": "txt"},
)


class CountingEmbedding(HashingEmbedding):
    """HashingEmbedding that counts how many texts it embedded."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed(self, texts):  # type: ignore[no-untyped-def]
        self.calls += len(texts)
        return super().embed(texts)


def build(*docs: Document, **kwargs: object) -> PersianRAG:
    return PersianRAG(list(docs) or [BAKERY, CLINIC, NOTE], **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------ save / load


def test_save_writes_versioned_folder(tmp_path: Path) -> None:
    rag = build()
    rag.save(tmp_path / "idx")
    files = {p.name for p in (tmp_path / "idx").iterdir()}
    assert {"rag.json", "documents.jsonl", "chunks.jsonl", "bm25.json", "store"} <= files
    info = PersianRAG.index_info(tmp_path / "idx")
    assert info["format_version"] == INDEX_FORMAT_VERSION == 2
    assert info["store"] == "memory" and info["n_chunks"] == len(rag)
    assert set(info["hashes"]) == {"bakery", "clinic", "note"}
    assert info["config"]["llm"]["provider"] == "fake"
    # no temporary folders left behind
    assert [p.name for p in tmp_path.iterdir()] == ["idx"]


def test_load_reuses_chunks_vectors_and_bm25(tmp_path: Path) -> None:
    rag = build(min_score=None, k=3)
    rag.save(tmp_path / "idx")
    emb = CountingEmbedding()
    loaded = PersianRAG.load(tmp_path / "idx", embedding=emb)
    assert emb.calls == 0  # nothing re-embedded
    assert loaded.k == 3 and len(loaded) == len(rag)
    assert [c.id for c in loaded.chunks] == [c.id for c in rag.chunks]
    assert loaded.bm25.state() == rag.bm25.state()
    q = "نانوایی سپیده از چه ساعتی باز است؟"
    assert loaded.ask(q) == rag.ask(q)
    assert loaded.index_version == rag.index_version


def test_load_without_arguments_uses_saved_config(tmp_path: Path) -> None:
    cfg = RAGConfig.model_validate({"retrieval": {"k": 2, "rerank": "none"}})
    rag = PersianRAG.from_config(cfg, load_documents=False)
    rag.add_documents([BAKERY, CLINIC])
    rag.calibrate(
        [
            QAItem(id="q1", question="ساعت کار نانوایی سپیده؟", answer="۶ صبح", doc_id="bakery"),
            QAItem(id="q2", question="قیمت بلیت هواپیما به ماه؟", answer=""),
        ]
    )
    rag.save(tmp_path / "idx")
    loaded = PersianRAG.load(tmp_path / "idx")
    assert loaded.k == 2 and loaded.reranker is None
    assert loaded.min_score == rag.min_score
    assert loaded.config.retrieval.rerank == "none"


def test_load_rejects_newer_format_and_wrong_embedding(tmp_path: Path) -> None:
    build(embedding=CountingEmbedding()).save(tmp_path / "idx")
    with pytest.raises(ValueError, match="CountingEmbedding"):
        PersianRAG.load(tmp_path / "idx")  # config would make HashingEmbedding
    assert len(PersianRAG.load(tmp_path / "idx", embedding=CountingEmbedding())) > 0
    info = json.loads((tmp_path / "idx" / "rag.json").read_text("utf-8"))
    info["format_version"] = INDEX_FORMAT_VERSION + 1
    (tmp_path / "idx" / "rag.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        PersianRAG.load(tmp_path / "idx", embedding=CountingEmbedding())


def test_wrong_dimension_vectors_are_recomputed(tmp_path: Path) -> None:
    build().save(tmp_path / "idx")
    small = HashingEmbedding(dim=64)
    loaded = PersianRAG.load(tmp_path / "idx", embedding=small)
    assert loaded.store.search("نانوایی", 1)[0].chunk.doc_id == "bakery"  # type: ignore[union-attr]


def test_loads_a_format_1_index_from_0_1_0(tmp_path: Path) -> None:
    """The folder layout written by parsrag 0.1.0: no version, no chunks, no BM25 state."""
    rag = build()
    old = tmp_path / "old"
    old.mkdir()
    with (old / "documents.jsonl").open("w", encoding="utf-8") as f:
        for doc in rag.documents.values():
            f.write(doc.model_dump_json() + "\n")
    options = {
        "k": 4,
        "fetch_k": 20,
        "min_score": None,
        "heading_context": True,
        "normalize_text": True,
        "embedding": "HashingEmbedding",
        "n_chunks": len(rag),
    }
    (old / "rag.json").write_text(json.dumps(options), encoding="utf-8")
    assert isinstance(rag.store, InMemoryVectorStore)
    rag.store.save(old / "store")
    assert PersianRAG.index_info(old)["format_version"] == 1
    loaded = PersianRAG.load(old)
    assert [c.id for c in loaded.chunks] == [c.id for c in rag.chunks]
    q = "ویزیت عمومی درمانگاه چقدر است؟"
    assert loaded.ask(q) == rag.ask(q)
    loaded.save(tmp_path / "new")  # re-saving upgrades the format
    assert PersianRAG.index_info(tmp_path / "new")["format_version"] == 2


def test_tfidf_index_round_trip(tmp_path: Path) -> None:
    rag = build(embedding=TfidfEmbedding())
    rag.save(tmp_path / "idx")
    loaded = PersianRAG.load(tmp_path / "idx", embedding=TfidfEmbedding())
    q = "سفارش کیک تولد"
    assert [r.chunk.id for r in loaded.search(q)] == [r.chunk.id for r in rag.search(q)]


def test_faiss_store_round_trip_and_delete(tmp_path: Path) -> None:
    pytest.importorskip("faiss", reason="needs the faiss extra")
    from parsrag.stores.faiss_store import FaissVectorStore

    emb = HashingEmbedding()
    rag = build(embedding=emb, store=FaissVectorStore(emb))
    rag.save(tmp_path / "idx")
    assert PersianRAG.index_info(tmp_path / "idx")["store"] == "faiss"
    counting = CountingEmbedding()
    loaded = PersianRAG.load(tmp_path / "idx", embedding=counting, store=FaissVectorStore(counting))
    assert counting.calls == 0 and isinstance(loaded.store, FaissVectorStore)
    assert loaded.remove_document("clinic") > 0
    assert {c.doc_id for c in loaded.store.chunks} == {"bakery", "note"}  # type: ignore[attr-defined]
    hits = loaded.store.search("درمانگاه نارون", 10)
    assert hits and all(h.chunk.doc_id != "clinic" for h in hits)


# ------------------------------------------------------------ incremental index


def test_upsert_skips_unchanged_and_reindexes_changed() -> None:
    emb = CountingEmbedding()
    rag = PersianRAG([BAKERY, CLINIC], embedding=emb)
    first = emb.calls
    version = rag.index_version
    report = rag.upsert_documents([BAKERY, CLINIC])
    assert report.unchanged == ["bakery", "clinic"] and not report.changed
    assert emb.calls == first and rag.index_version == version  # nothing re-embedded

    edited = CLINIC.model_copy(update={"text": "# هزینه\nویزیت عمومی ۳۵۰ هزار تومان است."})
    report = rag.upsert_documents([BAKERY, edited, NOTE])
    assert report.added == ["note"] and report.updated == ["clinic"]
    assert report.unchanged == ["bakery"] and report.chunks_removed == 2
    assert emb.calls - first == report.chunks_added  # only new chunks embedded
    assert rag.index_version == version + 1
    assert "350" in rag.ask("ویزیت عمومی درمانگاه چقدر است؟").text  # digits normalised
    assert not any("300 هزار" in c.text for c in rag.chunks)
    assert len(rag.store) == len(rag) == len(rag.bm25)


def test_upsert_prune_and_remove() -> None:
    rag = build()
    report = rag.upsert_documents([BAKERY], prune=True)
    assert sorted(report.removed) == ["clinic", "note"] and set(rag.documents) == {"bakery"}
    assert {c.doc_id for c in rag.chunks} == {"bakery"}
    rag.add_documents([CLINIC])
    removed = rag.remove_documents(["clinic"])
    assert removed == 2 and "clinic" not in rag.documents
    assert all(r.chunk.doc_id == "bakery" for r in rag.search("درمانگاه نارون", 10))
    with pytest.raises(KeyError):
        rag.remove_document("clinic")
    assert rag.remove_documents(["clinic"], missing_ok=True) == 0
    with pytest.raises(ValueError, match="duplicate"):
        rag.upsert_documents([BAKERY, BAKERY])


def test_bm25_after_removal_equals_fresh_index() -> None:
    rag = build()
    rag.remove_documents(["note"])
    fresh = BM25Retriever(
        [c for c in build().chunks if c.doc_id != "note"], tokenizer=PersianTokenizer(stem=True)
    )
    q = "ساعت کار نانوایی"
    assert [(r.chunk.id, round(r.score, 6)) for r in rag.bm25.search(q, 5)] == [
        (r.chunk.id, round(r.score, 6)) for r in fresh.search(q, 5)
    ]


def test_remove_with_tfidf_refits_and_can_empty_the_index() -> None:
    rag = build(embedding=TfidfEmbedding())
    rag.remove_documents(["bakery", "note"])
    assert {r.chunk.doc_id for r in rag.search("نوبت دندانپزشکی")} == {"clinic"}
    rag.remove_document("clinic")
    assert len(rag) == 0


def test_store_without_delete_is_reported() -> None:
    class AppendOnly:
        """A store with the protocol's methods but no ``delete``."""

        def __init__(self) -> None:
            self.inner = InMemoryVectorStore(HashingEmbedding())

        def __len__(self) -> int:
            return len(self.inner)

        def add(self, chunks, vectors=None):  # type: ignore[no-untyped-def]
            self.inner.add(chunks, vectors)

        def search(self, query, k=4, *, filter=None):  # type: ignore[no-untyped-def]
            return self.inner.search(query, k, filter=filter)

        def search_by_vector(self, vector, k=4, *, filter=None):  # type: ignore[no-untyped-def]
            return self.inner.search_by_vector(vector, k, filter=filter)

    rag = PersianRAG([BAKERY], store=AppendOnly())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="cannot delete"):
        rag.remove_document("bakery")
    assert "bakery" in rag.documents  # nothing was half-removed
    assert rag.upsert_documents([BAKERY]).unchanged == ["bakery"]


def test_memory_store_delete() -> None:
    rag = build()
    store = rag.store
    assert isinstance(store, InMemoryVectorStore)
    n = len(store)
    assert store.delete(["bakery#0", "missing"]) == 1 and len(store) == n - 1
    assert store.vectors().shape == (n - 1, store.embedding.dim)
    assert np.allclose(np.linalg.norm(store.vectors(), axis=1), 1.0, atol=1e-5)


def test_document_hash_ignores_source_path() -> None:
    a = BAKERY.model_copy(update={"metadata": {**BAKERY.metadata, "source": "a/x.md"}})
    b = BAKERY.model_copy(update={"metadata": {**BAKERY.metadata, "source": "b/x.md"}})
    assert document_hash(a) == document_hash(b) != document_hash(CLINIC)


def test_index_update_summary() -> None:
    text = IndexUpdate(added=["a"], unchanged=["b", "c"], chunks_added=3).summary()
    assert text == "added 1, updated 0, unchanged 2, removed 0 documents (+3 / -0 chunks)"


# ----------------------------------------------------------------------- filters


def test_filters_on_document_and_chunk_metadata() -> None:
    rag = build()
    q = "ساعت باز بودن چیست؟"
    assert {r.chunk.doc_id for r in rag.search(q, 10, filters={"lang": "en"})} == {"note"}
    assert {r.chunk.doc_id for r in rag.search(q, 10, filters={"tags": "health"})} == {"clinic"}
    by_tags = rag.search(q, 10, filters={"tags": ["food", "health"]})
    assert {r.chunk.doc_id for r in by_tags} <= {"bakery", "clinic"}
    assert {r.chunk.doc_id for r in rag.search(q, 10, filters={"title": "نانوایی سپیده"})} == {
        "bakery"
    }
    firsts = rag.search(q, 10, filters={"index": 0})
    assert firsts and all(r.chunk.metadata["index"] == 0 for r in firsts)
    both = rag.search(q, 10, filters={"doc_id": ["bakery", "note"], "lang": "fa"})
    assert {r.chunk.doc_id for r in both} == {"bakery"}
    # the old ``filter=`` keyword still works and resolves document keys too
    assert {r.chunk.doc_id for r in rag.search(q, 10, filter={"format": "pdf"})} == {"clinic"}


def test_resolve_filter() -> None:
    rag = build()
    assert rag.resolve_filter(None) is None
    assert rag.resolve_filter({"page": 1}) == {"page": 1}
    assert rag.resolve_filter({"lang": "en"}) == {"doc_id": ["note"]}
    assert rag.resolve_filter({"lang": "de"}) == {"doc_id": []}

    def pred(c: object) -> bool:
        return True

    assert rag.resolve_filter(pred) is pred


def test_ask_and_stream_with_filters() -> None:
    rag = build()
    q = "ویزیت عمومی چقدر است؟"
    answer = rag.ask(q, filters={"doc_id": "clinic"})
    assert answer.citations and {c.doc_id for c in answer.citations} == {"clinic"}
    nothing = rag.ask(q, filters={"lang": "de"})
    assert nothing.abstained and nothing.abstain_reason == "no_results"
    events = list(rag.stream(q, filters={"tags": "health"}))
    assert events[0].type == "sources"
    assert {r.chunk.doc_id for r in events[0].data} == {"clinic"}


# ----------------------------------------------------------------------- presets


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_templates(preset: str) -> None:
    text = config_template(preset)
    assert "api_key =" not in text and "sk-" not in text
    if sys.version_info >= (3, 11):
        import tomllib

        data = tomllib.loads(text)
        assert load_config(data) == preset_config(preset)
    cfg = preset_config(preset)
    if preset == "offline":
        assert cfg.llm.provider == "fake" and cfg.embedding.provider == "hashing"
        make_llm(cfg.llm)
    else:
        assert cfg.llm.model == "<chat-model>"
        with pytest.raises(ValueError, match="placeholder"):
            make_llm(cfg.llm)
        with pytest.raises(ValueError, match="placeholder"):
            make_embedding(cfg.embedding)


def test_unknown_preset_and_keys_in_config() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        config_template("cloud")
    with pytest.raises(ValueError, match="unknown preset"):
        preset_config("cloud")
    with pytest.raises(ValueError, match="environment variable"):
        load_config({"llm": {"provider": "openai", "model": "m", "api_key": "secret"}})


def test_ollama_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    emb = make_embedding(ProviderConfig(provider="ollama", model="some-embedder"))
    assert emb.base_url.startswith("http://localhost:11434")  # type: ignore[attr-defined]
    assert emb.api_key == "ollama"  # type: ignore[attr-defined]
    monkeypatch.setenv("MY_KEY", "from-env")
    emb = make_embedding(
        ProviderConfig(provider="openai", model="m", base_url="http://x/v1", api_key_env="MY_KEY")
    )
    assert emb.api_key == "from-env"  # type: ignore[attr-defined]


def test_chunker_change_on_load_rebuilds(tmp_path: Path) -> None:
    rag = build()
    rag.save(tmp_path / "idx")
    same = PersianRAG.load(tmp_path / "idx", chunker=RecursiveChunker(400, 50))
    assert [c.id for c in same.chunks] == [c.id for c in rag.chunks]
    small = PersianRAG.load(tmp_path / "idx", chunker=RecursiveChunker(40, 0))
    assert len(small) > len(rag)
    shutil.rmtree(tmp_path / "idx")

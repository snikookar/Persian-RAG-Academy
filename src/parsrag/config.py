"""Configuration for :class:`parsrag.PersianRAG` (lesson 13).

A configuration is a plain mapping, usually read from a TOML file::

    # parsrag.toml
    [documents]
    paths = ["docs/"]            # relative to this file

    [index]
    path = ".parsrag/index"      # where `parsrag ingest` saves the index

    [chunking]
    size = 400
    overlap = 50
    heading_context = true

    [retrieval]
    k = 4
    fetch_k = 20
    rerank = "lexical"           # "lexical" | "cross-encoder" | "none"
    stem = true
    # min_score = 4.2            # BM25 abstention threshold (see `parsrag eval --calibrate`)

    [embedding]
    provider = "hashing"         # "hashing" | "tfidf" | "st" | "openai"
    # model = "..."

    [llm]
    provider = "fake"            # "fake" | "openai" | "ollama"
    # model = "qwen2.5:7b"
    # base_url = "http://localhost:11434/v1"
    # api_key_env = "OPENAI_API_KEY"
    # timeout = 60

Real providers are created lazily, so reading a config never needs an extra
or a network connection. ``tomllib`` is in the standard library from Python
3.11; on 3.10 install ``tomli`` or use a ``.json`` file with the same keys.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from parsrag.embeddings import Embedding, HashingEmbedding, TfidfEmbedding
from parsrag.llm import LLM, FakeLLM, OpenAICompatibleLLM


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentsConfig(_Section):
    paths: list[str] = Field(default_factory=list)
    pattern: str = "*"
    recursive: bool = True


class IndexConfig(_Section):
    path: str = ".parsrag/index"


class ChunkingConfig(_Section):
    size: int = 400
    overlap: int = 50
    heading_context: bool = True
    normalize: bool = True


class RetrievalConfig(_Section):
    k: int = 4
    fetch_k: int = 20
    rerank: str = "lexical"
    cross_encoder: str | None = None
    stem: bool = True
    min_score: float | None = None


class ProviderConfig(_Section):
    provider: str
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 60.0
    temperature: float = 0.0


class RAGConfig(_Section):
    """The whole configuration; every section has course defaults."""

    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: ProviderConfig = Field(default_factory=lambda: ProviderConfig(provider="hashing"))
    llm: ProviderConfig = Field(default_factory=lambda: ProviderConfig(provider="fake"))
    #: Directory that relative paths are resolved against (the config file's folder).
    base_dir: str = "."

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else Path(self.base_dir) / p


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Reading TOML on Python 3.10 needs tomli (pip install tomli), "
                "or use a .json config with the same keys."
            ) from exc
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(source: str | Path | Mapping[str, Any] | RAGConfig | None = None) -> RAGConfig:
    """Read a config from a ``.toml``/``.json`` file, a mapping, or return the defaults."""
    if source is None:
        return RAGConfig()
    if isinstance(source, RAGConfig):
        return source
    if isinstance(source, Mapping):
        return RAGConfig.model_validate(dict(source))
    path = Path(source)
    data = json.loads(path.read_text("utf-8")) if path.suffix == ".json" else _read_toml(path)
    data.setdefault("base_dir", str(path.resolve().parent))
    return RAGConfig.model_validate(data)


def make_embedding(cfg: ProviderConfig) -> Embedding:
    """Build the embedding named in the config (real models are loaded lazily)."""
    provider = cfg.provider.lower()
    if provider == "hashing":
        return HashingEmbedding()
    if provider == "tfidf":
        return TfidfEmbedding()
    if provider in ("st", "sentence-transformers"):
        from parsrag.embeddings import SentenceTransformerEmbedding

        if not cfg.model:
            raise ValueError("embedding.model is required for sentence-transformers")
        return SentenceTransformerEmbedding(cfg.model)
    if provider == "openai":
        from parsrag.embeddings import OpenAICompatibleEmbedding

        if not cfg.model:
            raise ValueError("embedding.model is required for the openai provider")
        kwargs: dict[str, Any] = {"api_key": os.environ.get(cfg.api_key_env)}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        return OpenAICompatibleEmbedding(cfg.model, **kwargs)
    raise ValueError(f"unknown embedding provider {cfg.provider!r}")


OLLAMA_URL = "http://localhost:11434/v1"


def make_llm(cfg: ProviderConfig) -> LLM:
    """Build the LLM named in the config. ``fake`` is the offline extractive stand-in."""
    provider = cfg.provider.lower()
    if provider == "fake":
        from parsrag.retrieval import PERSIAN_STOPWORDS

        return FakeLLM(mode="extractive", cite=True, ignore=PERSIAN_STOPWORDS)
    if provider in ("openai", "ollama"):
        if not cfg.model:
            raise ValueError(f"llm.model is required for the {provider} provider")
        default_url = OLLAMA_URL if provider == "ollama" else "https://api.openai.com/v1"
        return OpenAICompatibleLLM(
            cfg.model,
            base_url=cfg.base_url or default_url,
            api_key=os.environ.get(cfg.api_key_env) or ("ollama" if provider == "ollama" else None),
            timeout=cfg.timeout,
            temperature=cfg.temperature,
        )
    raise ValueError(f"unknown llm provider {cfg.provider!r}")


__all__ = [
    "ChunkingConfig",
    "DocumentsConfig",
    "IndexConfig",
    "ProviderConfig",
    "RAGConfig",
    "RetrievalConfig",
    "load_config",
    "make_embedding",
    "make_llm",
]

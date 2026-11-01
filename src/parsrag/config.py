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
    provider = "hashing"         # "hashing" | "tfidf" | "st" | "openai" | "ollama"
    # model = "..."

    [llm]
    provider = "fake"            # "fake" | "openai" | "ollama"
    # model = "qwen2.5:7b"
    # base_url = "http://localhost:11434/v1"
    # api_key_env = "OPENAI_API_KEY"
    # timeout = 60

    [security]                   # lesson 14: prompt injection hidden in documents
    enabled = false
    # action = "drop"            # "drop" | "flag" suspicious chunks

Real providers are created lazily, so reading a config never needs an extra
or a network connection. API keys are never part of a config: a provider names
the environment variable that holds its key (``api_key_env``), and an
``api_key`` entry is rejected. :func:`config_template` writes ready-to-edit
files for the presets in :data:`PRESETS` (``parsrag init --preset ...``).
``tomllib`` is in the standard library from Python 3.11; on 3.10 install ``tomli``
or use a ``.json`` file with the same keys.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    #: Embedding prefixes, e.g. "query: " / "passage: " for the E5 family.
    query_prefix: str = ""
    document_prefix: str = ""

    @model_validator(mode="before")
    @classmethod
    def _no_secrets(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and any(k in data for k in ("api_key", "key", "token")):
            raise ValueError(
                "API keys are never read from a config file: put the key in an environment "
                "variable and name it with api_key_env (default OPENAI_API_KEY)"
            )
        return data


class SecurityConfig(_Section):
    """Lesson 14's prompt-injection guard (off by default)."""

    enabled: bool = False
    action: str = "drop"  # "drop" | "flag"
    sanitize: bool = True
    delimiters: str = "random"  # "random" | "fixed" | "none"
    reminder: bool = True
    check_output: bool = True

    def make_guard(self) -> Any:
        """An :class:`~parsrag.security.InjectionGuard`, or ``None`` when disabled."""
        if not self.enabled:
            return None
        from parsrag.security import InjectionGuard

        return InjectionGuard(
            action=self.action,  # type: ignore[arg-type]
            sanitize=self.sanitize,
            delimiters=self.delimiters,  # type: ignore[arg-type]
            reminder=self.reminder,
            check_output=self.check_output,
        )


class RAGConfig(_Section):
    """The whole configuration; every section has course defaults."""

    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: ProviderConfig = Field(default_factory=lambda: ProviderConfig(provider="hashing"))
    llm: ProviderConfig = Field(default_factory=lambda: ProviderConfig(provider="fake"))
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    #: Directory that relative paths are resolved against (the config file's folder).
    base_dir: str = "."

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else Path(self.base_dir) / p


def _read_toml(path: Path) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Reading TOML on Python 3.10 needs tomli (pip install tomli), "
                "or use a .json config with the same keys."
            ) from exc
    with path.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    return data


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


OLLAMA_URL = "http://localhost:11434/v1"


def _require_model(cfg: ProviderConfig, what: str) -> str:
    model = (cfg.model or "").strip()
    if not model:
        raise ValueError(f"{what}.model is required for the {cfg.provider} provider")
    if model.startswith("<") and model.endswith(">"):
        raise ValueError(
            f"{what}.model is still the placeholder {model!r}: edit parsrag.toml and write the "
            "name of a model you can use"
        )
    return model


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
        return SentenceTransformerEmbedding(
            cfg.model, query_prefix=cfg.query_prefix, document_prefix=cfg.document_prefix
        )
    if provider in ("openai", "ollama"):
        from parsrag.embeddings import OpenAICompatibleEmbedding

        model = _require_model(cfg, "embedding")
        api_key = os.environ.get(cfg.api_key_env)
        kwargs: dict[str, Any] = {
            "api_key": api_key or ("ollama" if provider == "ollama" else None),
            "query_prefix": cfg.query_prefix,
            "document_prefix": cfg.document_prefix,
        }
        if cfg.base_url or provider == "ollama":
            kwargs["base_url"] = cfg.base_url or OLLAMA_URL
        return OpenAICompatibleEmbedding(model, **kwargs)
    raise ValueError(f"unknown embedding provider {cfg.provider!r}")


def make_llm(cfg: ProviderConfig) -> LLM:
    """Build the LLM named in the config. ``fake`` is the offline extractive stand-in."""
    provider = cfg.provider.lower()
    if provider == "fake":
        from parsrag.retrieval import PERSIAN_STOPWORDS

        return FakeLLM(mode="extractive", cite=True, ignore=PERSIAN_STOPWORDS)
    if provider in ("openai", "ollama"):
        model = _require_model(cfg, "llm")
        default_url = OLLAMA_URL if provider == "ollama" else "https://api.openai.com/v1"
        return OpenAICompatibleLLM(
            model,
            base_url=cfg.base_url or default_url,
            api_key=os.environ.get(cfg.api_key_env) or ("ollama" if provider == "ollama" else None),
            timeout=cfg.timeout,
            temperature=cfg.temperature,
        )
    raise ValueError(f"unknown llm provider {cfg.provider!r}")


# --------------------------------------------------------------------- presets

_COMMON = """\
[documents]
paths = ["docs"]               # folders or files, relative to this file
pattern = "*"
recursive = true

[index]
path = ".parsrag/index"        # written by `parsrag ingest`, read by `ask` / `serve`

[chunking]
size = 400                     # lesson 04
overlap = 50
heading_context = true

[retrieval]
k = 4                          # passages shown to the model (lesson 09)
fetch_k = 20                   # candidates for the reranker (lesson 08)
rerank = "lexical"             # "lexical" | "cross-encoder" | "none"
stem = true                    # light stemming in BM25 (lesson 07)
# min_score = 4.0              # set by `parsrag eval --calibrate` (lesson 09)
"""

_SECURITY = """
[security]
enabled = false                # prompt-injection guard (lesson 14)
# action = "drop"              # "drop" | "flag" suspicious chunks
"""

_PROVIDERS = {
    "offline": """
# Offline preset: no model downloads, no network, no keys. Good for trying the
# pipeline; switch to the ollama or openai-compatible preset for real answers.

[embedding]
provider = "hashing"           # "hashing" (offline) | "st" | "openai" | "ollama" | "tfidf"
# model = "intfloat/multilingual-e5-base"

[llm]
provider = "fake"              # "fake" (offline, extractive) | "ollama" | "openai"
# model = "qwen2.5:7b"
# base_url = "http://localhost:11434/v1"
# api_key_env = "OPENAI_API_KEY"
timeout = 60
""",
    "ollama": """
# Ollama preset: everything runs on your machine through Ollama's
# OpenAI-compatible endpoint. Edit the two model names below (placeholders),
# then pull them first, for example:
#   ollama pull <embedding-model>
#   ollama pull <chat-model>
# No API key is needed.

[embedding]
provider = "ollama"
model = "<embedding-model>"    # EDIT: a multilingual embedding model you pulled
base_url = "http://localhost:11434/v1"
# query_prefix = "query: "     # only for models trained with prefixes (E5 family)
# document_prefix = "passage: "

[llm]
provider = "ollama"
model = "<chat-model>"         # EDIT: a chat model that handles Persian well
base_url = "http://localhost:11434/v1"
timeout = 120                  # local models can be slow on the first request
temperature = 0.0
""",
    "openai-compatible": """
# OpenAI-compatible preset: any server that speaks the OpenAI HTTP API
# (a hosted provider, vLLM, LM Studio, OpenRouter, ...). Edit the model names
# and base_url below (placeholders). The API key is read ONLY from the
# environment variable named in api_key_env — never write it in this file:
#   export OPENAI_API_KEY="..."

[embedding]
provider = "openai"
model = "<embedding-model>"    # EDIT: the provider's embedding model name
base_url = "https://api.openai.com/v1"   # EDIT for other providers
api_key_env = "OPENAI_API_KEY"

[llm]
provider = "openai"
model = "<chat-model>"         # EDIT: the provider's chat model name
base_url = "https://api.openai.com/v1"   # EDIT for other providers
api_key_env = "OPENAI_API_KEY"
timeout = 60
temperature = 0.0
""",
}

#: Names accepted by :func:`config_template` and ``parsrag init --preset``.
PRESETS: tuple[str, ...] = tuple(_PROVIDERS)


def config_template(preset: str = "offline") -> str:
    """A commented, ready-to-edit ``parsrag.toml`` for one of :data:`PRESETS`.

    Model names in the real-model presets are placeholders (``<chat-model>``)
    that must be edited; building a provider from an unedited placeholder raises
    a clear error. No preset ever contains an API key.
    """
    if preset not in _PROVIDERS:
        raise ValueError(f"unknown preset {preset!r} (choose from: {', '.join(PRESETS)})")
    header = f"# parsrag configuration ({preset} preset) — see `parsrag.config` and lesson 13.\n\n"
    return header + _COMMON + _PROVIDERS[preset] + _SECURITY


def preset_config(preset: str = "offline") -> RAGConfig:
    """The :class:`RAGConfig` that :func:`config_template` describes (no file needed)."""
    if preset not in _PROVIDERS:
        raise ValueError(f"unknown preset {preset!r} (choose from: {', '.join(PRESETS)})")
    cfg = RAGConfig(documents=DocumentsConfig(paths=["docs"]))
    if preset == "ollama":
        cfg.embedding = ProviderConfig(
            provider="ollama", model="<embedding-model>", base_url=OLLAMA_URL
        )
        cfg.llm = ProviderConfig(
            provider="ollama", model="<chat-model>", base_url=OLLAMA_URL, timeout=120
        )
    elif preset == "openai-compatible":
        url = "https://api.openai.com/v1"
        cfg.embedding = ProviderConfig(provider="openai", model="<embedding-model>", base_url=url)
        cfg.llm = ProviderConfig(provider="openai", model="<chat-model>", base_url=url)
    return cfg


__all__ = [
    "PRESETS",
    "ChunkingConfig",
    "DocumentsConfig",
    "IndexConfig",
    "ProviderConfig",
    "RAGConfig",
    "RetrievalConfig",
    "SecurityConfig",
    "config_template",
    "load_config",
    "make_embedding",
    "make_llm",
    "preset_config",
]

"""API settings from environment variables (lesson 12).

=========================== ============================================ ==========
variable                    meaning                                      default
=========================== ============================================ ==========
``PARSRAG_CONFIG``          ``parsrag.toml`` / ``.json`` (lesson 13)     —
``PARSRAG_INDEX``           folder written by ``parsrag ingest``         —
``PARSRAG_DOCS``            folder to index at startup                   —
``PARSRAG_DATABASE_URL``    PostgreSQL + pgvector instead of in-memory   —
``PARSRAG_LLM_PROVIDER``    ``fake`` / ``openai`` / ``ollama``           config
``PARSRAG_LLM_MODEL``       model name                                   config
``PARSRAG_LLM_BASE_URL``    OpenAI-compatible endpoint                   config
``PARSRAG_LLM_TIMEOUT``     seconds before ``504 llm_timeout``           30
``PARSRAG_CACHE_SIZE``      cached answers (0 = no cache)                256
``PARSRAG_CACHE_TTL``       seconds an answer stays cached               3600
``PARSRAG_RATE_LIMIT``      requests per minute per client (0 = off)     60
``PARSRAG_MAX_UPLOAD_MB``   largest accepted upload                      20
``PARSRAG_LOG_LEVEL``       ``DEBUG`` / ``INFO`` / ``WARNING``           INFO
``PARSRAG_SECURITY``        prompt-injection guard (lesson 14) on/off    config
=========================== ============================================ ==========

No ``pydantic-settings`` dependency: :meth:`Settings.from_env` reads
``os.environ`` (or any mapping, which is how the tests pass values).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel


class Settings(BaseModel):
    config: str | None = None
    index: str | None = None
    docs: str | None = None
    database_url: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_timeout: float = 30.0
    cache_size: int = 256
    cache_ttl: float = 3600.0
    rate_limit: int = 60
    max_upload_mb: float = 20.0
    log_level: str = "INFO"
    security: bool | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        values = {
            name: env[f"PARSRAG_{name.upper()}"]
            for name in cls.model_fields
            if env.get(f"PARSRAG_{name.upper()}", "") != ""
        }
        return cls.model_validate(values)


__all__ = ["Settings"]

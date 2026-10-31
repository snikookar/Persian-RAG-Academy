"""Core data types shared across parsrag modules."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document, e.g. one file of the corpus."""

    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable piece of a document."""

    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A chunk returned by a retriever together with its relevance score."""

    chunk: Chunk
    score: float

"""Language-model interfaces.

* :class:`LLM` — the protocol every model implements: ``generate(messages) -> str``.
* :class:`FakeLLM` — deterministic and offline; used in all tests, graders and examples.
* :class:`OpenAICompatibleLLM` — talks to any OpenAI-compatible ``/chat/completions``
  endpoint (OpenAI, vLLM, OpenRouter, Ollama's ``/v1`` ...). Needs the ``openai`` extra.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """One chat message."""

    role: Role
    content: str


MessageLike = Message | Mapping[str, str]


def to_messages(messages: Sequence[MessageLike]) -> list[Message]:
    """Coerce dicts like ``{"role": "user", "content": "..."}`` into :class:`Message`."""
    return [m if isinstance(m, Message) else Message.model_validate(dict(m)) for m in messages]


@runtime_checkable
class LLM(Protocol):
    """Anything that turns a list of chat messages into a reply."""

    def generate(self, messages: Sequence[MessageLike]) -> str: ...


# --------------------------------------------------------------------------- FakeLLM

_QUESTION_MARKERS = ("پرسش:", "سوال:", "سؤال:", "Question:")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n+")
_WORD = re.compile(r"\w+")
_BLOCK = re.compile(r"^\[(\d+)\]", re.MULTILINE)


def _words(text: str, ignore: frozenset[str] = frozenset()) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in ignore}


def _numbered_blocks(context: str) -> list[tuple[int | None, str]]:
    """Split a context at ``[n] header`` lines; text before the first block gets ``None``.

    The rest of the ``[n]`` line is a source label, not content, so it is dropped.
    """
    marks = list(_BLOCK.finditer(context))
    if not marks:
        return [(None, context)]
    blocks: list[tuple[int | None, str]] = [(None, context[: marks[0].start()])]
    for m, nxt in zip(marks, [*marks[1:], None], strict=True):
        end = nxt.start() if nxt is not None else len(context)
        _header, _, body = context[m.end() : end].partition("\n")  # skip "[n] source" line
        blocks.append((int(m.group(1)), body))
    return blocks


class FakeLLM:
    """A deterministic, offline stand-in for a real LLM.

    Behaviour, in priority order:

    1. ``handler`` — if given, ``handler(messages)`` is returned.
    2. ``responses`` — canned replies returned in order (cycling when exhausted).
    3. ``mode="echo"`` (default) — returns ``prefix + <last user message>``.
    4. ``mode="extractive"`` — splits the last user message at a question marker
       (``پرسش:`` / ``Question:``) and returns the context sentence with the
       largest word overlap with the question. Handy for RAG demos: the "answer"
       is copied straight from the retrieved context. Options (lesson 09):
       ``cite=True`` appends `` [n]`` when the sentence comes from a context
       block that starts with ``[n]``; ``ignore`` is a set of words (e.g. stop
       words) that do not count as overlap; with fewer than ``min_overlap``
       shared words the reply is ``fallback`` — a stand-in for a model that
       follows an "answer «نمی‌دانم» if the context does not say" instruction.

    Every call is recorded in :attr:`calls` so tests can inspect prompts.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        mode: Literal["echo", "extractive"] = "echo",
        handler: Callable[[list[Message]], str] | None = None,
        prefix: str = "[FakeLLM] ",
        fallback: str = "نمی‌دانم.",
        cite: bool = False,
        ignore: Iterable[str] = (),
        min_overlap: int = 1,
    ) -> None:
        if responses is not None and len(responses) == 0:
            raise ValueError("responses must be non-empty when given")
        self.responses = list(responses) if responses is not None else None
        self.mode = mode
        self.handler = handler
        self.prefix = prefix
        self.fallback = fallback
        self.cite = cite
        self.ignore = frozenset(w.lower() for w in ignore)
        self.min_overlap = max(1, min_overlap)
        self.calls: list[list[Message]] = []

    def generate(self, messages: Sequence[MessageLike]) -> str:
        msgs = to_messages(messages)
        self.calls.append(msgs)
        if self.handler is not None:
            return self.handler(msgs)
        if self.responses is not None:
            return self.responses[(len(self.calls) - 1) % len(self.responses)]
        last_user = next((m.content for m in reversed(msgs) if m.role == "user"), "")
        if self.mode == "extractive":
            return self._extract(last_user)
        return f"{self.prefix}{last_user}"

    def _extract(self, prompt: str) -> str:
        context, question = prompt, prompt
        for marker in _QUESTION_MARKERS:
            idx = prompt.rfind(marker)
            if idx != -1:
                context, question = prompt[:idx], prompt[idx + len(marker) :]
                break
        q_words = _words(question, self.ignore)
        best, best_block, best_score = "", None, 0
        for block, text in _numbered_blocks(context):
            for sentence in _SENTENCE_SPLIT.split(text):
                sentence = sentence.strip()
                score = len(q_words & _words(sentence, self.ignore))
                if score > best_score:
                    best, best_block, best_score = sentence, block, score
        if not best or best_score < self.min_overlap:
            return self.fallback
        if self.cite and best_block is not None:
            return f"{best} [{best_block}]"
        return best


# ------------------------------------------------------------- OpenAI-compatible HTTP


class OpenAICompatibleLLM:
    """Client for any OpenAI-compatible chat-completions endpoint.

    ``httpx`` is imported lazily (install with ``pip install "parsrag[openai]"``).
    A pre-built client can be injected via ``client`` — tests use this to mock
    the HTTP layer so no network request is ever made.

    Example (local Ollama, no key needed)::

        llm = OpenAICompatibleLLM(base_url="http://localhost:11434/v1", model="qwen2.5:7b")
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError(
                    "OpenAICompatibleLLM needs httpx. "
                    "Install it with: pip install 'parsrag[openai]'"
                ) from exc
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def build_payload(self, messages: Sequence[MessageLike]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in to_messages(messages)],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def generate(self, messages: Sequence[MessageLike]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self._get_client().post(
            f"{self.base_url}/chat/completions",
            json=self.build_payload(messages),
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response from {self.base_url}: {data!r}") from exc

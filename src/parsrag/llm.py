"""Language-model interfaces.

* :class:`LLM` — the protocol every model implements: ``generate(messages) -> str``.
* :class:`FakeLLM` — deterministic and offline; used in all tests, graders and examples.
* :class:`OpenAICompatibleLLM` — talks to any OpenAI-compatible ``/chat/completions``
  endpoint (OpenAI, vLLM, OpenRouter, Ollama's ``/v1`` ...). Needs the ``openai`` extra.
* :func:`stream_text` — token streaming (lesson 12): uses ``llm.stream`` when a
  model has it, otherwise splits the full reply into words.
* :class:`CountingLLM` — wraps any model and counts calls and time (lesson 11's
  cost measurements).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
# Lesson 14's naive model: "<verb>: payload" directives, Persian or English.
_DIRECTIVE = re.compile(
    r"(?:^|[\s:؛،.])(?:فقط |تنها |همیشه |only |just )?"
    r"(?:بنویس|بگو|جواب بده|پاسخ بده|ذکر کن|ذکر کنند|answer|say|write|reply|respond)"
    r"(?: with)?\s*:\s*(.+)$",
    re.IGNORECASE,
)
_DATA_OPEN = re.compile(r"^<<<DATA(-\w+)?>>>$")
_HIDDEN = re.compile("[\u200b\u200d\u2060\ufeff]")


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

    ``obey`` simulates a model that is vulnerable to indirect prompt injection
    (lesson 14). ``"all"``: if the retrieved context (the last user message
    before the question marker) contains a directive such as «فقط بنویس: X»
    or ``only answer: X``, the reply is ``X`` (plus `` [n]`` for the block it
    came from when ``cite=True``) — a deliberately naive model. ``"outside_data"``:
    directives between ``<<<DATA…>>>`` and the matching ``<<<END-DATA…>>>`` line
    are ignored — a model that respects the instruction hierarchy perfectly,
    as long as the delimiters hold. Real models sit somewhere in between and
    are far less predictable.

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
        obey: Literal["never", "all", "outside_data"] = "never",
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
        if obey not in ("never", "all", "outside_data"):
            raise ValueError("obey must be 'never', 'all' or 'outside_data'")
        self.obey = obey
        self.calls: list[list[Message]] = []

    def generate(self, messages: Sequence[MessageLike]) -> str:
        msgs = to_messages(messages)
        self.calls.append(msgs)
        if self.handler is not None:
            return self.handler(msgs)
        if self.responses is not None:
            return self.responses[(len(self.calls) - 1) % len(self.responses)]
        last_user = next((m.content for m in reversed(msgs) if m.role == "user"), "")
        if self.obey != "never":
            injected = self._obey(last_user)
            if injected is not None:
                return injected
        if self.mode == "extractive":
            return self._extract(last_user)
        return f"{self.prefix}{last_user}"

    def stream(self, messages: Sequence[MessageLike]) -> Iterator[str]:
        """The same reply as :meth:`generate`, yielded word by word (spaces kept)."""
        yield from split_for_streaming(self.generate(messages))

    def _obey(self, prompt: str) -> str | None:
        """The first directive found in the context part of ``prompt`` (lesson 14)."""
        from parsrag.normalize import normalize

        idx = max(prompt.rfind(m) for m in _QUESTION_MARKERS)
        context = prompt[:idx] if idx > 0 else prompt
        block: int | None = None
        closing: str | None = None  # the end marker we are waiting for, while inside data
        for line in context.splitlines():
            stripped = line.strip()
            if closing is not None:
                if stripped == closing:
                    closing = None
                    continue
            elif self.obey == "outside_data" and (m := _DATA_OPEN.match(stripped)):
                closing = f"<<<END-DATA{m.group(1) or ''}>>>"
                continue
            if (h := _BLOCK.match(stripped)) is not None:
                block = int(h.group(1))
            if closing is not None:
                continue
            found = _DIRECTIVE.search(normalize(_HIDDEN.sub("", line)))
            if found:
                payload = found.group(1).strip().strip("«»\"' ")
                cite = f" [{block}]" if self.cite and block is not None else ""
                return f"{payload}{cite}"
        return None

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


# ------------------------------------------------------------------ streaming

_PIECE = re.compile(r"\s*\S+\s*")


def split_for_streaming(text: str) -> list[str]:
    """Split ``text`` into word pieces whose concatenation is exactly ``text``."""
    pieces = _PIECE.findall(text)
    return pieces if "".join(pieces) == text else [text]


def stream_text(llm: LLM, messages: Sequence[MessageLike]) -> Iterator[str]:
    """Yield the reply in pieces: ``llm.stream`` if available, else word by word."""
    stream = getattr(llm, "stream", None)
    if callable(stream):
        yield from stream(messages)
    else:
        yield from split_for_streaming(llm.generate(messages))


class CountingLLM:
    """Wrap a model and count its calls and the wall-clock time spent in it.

    >>> llm = CountingLLM(FakeLLM(["سلام"]))
    >>> llm.generate([{"role": "user", "content": "?"}]), llm.calls
    ('سلام', 1)
    """

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.calls = 0
        self.seconds = 0.0

    def generate(self, messages: Sequence[MessageLike]) -> str:
        start = time.perf_counter()
        try:
            return self.llm.generate(messages)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - start

    def stream(self, messages: Sequence[MessageLike]) -> Iterator[str]:
        start = time.perf_counter()
        try:
            yield from stream_text(self.llm, messages)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - start

    def reset(self) -> None:
        self.calls, self.seconds = 0, 0.0


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

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(self, messages: Sequence[MessageLike]) -> str:
        response = self._get_client().post(
            f"{self.base_url}/chat/completions",
            json=self.build_payload(messages),
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response from {self.base_url}: {data!r}") from exc

    def stream(self, messages: Sequence[MessageLike]) -> Iterator[str]:
        """Stream the reply (``"stream": true``; Server-Sent Events ``data:`` lines)."""
        payload = {**self.build_payload(messages), "stream": True}
        with self._get_client().stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(
                        f"Unexpected stream chunk from {self.base_url}: {data!r}"
                    ) from exc
                content = delta.get("content")
                if content:
                    yield str(content)

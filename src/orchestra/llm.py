"""LLM client abstraction.

The orchestrator never talks to a vendor SDK directly. It only needs an object
that implements :class:`LLMClient`, which makes swapping providers a one-line
configuration change and keeps the tests hermetic.

Two clients ship with the project:

``OpenAICompatibleClient``
    Works with any endpoint that speaks the OpenAI ``/chat/completions``
    protocol (OpenAI, DeepSeek, Moonshot, Qwen, vLLM, Ollama, ...).

``orchestra.mock_llm.MockClient``
    Returns scripted, deterministic answers. It is what makes the demo, the
    test suite and CI run without a network connection or an API key.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

Message = dict[str, str]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(RuntimeError):
    """Raised when the upstream model call fails."""


@dataclass(slots=True)
class Completion:
    """A single assistant message plus light metadata for tracing."""

    text: str
    model: str = "unknown"


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface every model backend has to implement."""

    name: str

    def complete(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> Completion:
        """Return one assistant message for the given conversation."""


class OpenAICompatibleClient:
    """Tiny ``/chat/completions`` client built on :mod:`urllib`.

    Configuration is resolved in this order: explicit argument, environment
    variable, built-in default.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.getenv("ORCHESTRA_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("ORCHESTRA_API_KEY", "")
        self.base_url = (base_url or os.getenv("ORCHESTRA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self.name = f"openai-compatible:{self.model}"

    def complete(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> Completion:
        if not self.api_key:
            raise LLMError(
                "Missing API key. Set ORCHESTRA_API_KEY, or run with --provider mock."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"Model endpoint returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network dependent
            raise LLMError(f"Cannot reach {self.base_url}: {exc.reason}") from exc

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - network dependent
            raise LLMError(f"Unexpected response shape: {body}") from exc
        return Completion(text=text, model=body.get("model", self.model))

"""LLM client abstraction.

The orchestrator never talks to a vendor SDK directly. It only needs an object
that implements :class:`LLMClient`, which makes swapping providers a one-line
configuration change and keeps the tests hermetic.

Two clients ship with the project:

``OpenAICompatibleClient``
    Works with any endpoint that speaks the OpenAI ``/chat/completions``
    protocol: OpenAI, DeepSeek, Moonshot, DashScope, vLLM, Ollama, or an
    internal gateway. Hosted providers have presets; everything else is just
    ``ORCHESTRA_BASE_URL``.

``orchestra.mock_llm.MockClient``
    Returns scripted, deterministic answers. It is what makes the demo, the
    test suite and CI run without a network connection or an API key.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

Message = dict[str, str]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# Hosted endpoints that speak the OpenAI chat-completions protocol verbatim.
# Anything else (vLLM, Ollama, a company gateway) is configuration, not code.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {"base_url": DEFAULT_BASE_URL, "model": DEFAULT_MODEL},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}

# Rate limits, gateway hiccups and transient 5xx are worth another attempt.
# Auth errors and bad payloads are not: retrying them only burns time.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Raised when the upstream model call fails."""


class LLMRetryableError(LLMError):
    """A failure worth retrying: rate limit, 5xx, or a network blip."""


@dataclass(slots=True)
class Completion:
    """A single assistant message plus light metadata for tracing and costing."""

    text: str
    model: str = "unknown"
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface every model backend has to implement."""

    name: str

    def complete(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> Completion:
        """Return one assistant message for the given conversation."""


def _error_message(exc: urllib.error.HTTPError) -> str:
    """Pull the human-readable part out of an error body.

    Providers disagree on the shape, so try the OpenAI envelope first and fall
    back to the raw text instead of dumping a wall of JSON into the trace.
    """
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - the body may already be gone
        return str(exc.reason or "no response body")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()[:300] or "no response body"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])[:300]
    return raw.strip()[:300] or "no response body"


class OpenAICompatibleClient:
    """Tiny ``/chat/completions`` client built on :mod:`urllib`.

    Configuration is resolved in this order: explicit argument, environment
    variable, provider preset, built-in default. Passing ``provider="deepseek"``
    is therefore enough to get the right base URL and model name.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        provider: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        json_mode: bool = False,
    ) -> None:
        preset = PROVIDER_PRESETS.get(provider or "", {})
        self.provider = provider or "custom"
        self.model = model or os.getenv("ORCHESTRA_MODEL") or preset.get("model") or DEFAULT_MODEL
        self.api_key = api_key or os.getenv("ORCHESTRA_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("ORCHESTRA_BASE_URL")
            or preset.get("base_url")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.json_mode = json_mode
        self.name = f"{self.provider}:{self.model}"

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
            "stream": False,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        body = self._post(payload)
        try:
            message = body["choices"][0]["message"]
            text = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {body}") from exc
        usage = {
            key: value
            for key, value in (body.get("usage") or {}).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        return Completion(text=text, model=body.get("model", self.model), usage=usage)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "orchestra/0.1",
            },
            method="POST",
        )
        last_error: LLMRetryableError | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                message = f"{self.name} returned HTTP {exc.code}: {_error_message(exc)}"
                if exc.code not in RETRYABLE_STATUS:
                    raise LLMError(message) from exc
                last_error = LLMRetryableError(message)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                last_error = LLMRetryableError(f"Cannot reach {self.base_url}: {reason}")
        if last_error is None:  # pragma: no cover - defensive
            raise LLMError("Model call failed without an error")
        raise last_error

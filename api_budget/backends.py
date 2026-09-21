"""Provider backends.

Groq, Ollama and any future frontier provider all speak the OpenAI wire
protocol, so one backend class with a configurable ``base_url`` covers all
three. That is the whole reason for choosing this shape: the zero-spend
decision in README section 5 stays a config change rather than a rewrite.

(If Anthropic is ever added, it gets its own backend using the ``anthropic``
SDK rather than an OpenAI-compatible shim -- shims lag behind the real API and
silently drop features.)

The mock backend is the default in tests. It is deterministic and never touches
the network, so the suite runs offline at zero cost.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol


class RateLimitError(RuntimeError):
    """Provider said 429. Retryable, and the trigger for the local fallback."""


class BackendError(RuntimeError):
    """Any other provider failure."""


@dataclass(frozen=True)
class BackendResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class Backend(Protocol):
    name: str

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        system: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> BackendResponse: ...


# --------------------------------------------------------------------------- #
# OpenAI-compatible
# --------------------------------------------------------------------------- #


class OpenAICompatibleBackend:
    """Groq, Ollama, or anything else speaking the OpenAI chat-completions API."""

    def __init__(self, name: str, base_url: str, api_key: str, *, timeout: float = 120.0):
        self.name = name
        self.base_url = base_url
        self._timeout = timeout
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install
            raise BackendError(
                "the `openai` package is required for live providers: pip install openai"
            ) from exc
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        system: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> BackendResponse:
        # Retries are handled by the client layer, not here, so that the
        # fallback decision and the cost log see every individual attempt.
        import openai

        full: list[dict[str, Any]] = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code == 429:
                raise RateLimitError(str(exc)) from exc
            raise BackendError(f"{self.name} returned {exc.status_code}: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise BackendError(f"{self.name} unreachable at {self.base_url}: {exc}") from exc

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return BackendResponse(
            text=choice.message.content or "",
            model=resp.model or model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #


class MockBackend:
    """Deterministic offline backend.

    Returns a stable, content-derived response so tests are reproducible and
    never touch the network. It also honours a JSON response_format by emitting
    parseable JSON, which lets the structured-output call paths be tested
    without a provider.

    Scripted replies can be registered for specific prompt substrings, which is
    how a component's real parsing logic gets exercised against a known answer.
    """

    name = "mock"

    def __init__(self, *, scripted: dict[str, str] | None = None, fail_with: Exception | None = None):
        self.scripted = scripted or {}
        self.fail_with = fail_with
        self.calls: list[dict[str, Any]] = []

    def register(self, substring: str, reply: str) -> None:
        self.scripted[substring] = reply

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        system: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> BackendResponse:
        self.calls.append(
            {"model": model, "messages": messages, "system": system,
             "temperature": temperature, "max_tokens": max_tokens,
             "response_format": response_format}
        )
        if self.fail_with is not None:
            raise self.fail_with

        joined = json.dumps(messages, sort_keys=True) + (system or "")
        for needle, reply in self.scripted.items():
            if needle in joined:
                return self._wrap(reply, model, joined)

        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
        if response_format and response_format.get("type") == "json_object":
            text = json.dumps({"mock": True, "digest": digest}, sort_keys=True)
        else:
            text = f"[mock:{digest}]"
        return self._wrap(text, model, joined)

    @staticmethod
    def _wrap(text: str, model: str, prompt: str) -> BackendResponse:
        # Rough token estimate. Good enough for exercising the accounting path;
        # real numbers come from the provider's usage block.
        return BackendResponse(
            text=text,
            model=model,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
        )

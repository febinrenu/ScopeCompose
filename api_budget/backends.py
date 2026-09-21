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


class TruncatedResponseError(BackendError):
    """The model hit ``max_tokens`` before finishing.

    Raised whether the response came back empty or merely cut off. Both are
    the same failure for this pipeline: every caller wants a complete
    structured object, and JSON truncated mid-string is garbage rather than an
    obvious error.

    Reasoning models such as gpt-oss make this easy to hit. The ``reasoning``
    field is generated first and its tokens count against ``max_tokens``, so a
    limit that looks generous for the *answer* can be consumed before the
    answer starts.

    It gets its own type because the correct response is to raise
    ``max_tokens`` or lower ``reasoning_effort`` -- not to retry, which would
    burn three more calls against a daily cap and fail identically. Returning
    the empty string instead would be worse still: every downstream JSON parse
    would quietly fall back, and a whole corpus run would produce plausible
    but empty extractions.
    """


@dataclass(frozen=True)
class BackendResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None
    reasoning_tokens: int = 0
    """Tokens spent on reasoning rather than the answer. Tracked separately
    because on a capped free tier they are a real and otherwise invisible
    drain on the daily budget."""


#: Model families that accept a ``reasoning_effort`` parameter. Sending it to
#: one that does not (allam-2-7b) is a 400, which would turn an overflow
#: fallback into a hard failure at exactly the moment throughput matters.
_REASONING_EFFORT_FAMILIES = ("gpt-oss", "qwen")


def supports_reasoning_effort(model: str) -> bool:
    low = model.lower()
    return any(fam in low for fam in _REASONING_EFFORT_FAMILIES)


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
        reasoning_effort: str | None = None,
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
        reasoning_effort: str | None = None,
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
        if reasoning_effort and supports_reasoning_effort(model):
            # Supported by gpt-oss and qwen on Groq; allam-2-7b rejects it with
            # a 400. The guard matters on the overflow path, where a request
            # built for one model gets re-sent to another.
            kwargs["reasoning_effort"] = reasoning_effort

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
        text = choice.message.content or ""
        finish = getattr(choice, "finish_reason", None)

        # Reasoning models put their chain in a separate field and charge its
        # tokens to the completion. Measure it so the drain is visible.
        reasoning = getattr(choice.message, "reasoning", None) or ""
        reasoning_tokens = len(reasoning) // 4 if reasoning else 0

        if finish == "length":
            # Raised whether or not any content came back. Partial output is
            # not a lesser problem than empty output here: every caller in
            # this pipeline wants a complete structured object, and a JSON
            # document cut off mid-string parses as garbage rather than as an
            # obvious failure.
            got = f"got {text.strip()[:40]!r}" if text.strip() else "produced no content"
            raise TruncatedResponseError(
                f"{model} hit max_tokens={max_tokens} and {got}"
                + (f"; ~{reasoning_tokens} of those tokens went to reasoning" if reasoning else "")
                + ". Raise max_tokens for this tier in config/models.yaml, or lower "
                  "reasoning_effort. Retrying would fail identically."
            )

        return BackendResponse(
            text=text,
            model=resp.model or model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=finish,
            reasoning_tokens=reasoning_tokens,
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
        reasoning_effort: str | None = None,
    ) -> BackendResponse:
        self.calls.append(
            {"model": model, "messages": messages, "system": system,
             "temperature": temperature, "max_tokens": max_tokens,
             "response_format": response_format, "reasoning_effort": reasoning_effort}
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

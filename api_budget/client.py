"""The single entry point for every LLM call in the project.

No module may call a provider SDK directly. Going through here is what
guarantees three things that are individually easy to skip and collectively
decide whether the budget discipline works:

1. **Caching.** The second run of any script is free.
2. **Cost logging.** Every call is attributed to a pipeline step.
3. **Tier indirection.** Code asks for ``Tier.BULK``; ``config/models.yaml``
   decides which model that is.

Usage::

    from api_budget import complete, Tier

    result = complete(
        messages=[{"role": "user", "content": prompt}],
        tier=Tier.BULK,
        step="a2_stage2_detection",
    )
    print(result.text, result.cached)
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import settings
from api_budget.backends import (
    Backend,
    BackendError,
    MockBackend,
    OpenAICompatibleBackend,
    RateLimitError,
)
from api_budget.cache import RequestCache, request_key
from api_budget.costlog import CostLog


class Tier(str, Enum):
    """Logical model tiers. Never name a concrete model in a module."""

    BULK = "BULK"
    """High volume, low difficulty. Stage-2 detection, descriptor extraction,
    counterfactual question generation, candidate-condition proposal."""

    JUDGE = "JUDGE"
    """Load-bearing. The PR/SR/HCR/SCR metric judge and final scoped-answer
    generation. Validated against human labels before its numbers are trusted."""

    LONG_CONTEXT = "LONG_CONTEXT"
    """The structured long-context baseline -- the decisive experiment.
    Read README section 5 before interpreting anything it produces."""

    SECOND_BACKBONE = "SECOND_BACKBONE"
    """Robustness ablation. Must be a different model family from JUDGE."""


class TierNotConfigured(RuntimeError):
    """A tier has no model assigned yet."""


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str
    provider: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    fell_back: bool
    latency_s: float
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def json(self) -> Any:
        """Parse the response as JSON.

        Raises a clear error rather than a bare JSONDecodeError, because a
        model returning prose where JSON was requested is a prompt problem and
        the message should say so.
        """
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as exc:
            snippet = self.text[:200].replace("\n", " ")
            raise ValueError(
                f"expected JSON from {self.model} but got: {snippet!r}"
            ) from exc


class LLMClient:
    """Cached, logged, tier-routed LLM client."""

    def __init__(
        self,
        *,
        cache: RequestCache | None = None,
        cost_log: CostLog | None = None,
        backend_override: Backend | None = None,
        max_retries: int = 4,
        base_backoff: float = 1.0,
        seed: int = 0,
    ):
        self.cache = cache if cache is not None else RequestCache()
        self.cost_log = cost_log if cost_log is not None else CostLog()
        self.backend_override = backend_override
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._rng = random.Random(seed)
        self._backends: dict[str, Backend] = {}

    # -- backend resolution -------------------------------------------------- #

    def _backend(self, provider_name: str) -> Backend:
        if self.backend_override is not None:
            return self.backend_override
        if provider_name in self._backends:
            return self._backends[provider_name]

        spec = settings.models().provider(provider_name)
        if not spec.configured:
            raise BackendError(
                f"provider {provider_name!r} is not configured: base_url="
                f"{spec.base_url!r}, api_key={'set' if spec.api_key else 'MISSING'}. "
                f"Copy .env.example to .env and fill it in."
            )
        backend = OpenAICompatibleBackend(provider_name, spec.base_url, spec.api_key)
        self._backends[provider_name] = backend
        return backend

    def _resolve_tier(self, tier: Tier | str) -> tuple[str, str, float, int]:
        name = tier.value if isinstance(tier, Tier) else str(tier).upper()
        spec = settings.models().tier(name)
        if not spec.resolved:
            raise TierNotConfigured(
                f"tier {name} has no model assigned in config/models.yaml. "
                f"Run `python scripts/discover_models.py` to populate it from your "
                f"account's real model list."
            )
        return spec.provider, spec.model, spec.temperature, spec.max_tokens

    # -- the call ------------------------------------------------------------ #

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tier: Tier | str = Tier.BULK,
        step: str = "unspecified",
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        use_cache: bool = True,
        model: str | None = None,
        provider: str | None = None,
    ) -> CompletionResult:
        """Make one completion request.

        ``step`` is not optional in spirit: without it the cost log cannot
        attribute spend to a pipeline component, which is the reason the log
        exists.
        """
        tier_name = tier.value if isinstance(tier, Tier) else str(tier).upper()

        if model is None or provider is None:
            p, m, t, mt = self._resolve_tier(tier_name)
            provider = provider or p
            model = model or m
            temperature = t if temperature is None else temperature
            max_tokens = mt if max_tokens is None else max_tokens
        else:
            temperature = 0.0 if temperature is None else temperature
            max_tokens = 1024 if max_tokens is None else max_tokens

        key = request_key(
            provider=provider, model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            system=system, response_format=response_format,
        )

        # -- cache ----------------------------------------------------------- #
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.cost_log.log(
                    step=step, tier=tier_name, provider=hit.provider, model=hit.model,
                    prompt_tokens=hit.prompt_tokens, completion_tokens=hit.completion_tokens,
                    cached=True,
                )
                return CompletionResult(
                    text=hit.text, model=hit.model, provider=hit.provider, tier=tier_name,
                    prompt_tokens=hit.prompt_tokens, completion_tokens=hit.completion_tokens,
                    cached=True, fell_back=False, latency_s=0.0, cost_usd=0.0,
                )

        # -- live call, with retry then fallback ------------------------------ #
        cfg = settings.models()
        started = time.time()
        last_exc: Exception | None = None
        rate_limited = 0

        for attempt in range(self.max_retries):
            try:
                backend = self._backend(provider)
                resp = backend.complete(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, system=system, response_format=response_format,
                )
                return self._finish(
                    key=key, step=step, tier_name=tier_name, provider=provider,
                    resp=resp, started=started, fell_back=False, use_cache=use_cache,
                    request={"messages": messages, "system": system, "temperature": temperature},
                )
            except RateLimitError as exc:
                last_exc = exc
                rate_limited += 1
                self._sleep(attempt)
            except BackendError as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                self._sleep(attempt)

        # Every retry exhausted. If the failure was rate-limiting and a local
        # fallback is configured, use it -- and mark the result so the
        # substitution shows up in the report instead of silently changing
        # what model produced a number.
        if (
            rate_limited >= cfg.fallback.trigger_after_retries
            and cfg.fallback.enabled
            and cfg.fallback.model
            and self.backend_override is None
        ):
            try:
                fb_backend = self._backend(cfg.fallback.provider)
                resp = fb_backend.complete(
                    model=cfg.fallback.model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, system=system, response_format=response_format,
                )
                return self._finish(
                    key=key, step=step, tier_name=tier_name, provider=cfg.fallback.provider,
                    resp=resp, started=started, fell_back=True, use_cache=use_cache,
                    request={"messages": messages, "system": system, "temperature": temperature},
                )
            except (RateLimitError, BackendError) as exc:
                last_exc = exc

        self.cost_log.log(
            step=step, tier=tier_name, provider=provider, model=model,
            latency_s=time.time() - started, error=str(last_exc),
        )
        raise BackendError(
            f"{step}: all {self.max_retries} attempts against {provider}/{model} failed "
            f"({last_exc})"
        ) from last_exc

    def _finish(self, *, key, step, tier_name, provider, resp, started, fell_back,
                use_cache, request) -> CompletionResult:
        latency = time.time() - started
        in_price, out_price = settings.models().price(resp.model)
        cost = (resp.prompt_tokens / 1e6) * in_price + (resp.completion_tokens / 1e6) * out_price

        if use_cache:
            self.cache.put(
                key, provider=provider, model=resp.model, request=request,
                text=resp.text, prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
            )

        self.cost_log.log(
            step=step, tier=tier_name, provider=provider, model=resp.model,
            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
            cached=False, fell_back=fell_back, cost_usd=cost, latency_s=latency,
        )
        return CompletionResult(
            text=resp.text, model=resp.model, provider=provider, tier=tier_name,
            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
            cached=False, fell_back=fell_back, latency_s=latency, cost_usd=cost,
        )

    def _sleep(self, attempt: int) -> None:
        """Exponential backoff with jitter.

        Jitter matters on a free tier: without it, a batch of parallel workers
        that all get 429'd retry in lockstep and trip the limit again.
        """
        delay = self.base_backoff * (2 ** attempt) * (0.5 + self._rng.random())
        time.sleep(min(delay, 30.0))


# --------------------------------------------------------------------------- #
# Module-level default
# --------------------------------------------------------------------------- #

_default: LLMClient | None = None


def get_client() -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient()
    return _default


def set_client(client: LLMClient | None) -> None:
    """Swap the default client. Used by tests to install a mock backend."""
    global _default
    _default = client


def complete(**kwargs: Any) -> CompletionResult:
    """Convenience wrapper over the default client."""
    return get_client().complete(**kwargs)


def mock_client(**kwargs: Any) -> LLMClient:
    """A client wired to the deterministic mock backend, with caching off.

    Caching is off by default so a test sees every call the code makes rather
    than a cached echo of the first one.
    """
    kwargs.setdefault("backend_override", MockBackend())
    kwargs.setdefault("cache", RequestCache(enabled=False))
    kwargs.setdefault("cost_log", CostLog(enabled=False))
    return LLMClient(**kwargs)

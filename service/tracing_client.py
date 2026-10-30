"""Every LLM call, visible.

Installed once as the process-wide client (``api_budget.set_client``). It is
an :class:`~api_budget.client.LLMClient` -- same cache, cost log, tiers and
fallback -- that additionally reports each call to the run that made it: the
tracer of the calling thread. Calls made outside a run are untouched.
"""

from __future__ import annotations

import time
from typing import Any

from api_budget.client import CompletionResult, LLMClient, get_client, set_client
from service.events import current, current_stage

#: How much of a prompt or response travels in the event. Enough to read what
#: was asked and answered, without shipping whole passages twice.
EXCERPT = 1200


def _excerpt(text: str | None) -> str:
    text = text or ""
    return text if len(text) <= EXCERPT else text[:EXCERPT] + " …"


class TracingLLMClient(LLMClient):
    def complete(self, **kwargs: Any) -> CompletionResult:  # type: ignore[override]
        tracer = current()
        if tracer is None:
            return super().complete(**kwargs)

        stage = current_stage()
        messages = kwargs.get("messages") or []
        prompt = "\n\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        step = kwargs.get("step", "unspecified")
        tier = kwargs.get("tier")
        tier_name = getattr(tier, "value", tier) or "BULK"
        t0 = time.perf_counter()
        try:
            result = super().complete(**kwargs)
        except Exception as exc:
            tracer.record_llm(prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
                              cached=False, fell_back=False, error=True)
            tracer.emit(stage, "llm", f"{step} failed", {
                "step": step, "tier": tier_name, "error": f"{type(exc).__name__}: {exc}"[:300],
                "system": _excerpt(kwargs.get("system")), "prompt": _excerpt(prompt),
                "ms": (time.perf_counter() - t0) * 1000.0,
            })
            raise

        tracer.record_llm(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                          cost_usd=result.cost_usd, cached=result.cached, fell_back=result.fell_back)
        tracer.emit(stage, "llm", step, {
            "step": step, "tier": result.tier, "provider": result.provider, "model": result.model,
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "cost_usd": result.cost_usd, "cached": result.cached, "fell_back": result.fell_back,
            "latency_s": result.latency_s, "ms": (time.perf_counter() - t0) * 1000.0,
            "system": _excerpt(kwargs.get("system")), "prompt": _excerpt(prompt),
            "response": _excerpt(result.text),
        })
        return result


def install() -> TracingLLMClient:
    """Make the tracing client the default (idempotent)."""
    existing = get_client()
    if isinstance(existing, TracingLLMClient):
        return existing
    client = TracingLLMClient(cache=existing.cache, cost_log=existing.cost_log,
                              backend_override=existing.backend_override)
    set_client(client)
    return client

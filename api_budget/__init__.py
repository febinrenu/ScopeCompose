"""Cached, logged, tier-routed access to LLM providers.

Every LLM call in the project goes through here. Calling a provider SDK
directly from a module bypasses the cache and the cost log, which is how a
budget or rate-limit problem gets discovered after a full-corpus run instead of
before one.
"""

from api_budget.backends import (
    BackendError,
    MockBackend,
    RateLimitError,
    TruncatedResponseError,
)
from api_budget.cache import RequestCache, request_key
from api_budget.client import (
    CompletionResult,
    LLMClient,
    Tier,
    TierNotConfigured,
    complete,
    get_client,
    mock_client,
    set_client,
)
from api_budget.costlog import CallRecord, CostLog, render, summarise

__all__ = [
    "BackendError",
    "CallRecord",
    "CompletionResult",
    "CostLog",
    "LLMClient",
    "MockBackend",
    "RateLimitError",
    "RequestCache",
    "TruncatedResponseError",
    "Tier",
    "TierNotConfigured",
    "complete",
    "get_client",
    "mock_client",
    "render",
    "request_key",
    "set_client",
    "summarise",
]

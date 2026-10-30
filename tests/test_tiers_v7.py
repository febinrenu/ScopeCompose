"""The v7 tiers: pinned (no silent substitution) and evaluated/evaluation-only apart."""

from __future__ import annotations

import pytest

import settings
from api_budget.backends import MockBackend, RateLimitError
from api_budget.cache import RequestCache
from api_budget.client import LLMClient, Tier
from api_budget.costlog import CostLog

EVALUATED = ("BACKBONE_PRIMARY", "BACKBONE_LOCAL")
EVALUATION_ONLY = ("READER", "EVAL")
MSG = [{"role": "user", "content": "x"}]


@pytest.fixture(autouse=True)
def _fresh_config():
    settings.reload()
    yield
    settings.reload()


def test_all_v7_tiers_exist_and_resolve():
    cfg = settings.models()
    for name in EVALUATED + EVALUATION_ONLY:
        assert cfg.tier(name).resolved, f"{name} has no model"
        assert Tier(name).value == name


def test_every_v7_tier_is_pinned():
    """A reported number must come from the model the paper names."""
    cfg = settings.models()
    for name in EVALUATED + EVALUATION_ONLY:
        assert cfg.tier(name).allow_fallback is False, f"{name} can be silently substituted"


def test_legacy_tiers_keep_their_fallback():
    cfg = settings.models()
    for name in ("BULK", "JUDGE", "LONG_CONTEXT", "SECOND_BACKBONE"):
        assert cfg.tier(name).allow_fallback is True


def test_an_evaluated_model_is_never_an_evaluation_only_model():
    """The separation the design rests on, as a check rather than a convention."""
    cfg = settings.models()
    evaluated = {cfg.tier(n).model for n in EVALUATED}
    evaluation_only = {cfg.tier(n).model for n in EVALUATION_ONLY}
    assert not evaluated & evaluation_only
    # ... and the two groups are not served by the same provider model family
    # in a way that would let one grade the other's homework.
    assert not any("qwen" in m for m in evaluation_only & {"scopecompose-qwen7b"})


def test_reader_is_local_and_has_a_pinned_context():
    t = settings.models().tier("READER")
    assert t.provider == "ollama"
    assert t.reasoning_effort is None
    modelfile = settings.REPO_ROOT / "config" / "ollama" / "reader.Modelfile"
    text = modelfile.read_text(encoding="utf-8")
    assert "num_ctx" in text and "FROM llama3.1:8b" in text


def test_local_backbone_modelfile_pins_context_and_seed():
    text = (settings.REPO_ROOT / "config" / "ollama" / "backbone7b.Modelfile").read_text(encoding="utf-8")
    assert "num_ctx 8192" in text and "seed 0" in text


def _client(tmp_path, backends: dict) -> LLMClient:
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                  max_retries=3, base_backoff=0.0)
    c._backend = lambda name: backends[name]  # type: ignore[method-assign]
    return c


def test_a_rate_limited_unpinned_tier_falls_back_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "local-fallback")
    settings.reload()
    groq = MockBackend(fail_with=RateLimitError("429"))
    ollama = MockBackend()
    r = _client(tmp_path, {"groq": groq, "ollama": ollama}).complete(
        messages=MSG, tier=Tier.JUDGE, step="t")
    assert r.fell_back is True
    assert len(ollama.calls) == 1


def test_a_rate_limited_pinned_tier_raises_instead_of_falling_back(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "local-fallback")
    settings.reload()
    groq = MockBackend(fail_with=RateLimitError("429"))
    ollama = MockBackend()
    with pytest.raises(Exception, match="all 3 attempts"):
        _client(tmp_path, {"groq": groq, "ollama": ollama}).complete(
            messages=MSG, tier=Tier.EVAL, step="t")
    assert len(ollama.calls) == 0, "a pinned tier must never be answered by another model"
    assert len(groq.calls) == 3


def test_primary_backbone_is_pinned_too(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "local-fallback")
    settings.reload()
    ollama = MockBackend()
    with pytest.raises(Exception):
        _client(tmp_path, {"groq": MockBackend(fail_with=RateLimitError("429")),
                           "ollama": ollama}).complete(
            messages=MSG, tier=Tier.BACKBONE_PRIMARY, step="t")
    assert ollama.calls == []

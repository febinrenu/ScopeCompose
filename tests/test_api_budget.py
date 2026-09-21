"""Tests for the API budget layer.

Every test here runs OFFLINE against the mock backend. If any of these ever
needs the network, that is a bug in the test, not a reason to mark it skip:
the suite must be runnable at zero cost and zero rate-limit consumption, or
people stop running it.
"""

from __future__ import annotations

import json

import pytest

import settings
from api_budget.backends import BackendError, MockBackend, RateLimitError
from api_budget.cache import RequestCache, request_key
from api_budget.client import (
    CompletionResult,
    LLMClient,
    Tier,
    TierNotConfigured,
    mock_client,
)
from api_budget.costlog import CallRecord, CostLog, render, summarise


# --------------------------------------------------------------------------- #
# Cache keys
# --------------------------------------------------------------------------- #


def _key(**over):
    base = dict(
        provider="groq", model="m1",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0, max_tokens=100,
    )
    base.update(over)
    return request_key(**base)


def test_identical_requests_share_a_key():
    assert _key() == _key()


def test_key_is_insensitive_to_dict_ordering():
    """Python dicts are insertion-ordered, so two semantically identical
    requests built in different orders must still hash the same -- otherwise
    they silently miss the cache."""
    a = request_key(provider="groq", model="m", messages=[{"role": "user", "content": "x"}],
                    temperature=0.0, max_tokens=10)
    b = request_key(max_tokens=10, temperature=0.0,
                    messages=[{"content": "x", "role": "user"}], model="m", provider="groq")
    assert a == b


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "m2"),
        ("provider", "ollama"),
        ("temperature", 0.7),
        ("max_tokens", 200),
        ("messages", [{"role": "user", "content": "different"}]),
        ("system", "a system prompt"),
        ("response_format", {"type": "json_object"}),
    ],
)
def test_every_response_affecting_field_changes_the_key(field, value):
    """Anything omitted from the key is a correctness bug: the cache would
    return a response generated under different settings."""
    assert _key() != _key(**{field: value})


# --------------------------------------------------------------------------- #
# Cache behaviour
# --------------------------------------------------------------------------- #


@pytest.fixture
def cache(tmp_path):
    c = RequestCache(tmp_path / "c.sqlite")
    yield c
    c.close()


def test_miss_then_hit(cache):
    k = _key()
    assert cache.get(k) is None
    cache.put(k, provider="groq", model="m1", request={}, text="hi",
              prompt_tokens=10, completion_tokens=3)
    got = cache.get(k)
    assert got is not None
    assert got.text == "hi"
    assert got.prompt_tokens == 10
    assert got.completion_tokens == 3
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_persists_across_instances(tmp_path):
    """The whole value proposition: the SECOND RUN of a script is free."""
    path = tmp_path / "c.sqlite"
    with RequestCache(path) as c1:
        c1.put(_key(), provider="groq", model="m1", request={}, text="persisted")
    with RequestCache(path) as c2:
        assert c2.get(_key()).text == "persisted"


def test_disabled_cache_never_stores(tmp_path):
    c = RequestCache(tmp_path / "c.sqlite", enabled=False)
    c.put(_key(), provider="groq", model="m", request={}, text="x")
    assert c.get(_key()) is None


def test_clear_scoped_to_one_model(cache):
    cache.put(_key(model="a"), provider="groq", model="a", request={}, text="1")
    cache.put(_key(model="b"), provider="groq", model="b", request={}, text="2")
    assert cache.clear(model="a") == 1
    assert cache.get(_key(model="a")) is None
    assert cache.get(_key(model="b")) is not None


def test_stats_report_hit_rate(cache):
    cache.put(_key(), provider="groq", model="m1", request={}, text="x",
              prompt_tokens=5, completion_tokens=2)
    cache.get(_key())
    cache.get(_key(model="absent"))
    s = cache.stats()
    assert s["entries"] == 1
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["hit_rate"] == 0.5
    assert s["cached_prompt_tokens"] == 5


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #


def test_mock_backend_is_deterministic():
    b = MockBackend()
    msgs = [{"role": "user", "content": "same"}]
    r1 = b.complete(model="m", messages=msgs, temperature=0.0, max_tokens=10)
    r2 = b.complete(model="m", messages=msgs, temperature=0.0, max_tokens=10)
    assert r1.text == r2.text


def test_mock_backend_varies_with_content():
    b = MockBackend()
    a = b.complete(model="m", messages=[{"role": "user", "content": "a"}],
                   temperature=0.0, max_tokens=10)
    z = b.complete(model="m", messages=[{"role": "user", "content": "z"}],
                   temperature=0.0, max_tokens=10)
    assert a.text != z.text


def test_mock_backend_honours_json_response_format():
    b = MockBackend()
    r = b.complete(model="m", messages=[{"role": "user", "content": "x"}],
                   temperature=0.0, max_tokens=10,
                   response_format={"type": "json_object"})
    assert json.loads(r.text)


def test_mock_backend_scripted_replies():
    """Lets a component's real parsing logic run against a known answer."""
    b = MockBackend(scripted={"premium": '{"scope_relation": "refinement"}'})
    r = b.complete(model="m", messages=[{"role": "user", "content": "is premium exempt?"}],
                   temperature=0.0, max_tokens=10)
    assert json.loads(r.text)["scope_relation"] == "refinement"


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


def test_client_returns_a_result():
    c = mock_client()
    r = c.complete(messages=[{"role": "user", "content": "hi"}],
                   tier=Tier.BULK, step="t", model="m", provider="mock")
    assert isinstance(r, CompletionResult)
    assert r.text
    assert not r.cached
    assert not r.fell_back
    assert r.total_tokens > 0


def test_client_serves_second_identical_call_from_cache(tmp_path):
    backend = MockBackend()
    c = LLMClient(
        cache=RequestCache(tmp_path / "c.sqlite"),
        cost_log=CostLog(tmp_path / "log.jsonl"),
        backend_override=backend,
    )
    kw = dict(messages=[{"role": "user", "content": "x"}], tier=Tier.BULK,
              step="t", model="m", provider="mock")

    first = c.complete(**kw)
    second = c.complete(**kw)

    assert not first.cached
    assert second.cached
    assert second.text == first.text
    assert len(backend.calls) == 1, "the cached call must not reach the backend"


def test_cached_call_costs_nothing(tmp_path):
    c = LLMClient(cache=RequestCache(tmp_path / "c.sqlite"),
                  cost_log=CostLog(tmp_path / "log.jsonl"),
                  backend_override=MockBackend())
    kw = dict(messages=[{"role": "user", "content": "x"}], tier=Tier.BULK,
              step="t", model="m", provider="mock")
    c.complete(**kw)
    second = c.complete(**kw)
    assert second.cost_usd == 0.0
    assert second.latency_s == 0.0


def test_use_cache_false_bypasses_the_cache(tmp_path):
    backend = MockBackend()
    c = LLMClient(cache=RequestCache(tmp_path / "c.sqlite"),
                  cost_log=CostLog(enabled=False), backend_override=backend)
    kw = dict(messages=[{"role": "user", "content": "x"}], tier=Tier.BULK,
              step="t", model="m", provider="mock", use_cache=False)
    c.complete(**kw)
    c.complete(**kw)
    assert len(backend.calls) == 2


def test_system_prompt_is_passed_through():
    backend = MockBackend()
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(enabled=False),
                  backend_override=backend)
    c.complete(messages=[{"role": "user", "content": "x"}], system="be terse",
               tier=Tier.BULK, step="t", model="m", provider="mock")
    assert backend.calls[0]["system"] == "be terse"


def test_result_json_parses():
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(enabled=False),
                  backend_override=MockBackend())
    r = c.complete(messages=[{"role": "user", "content": "x"}], tier=Tier.BULK, step="t",
                   model="m", provider="mock", response_format={"type": "json_object"})
    assert isinstance(r.json(), dict)


def test_result_json_error_is_actionable():
    """A model returning prose where JSON was asked for is a prompt problem;
    the error should say so rather than raise a bare JSONDecodeError."""
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(enabled=False),
                  backend_override=MockBackend())
    r = c.complete(messages=[{"role": "user", "content": "x"}], tier=Tier.BULK,
                   step="t", model="m", provider="mock")
    with pytest.raises(ValueError, match="expected JSON"):
        r.json()


def test_unresolved_tier_raises_with_a_fix(tmp_path, monkeypatch):
    """A tier with no model must fail with an actionable message.

    Pointed at a throwaway config rather than the repo's, so this stays true
    whatever is currently assigned in config/models.yaml -- otherwise the test
    passes or fails depending on how far tier configuration has progressed,
    which tests nothing.
    """
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "default_provider: groq\n"
        "providers:\n"
        "  groq:\n"
        "    base_url_env: GROQ_BASE_URL\n"
        "    api_key_env: GROQ_API_KEY\n"
        "tiers:\n"
        "  BULK:\n"
        "    provider: groq\n"
        "    model: null\n"
        "fallback: {enabled: false}\n"
        "pricing: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "MODELS_YAML", cfg)
    settings.reload()
    try:
        with pytest.raises(TierNotConfigured, match="discover_models"):
            mock_client().complete(
                messages=[{"role": "user", "content": "x"}], tier=Tier.BULK, step="t"
            )
    finally:
        settings.reload()


def test_configured_tiers_resolve_to_a_model():
    """Whatever is in config/models.yaml must at least be internally valid.

    SECOND_BACKBONE is allowed to be unset: it has to be a DIFFERENT family
    from JUDGE, and picking one requires seeing the provider's real catalogue.
    """
    settings.reload()
    cfg = settings.models()
    for name in ("BULK", "JUDGE", "LONG_CONTEXT"):
        assert cfg.tier(name).resolved, (
            f"tier {name} has no model. Run `python scripts/discover_models.py`."
        )


def test_second_backbone_is_a_different_family_from_judge():
    """A same-family robustness run demonstrates nothing about
    model-independence, which is the whole point of RQ3."""
    settings.reload()
    cfg = settings.models()
    judge = cfg.tier("JUDGE").model
    second = cfg.tier("SECOND_BACKBONE").model
    if not second:
        pytest.skip("SECOND_BACKBONE not assigned yet -- pick one after model discovery")

    def family(model_id: str) -> str:
        low = model_id.lower()
        for fam in ("gpt-oss", "llama", "qwen", "deepseek", "mixtral", "mistral",
                    "gemma", "kimi", "compound"):
            if fam in low:
                return fam
        return low.split("/")[0]

    assert family(judge) != family(second), (
        f"SECOND_BACKBONE ({second}) is the same family as JUDGE ({judge}); "
        "the robustness ablation would show nothing"
    )


def test_retries_then_gives_up(tmp_path):
    backend = MockBackend(fail_with=BackendError("boom"))
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                  backend_override=backend, max_retries=3, base_backoff=0.0)
    with pytest.raises(BackendError, match="all 3 attempts"):
        c.complete(messages=[{"role": "user", "content": "x"}], tier=Tier.BULK,
                   step="t", model="m", provider="mock")
    assert len(backend.calls) == 3


def test_failure_is_logged(tmp_path):
    log_path = tmp_path / "l.jsonl"
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(log_path),
                  backend_override=MockBackend(fail_with=BackendError("boom")),
                  max_retries=2, base_backoff=0.0)
    with pytest.raises(BackendError):
        c.complete(messages=[{"role": "user", "content": "x"}], tier=Tier.BULK,
                   step="t", model="m", provider="mock")
    records = list(CostLog(log_path).read())
    assert records and records[-1].error


def test_rate_limit_is_retried(tmp_path):
    backend = MockBackend(fail_with=RateLimitError("429"))
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(enabled=False),
                  backend_override=backend, max_retries=3, base_backoff=0.0)
    with pytest.raises(BackendError):
        c.complete(messages=[{"role": "user", "content": "x"}], tier=Tier.BULK,
                   step="t", model="m", provider="mock")
    assert len(backend.calls) == 3


# --------------------------------------------------------------------------- #
# Cost log
# --------------------------------------------------------------------------- #


def test_costlog_round_trips(tmp_path):
    log = CostLog(tmp_path / "l.jsonl")
    log.log(step="a2", tier="BULK", provider="groq", model="m",
            prompt_tokens=100, completion_tokens=20, cost_usd=0.001)
    records = list(log.read())
    assert len(records) == 1
    assert records[0].step == "a2"
    assert records[0].total_tokens == 120


def test_costlog_tolerates_a_truncated_line(tmp_path):
    """A killed run can leave a half-written line. That must not make the
    whole log unreadable."""
    path = tmp_path / "l.jsonl"
    log = CostLog(path)
    log.log(step="a", tier="BULK", provider="p", model="m")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"step": "trunc"\n')
    log.log(step="b", tier="BULK", provider="p", model="m")
    assert [r.step for r in CostLog(path).read()] == ["a", "b"]


def _recs():
    return [
        CallRecord(timestamp=0, step="a2", tier="BULK", provider="groq", model="m",
                   prompt_tokens=100, completion_tokens=20, cached=False, cost_usd=0.01),
        CallRecord(timestamp=1, step="a2", tier="BULK", provider="groq", model="m",
                   prompt_tokens=100, completion_tokens=20, cached=True),
        CallRecord(timestamp=2, step="judge", tier="JUDGE", provider="ollama", model="q",
                   prompt_tokens=500, completion_tokens=50, cached=False, fell_back=True),
    ]


def test_summary_counts_and_hit_rate():
    s = summarise(_recs())
    assert s["total"]["calls"] == 3
    assert s["total"]["cached"] == 1
    assert s["total"]["billable_calls"] == 2
    assert abs(s["total"]["cache_hit_rate"] - 1 / 3) < 1e-9
    assert s["by_step"]["a2"]["calls"] == 2
    assert s["by_tier"]["JUDGE"]["tokens"] == 550


def test_summary_of_nothing_is_safe():
    s = summarise([])
    assert s["total"]["calls"] == 0
    assert s["total"]["cache_hit_rate"] == 0.0


def test_report_shouts_about_a_fallback():
    """A silent model substitution changes what produced a number. The report
    must make it impossible to miss."""
    out = render(summarise(_recs()))
    assert "FELL BACK" in out
    assert "reported, not averaged in silently" in out


def test_report_is_quiet_when_nothing_fell_back():
    clean = [r for r in _recs() if not r.fell_back]
    assert "FELL BACK" not in render(summarise(clean))


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_hardware_profile_loads():
    hw = settings.hardware("conservative_6gb")
    assert hw.name == "conservative_6gb"
    assert hw.top_k == 5
    assert hw.cross_encoder.batch_size > 0


def test_conservative_profile_keeps_the_effective_batch_at_32():
    """Both profiles must train at the same effective batch size, or the 8 GB
    machine and the 6 GB machine produce different models from one config."""
    for name in ("conservative_6gb", "extended_8gb"):
        assert settings.hardware(name).finetune.effective_batch_size == 32


def test_extended_profile_only_raises_throughput():
    """The 8 GB profile is an opt-in speedup, not a different experiment: it
    must not change the checkpoint being used."""
    small = settings.hardware("conservative_6gb")
    big = settings.hardware("extended_8gb")
    assert small.finetune.model == big.finetune.model
    assert small.cross_encoder.model == big.cross_encoder.model
    assert big.cross_encoder.batch_size >= small.cross_encoder.batch_size


def test_unknown_profile_lists_the_valid_ones():
    with pytest.raises(KeyError, match="available"):
        settings.hardware("does_not_exist")


def test_cpu_profile_exists_for_ci():
    assert settings.hardware("cpu_only").device == "cpu"


def test_models_config_defines_all_four_tiers():
    cfg = settings.models()
    assert set(cfg.tiers) == {t.value for t in Tier}


def test_unknown_tier_is_an_error():
    with pytest.raises(KeyError, match="unknown tier"):
        settings.models().tier("NOPE")

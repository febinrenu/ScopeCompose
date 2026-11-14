"""A spent provider hands the question to the next provider of the same weights, and the answer says which."""

from types import SimpleNamespace

import pytest

from api_budget import RateLimitError
from api_budget.client import Tier
from study import cqa, reader, run


def _gen():
    parsed = SimpleNamespace(answers=[("yes", ())], evidence=(), ok=True, bad_citations=0)
    return SimpleNamespace(parsed=parsed, text="{}", prompt_tokens=10, completion_tokens=2, latency_s=0.1)


@pytest.fixture
def cell(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RUNS", tmp_path)
    monkeypatch.setattr(run, "_route", {})
    return [q for q in cqa.hosted_questions() if q.kind == "conditional"][:2]


def test_spent_primary_switches_and_records_provider(cell, monkeypatch):
    calls = []

    def fake(client, tier, q, evidence, step):
        calls.append(tier)
        if tier is Tier.STUDY_OSS20B:
            raise RateLimitError("tokens per day (TPD)")
        return _gen()

    monkeypatch.setattr(reader, "generate", fake)
    s = run.run_cell("gold", "oss20b", "hosted", questions=cell, client=object())
    assert s["ran"] == 2
    assert calls == [Tier.STUDY_OSS20B, Tier.STUDY_OSS20B_NV, Tier.STUDY_OSS20B_NV]     # sticky after the switch
    recs = run.read_cell(run.cell_path("gold", "oss20b", "hosted", run.systems.Env()))
    assert {r["served_by"] for r in recs.values()} == {"nvidia:openai/gpt-oss-20b"}
    assert {r["model"] for r in recs.values()} == {"openai/gpt-oss-20b"}


def test_parallel_workers_write_every_answer_once(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RUNS", tmp_path)
    monkeypatch.setattr(run, "_route", {})
    qs = [q for q in cqa.hosted_questions() if q.kind == "conditional"][:12]
    spent = {"n": 0}

    def fake(client, tier, q, evidence, step):
        if tier is Tier.STUDY_OSS20B and spent["n"] < 3:        # the primary fails for the first few
            spent["n"] += 1
            raise RateLimitError("tokens per day (TPD)")
        return _gen()

    monkeypatch.setattr(reader, "generate", fake)
    s = run.run_cell("gold", "oss20b", "hosted", questions=qs, client=object(), workers=4)
    lines = run.cell_path("gold", "oss20b", "hosted", run.systems.Env()).read_text(encoding="utf-8").splitlines()
    assert s["ran"] == 12 and len(lines) == 12 and len(run.read_cell(run.cell_path("gold", "oss20b", "hosted", run.systems.Env()))) == 12
    assert run._route["oss20b"] == 1                            # moved on exactly one provider, not past NVIDIA


def test_every_provider_spent_raises(cell, monkeypatch):
    def fake(client, tier, q, evidence, step):
        raise RateLimitError("spent")

    monkeypatch.setattr(reader, "generate", fake)
    with pytest.raises(RateLimitError):
        run.run_cell("gold", "oss120b", "hosted", questions=cell, client=object())


def test_single_route_model_unchanged(cell, monkeypatch):
    monkeypatch.setattr(reader, "generate", lambda client, tier, q, evidence, step: _gen())
    run.run_cell("gold", "qwen27b", "hosted", questions=cell[:1], client=object())
    recs = run.read_cell(run.cell_path("gold", "qwen27b", "hosted", run.systems.Env()))
    assert [r["served_by"] for r in recs.values()] == ["groq:qwen/qwen3.8-27b"]

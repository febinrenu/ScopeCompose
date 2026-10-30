"""The scheduler: quota-aware, resumable, visible, bounded."""

from __future__ import annotations

import json
import time

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient
from api_budget.costlog import CallRecord, CostLog
from contract.mock import generate
from experiments.run_matrix import Cell, read_done
from experiments.scheduler import (
    UNLIMITED,
    Job,
    budget_for,
    expected_items,
    prevent_sleep,
    run_schedule,
    tokens_last_24h,
)

PRIMARY = "openai/gpt-oss-120b"


def insts(n=3):
    return [i for i in generate(120, seed=0) if i.personas][:n]


def cell(**kw):
    base = dict(system="B0", backbone="local", population="S5", setting="underspecified", split="train")
    base.update(kw)
    return Cell(**base)


def client(tmp_path, backend=None):
    return LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                     backend_override=backend or MockBackend(), max_retries=1, base_backoff=0.0)


def log_with(tmp_path, records):
    log = CostLog(tmp_path / "quota.jsonl")
    for r in records:
        log.log(**r)
    return log


def rec(model, tokens, age_s=0, cached=False, error=None, now=1_000_000.0):
    """A cost-log record. ``log()`` stamps the current time, so tests pass an explicit ``now``."""
    return dict(step="t", tier="BACKBONE_PRIMARY", provider="groq", model=model, prompt_tokens=tokens,
                completion_tokens=0, cached=cached, error=error)


def write_records(path, records, now):
    with path.open("w", encoding="utf-8") as fh:
        for model, tokens, age, cached, error in records:
            fh.write(json.dumps(asdict_record(model, tokens, now - age, cached, error)) + "\n")


def asdict_record(model, tokens, ts, cached, error):
    return {"timestamp": ts, "step": "t", "tier": "BACKBONE_PRIMARY", "provider": "groq", "model": model,
            "prompt_tokens": tokens, "completion_tokens": 0, "cached": cached, "fell_back": False,
            "cost_usd": 0.0, "latency_s": 0.0, "error": error, "meta": {}}


# --------------------------------------------------------------------------- #
# Quota
# --------------------------------------------------------------------------- #


def test_only_uncached_successful_recent_calls_to_that_model_count(tmp_path):
    now = 1_000_000.0
    p = tmp_path / "q.jsonl"
    write_records(p, [(PRIMARY, 1000, 60, False, None),            # counts
                      (PRIMARY, 500, 90_000, False, None),         # older than 24 h
                      (PRIMARY, 700, 10, True, None),              # cache hit: free
                      (PRIMARY, 300, 10, False, "429"),            # failed: not spent
                      ("qwen/qwen3.8-27b", 900, 10, False, None)], now)  # a different model
    assert tokens_last_24h(PRIMARY, CostLog(p), now=now) == 1000


def test_the_budget_keeps_a_safety_margin_and_is_never_negative(tmp_path):
    now = 1_000_000.0
    p = tmp_path / "q.jsonl"
    write_records(p, [(PRIMARY, 50_000, 10, False, None)], now)
    assert budget_for(PRIMARY, CostLog(p), daily_cap=200_000, margin=0.10, now=now) == 130_000
    write_records(p, [(PRIMARY, 500_000, 10, False, None)], now)
    assert budget_for(PRIMARY, CostLog(p), daily_cap=200_000, now=now) == 0


def test_a_local_model_is_never_throttled(tmp_path):
    now = 1_000_000.0
    p = tmp_path / "q.jsonl"
    write_records(p, [("scopecompose-qwen7b", 10**9, 10, False, None)], now)
    assert budget_for("scopecompose-qwen7b", CostLog(p), now=now, provider="ollama") == UNLIMITED


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_expected_items_counts_per_setting():
    ii = insts(3)
    assert expected_items(cell(), ii) == 3
    assert expected_items(cell(setting="persona"), ii) == sum(len(i.personas) for i in ii)


def test_a_schedule_runs_every_cell_to_completion_and_reports_it(tmp_path):
    out = tmp_path / "runs"
    jobs = run_schedule([cell(), cell(system="B3")], insts(3), out, client=client(tmp_path),
                        log=CostLog(tmp_path / "none.jsonl"), sleep=lambda s: None,
                        heartbeat=out / "hb.json", status=out / "status.json")
    assert [j.state for j in jobs] == ["done", "done"] and all(j.complete for j in jobs)
    assert (out / "hb.json").exists()
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert [j["complete"] for j in status["jobs"]] == [True, True]


def test_jobs_run_in_the_order_given(tmp_path):
    out = tmp_path / "runs"
    order = []

    class Spy(MockBackend):
        def complete(self, **kw):
            order.append(kw["messages"][0]["content"][:0] or "call")
            return super().complete(**kw)

    run_schedule([cell(system="B3"), cell()], insts(2), out, client=client(tmp_path, Spy()),
                 log=CostLog(tmp_path / "none.jsonl"), sleep=lambda s: None)
    files = sorted(out.glob("*.jsonl"), key=lambda p: p.stat().st_mtime_ns)
    assert files[0].name.startswith("B3__") and files[1].name.startswith("B0__")


def test_a_rerun_of_a_finished_schedule_does_nothing(tmp_path):
    out = tmp_path / "runs"
    backend = MockBackend()
    c = client(tmp_path, backend)
    run_schedule([cell()], insts(3), out, client=c, log=CostLog(tmp_path / "none.jsonl"), sleep=lambda s: None)
    calls = len(backend.calls)
    jobs = run_schedule([cell()], insts(3), out, client=c, log=CostLog(tmp_path / "none.jsonl"), sleep=lambda s: None)
    assert jobs[0].state == "done" and len(backend.calls) == calls


def test_an_exhausted_quota_waits_instead_of_calling_and_gives_up_after_max_rounds(tmp_path):
    """A primary-backbone cell with the daily quota already spent must not touch the provider."""
    now = time.time()
    p = tmp_path / "q.jsonl"
    write_records(p, [(PRIMARY, 500_000, 10, False, None)], now)
    backend = MockBackend()
    slept = []
    jobs = run_schedule([cell(backbone="primary")], insts(2), tmp_path / "runs", client=client(tmp_path, backend),
                        log=CostLog(p), sleep=slept.append, wait_s=99, max_rounds=3)
    assert jobs[0].state == "waiting" and "quota" in jobs[0].note and not jobs[0].complete
    assert backend.calls == [] and slept == [99, 99, 99], "waited each round, never called"


def test_a_cell_that_hits_its_budget_resumes_next_round(tmp_path):
    out = tmp_path / "runs"
    c = client(tmp_path)
    # a tiny cap forces an interruption after the first item each round
    jobs = run_schedule([cell(backbone="primary")], insts(3), out, client=c,
                        log=CostLog(tmp_path / "none.jsonl"), daily_cap=50, sleep=lambda s: None, max_rounds=1)
    assert jobs[0].done == 1 and not jobs[0].complete and jobs[0].state == "waiting"
    assert "token budget" in jobs[0].note
    jobs = run_schedule([cell(backbone="primary")], insts(3), out, client=c,
                        log=CostLog(tmp_path / "none.jsonl"), daily_cap=10**9, sleep=lambda s: None)
    assert jobs[0].complete and jobs[0].done == 3


def test_a_cell_file_from_another_model_marks_the_job_failed_and_the_rest_continue(tmp_path):
    out = tmp_path / "runs"
    c = client(tmp_path)
    run_schedule([cell()], insts(2), out, client=c, log=CostLog(tmp_path / "none.jsonl"), sleep=lambda s: None)
    path = out / f"{cell().name}.jsonl"
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    recs[0]["model"] = "other-model"
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    jobs = run_schedule([cell(), cell(system="B3")], insts(2), out, client=c, log=CostLog(tmp_path / "none.jsonl"),
                        sleep=lambda s: None, max_rounds=2)
    assert jobs[0].state == "failed" and "never mixes models" in jobs[0].note
    assert jobs[1].state == "done"


def test_prevent_sleep_never_raises_and_restores():
    with prevent_sleep() as awake:
        assert isinstance(awake, bool)

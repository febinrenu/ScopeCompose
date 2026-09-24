"""Tests for the cost and latency instrumentation.

The failure mode this guards against is not a crash -- it is a number that
looks reportable and is not. Three ways that happens, one test each: spend from
earlier runs attributed to this one, a cached run reported as if it cost
something, and a bimodal latency distribution summarised by its mean.
"""

from __future__ import annotations

import pytest

from api_budget.costlog import CallRecord, CostLog
from experiments.run_cost_latency import CostLatencyReport, StageTiming, _percentile


def _record(*, cached=False, prompt=100, completion=50, cost=0.001, latency=0.4):
    return CallRecord(
        timestamp=0.0, step="s", tier="BULK", provider="mock", model="m",
        prompt_tokens=prompt, completion_tokens=completion,
        cached=cached, fell_back=False, cost_usd=cost, latency_s=latency,
    )


def test_only_this_run_is_counted(tmp_path):
    """The cost log is append-only and shared across every experiment.

    Summing the whole file would bill this run for every call any earlier run
    made -- which inflates cost-per-query by however many experiments happen to
    have run first. The measurement takes a length before and slices after.
    """
    log = CostLog(tmp_path / "cost.jsonl")
    for _ in range(7):
        log.record(_record())

    before = len(list(log.read()))
    log.record(_record(cost=0.005))
    new = list(log.read())[before:]

    assert len(new) == 1
    assert sum(r.cost_usd for r in new) == pytest.approx(0.005)


def test_a_fully_cached_run_says_so():
    """Zero cost on a warm cache is true and useless for planning.

    Reporting it without the caveat implies the system is free to run, when
    what actually happened is that somebody ran it before.
    """
    r = CostLatencyReport(n_queries=4)
    r.timing("A2+A3 detection").durations.extend([0.1] * 4)
    r.api_calls, r.api_cached = 12, 12

    assert r.billable_calls == 0
    assert "EVERY API CALL WAS CACHED" in r.render()


def test_a_partly_cached_run_does_not_carry_the_caveat():
    r = CostLatencyReport(n_queries=4)
    r.timing("A2+A3 detection").durations.extend([0.1] * 4)
    r.api_calls, r.api_cached, r.cost_usd = 12, 8, 0.02

    assert r.billable_calls == 4
    assert "EVERY API CALL WAS CACHED" not in r.render()


def test_sub_millisecond_noise_is_not_reported_as_bimodal():
    """The ratio alone is not enough.

    Offline, both modes are sub-millisecond, and the p95/median ratio between
    two samples of timer noise runs past 50x. The first version of this report
    duly announced a bimodal distribution on a run where every stage took
    effectively no time.
    """
    r = CostLatencyReport(n_queries=20)
    r.timing("A4 scope analysis").durations.extend([0.00001] * 18 + [0.0006, 0.0008])

    assert "bimodal" not in r.render()


def test_bimodal_latency_is_flagged_rather_than_averaged():
    """Most queries resolve in stage 1; a few escalate to an API call.

    The mean of those two modes describes no query that actually ran, so the
    report has to say the distribution is bimodal rather than quote one number.
    """
    r = CostLatencyReport(n_queries=20)
    t = r.timing("A2+A3 detection")
    t.durations.extend([0.05] * 18 + [4.0, 4.5])

    assert t.p95 > 3 * t.median
    out = r.render()
    assert "bimodal" in out
    assert "median and p95" in out


def test_stage_totals_and_percentiles():
    t = StageTiming("x", [0.1, 0.2, 0.3, 0.4])
    assert t.total == pytest.approx(1.0)
    assert t.mean == pytest.approx(0.25)
    assert t.median == pytest.approx(0.2) or t.median == pytest.approx(0.3)
    assert t.p95 == pytest.approx(0.4)


def test_percentile_of_nothing_is_zero_not_an_exception():
    assert _percentile([], 0.5) == 0.0


def test_empty_report_does_not_divide_by_zero():
    r = CostLatencyReport()
    assert r.per_query == 0.0
    assert r.calls_per_query == 0.0
    assert r.tokens_per_query == 0.0
    r.render()


def test_measure_times_every_stage_and_spends_nothing_offline(tmp_path):
    from contract.mock import generate
    from experiments.run_cost_latency import measure

    records = [g.to_query_record() for g in generate(4, seed=0)]
    report = measure(records, live=False)

    assert report.n_queries == 4
    assert set(report.stages) == {"A2+A3 detection", "A4 scope analysis"}
    assert all(len(t.durations) == 4 for t in report.stages.values())
    assert report.cost_usd == 0.0, "an offline run must not spend"


def test_serialisation_round_trips_the_numbers():
    r = CostLatencyReport(n_queries=2, api_calls=3, api_cached=1,
                          prompt_tokens=200, completion_tokens=100, cost_usd=0.01)
    r.timing("A4 scope analysis").durations.extend([0.5, 1.5])

    d = r.as_dict()
    assert d["n_queries"] == 2
    assert d["api"]["billable"] == 2
    assert d["stages"]["A4 scope analysis"]["total_s"] == pytest.approx(2.0)

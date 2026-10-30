"""Gate G1: measurement, projection against the quota, and the pre-agreed cut order."""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient, Tier
from api_budget.costlog import CostLog
from contract.mock import generate
from experiments.g1 import (
    CUT_ORDER,
    DAILY_CAP,
    ESTIMATE,
    PROTECTED,
    SystemMeasure,
    fit_to_quota,
    measure,
    planned_matrix,
    project,
    render,
    update_gate,
)


def sm(system, total, calls=1.0):
    est = ESTIMATE[system]
    return SystemMeasure(system, 5, total * 0.8, total * 0.2, total, total * 1.2, calls, est, total <= est * 1.25)


def measures(**totals):
    return {k: sm(k, v) for k, v in totals.items()}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def test_measure_reports_real_metered_tokens_per_system(tmp_path):
    insts = [i for i in generate(60, seed=0) if i.personas][:3]
    backend = MockBackend()
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"), backend_override=backend)
    out = measure(["B0", "B3"], insts, tier=Tier.BACKBONE_LOCAL, client=c)
    assert set(out) == {"B0", "B3"} and all(m.n == 3 for m in out.values())
    assert out["B0"].mean_calls == 1.0 and out["B0"].mean_prompt > 0 and out["B0"].mean_total > 0
    assert out["B0"].mean_total == pytest.approx(out["B0"].mean_prompt + out["B0"].mean_completion)
    assert out["B0"].p90_total >= out["B0"].mean_total * 0.5
    assert len(backend.calls) == 6


def test_a_system_over_its_estimate_by_more_than_the_tolerance_is_flagged():
    assert sm("B7c", 4000 * 1.25).within_tolerance and not sm("B7c", 4000 * 1.26).within_tolerance


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def test_the_planned_matrix_runs_b0_b3_on_controls_too_and_b7c_o3_on_s1_only():
    p = planned_matrix(50, 20, persona_n=100)
    assert p == {"B0": 170, "B3": 170, "B7c": 50, "O3": 50, "B8": 50}
    assert "B8" not in planned_matrix(50, 20, include_b8=False)


def test_projection_uses_the_measured_means_and_the_daily_cap():
    ms = measures(B0=1000, B3=1000, B7c=4000, O3=8000, B8=8000)
    proj = project(ms, planned_matrix(50, 0, persona_n=0, include_b8=False))
    assert proj.total == 50 * (1000 + 1000 + 4000 + 8000)
    assert proj.days == pytest.approx(proj.total / DAILY_CAP) and proj.fits == (proj.days <= 3.5)


def test_projection_refuses_an_unmeasured_system():
    with pytest.raises(KeyError, match="no measurement"):
        project(measures(B0=1000), {"B0": 10, "O3": 10})


# --------------------------------------------------------------------------- #
# The cut order
# --------------------------------------------------------------------------- #


def test_a_matrix_that_fits_is_not_cut():
    ms = measures(B0=500, B3=500, B7c=1000, O3=1500, B8=1500)
    p = fit_to_quota(ms, n_s1=20, n_controls=10, persona_n=30)
    assert p.fits and p.cuts == []


def test_cuts_are_applied_in_the_pre_agreed_order_and_stop_as_soon_as_it_fits():
    ms = measures(B0=1500, B3=1500, B7c=4000, O3=7500, B8=7500)
    p = fit_to_quota(ms, n_s1=40, n_controls=30, persona_n=120)
    assert p.cuts == [c for c in CUT_ORDER if c in p.cuts], "never out of order"
    assert p.cuts[0] == CUT_ORDER[0], "the persona setting goes first"
    one_cut = fit_to_quota(ms, n_s1=40, n_controls=30, persona_n=120, available_days=p.days + 0.01)
    assert len(one_cut.cuts) <= len(p.cuts)


def test_each_cut_actually_removes_work():
    ms = measures(B0=1500, B3=1500, B7c=4000, O3=7500, B8=7500)
    full = project(ms, planned_matrix(40, 30, persona_n=120)).total
    p = fit_to_quota(ms, n_s1=40, n_controls=30, persona_n=120, available_days=0.2)
    assert p.total < full and CUT_ORDER[0] in p.cuts and CUT_ORDER[1] in p.cuts and CUT_ORDER[2] in p.cuts


def test_the_confirmatory_cells_are_never_cut_even_when_nothing_else_is_left():
    ms = measures(B0=20000, B3=20000, B7c=40000, O3=60000, B8=60000)
    p = fit_to_quota(ms, n_s1=50, n_controls=20, persona_n=100)
    assert not p.fits and "NOTHING LEFT TO CUT" in p.cuts[-1]
    assert set(p.plan) == {"B0", "B3", "B7c", "O3"} and p.plan["B7c"] == p.plan["O3"] == 50
    assert PROTECTED == ("B0, B3 on S1", "B7c, O3 on S1")


# --------------------------------------------------------------------------- #
# Reporting and the gate file
# --------------------------------------------------------------------------- #


def test_render_shows_the_table_the_verdict_and_the_cuts():
    ms = measures(B0=1500, B3=1500, B7c=4000, O3=7500, B8=7500)
    text = render(ms, fit_to_quota(ms, n_s1=40, n_controls=30, persona_n=120, available_days=1.0))
    assert "G1: measured tokens per instance" in text and "Cuts applied" in text and "Protected, never cut" in text
    assert "DOES NOT FIT" in text or "FITS" in text


def test_the_gate_file_is_only_changed_when_asked_and_records_the_measurement(tmp_path):
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({"G1": {"status": "open", "what": "x"}, "G2": {"status": "open"}}), encoding="utf-8")
    ms = measures(B0=500, B3=500, B7c=1000, O3=1500)
    proj = fit_to_quota(ms, n_s1=10, n_controls=5, persona_n=0)
    g1 = update_gate(gates, proj, ms)
    saved = json.loads(gates.read_text(encoding="utf-8"))
    assert g1["status"] == "pass" and saved["G1"]["what"] == "x" and saved["G2"]["status"] == "open"
    assert saved["G1"]["measured"]["O3"]["mean_total"] == 1500
    bad = fit_to_quota(measures(B0=50000, B3=50000, B7c=50000, O3=50000), n_s1=50, n_controls=0, persona_n=0)
    assert update_gate(gates, bad, measures(B0=50000, B3=50000, B7c=50000, O3=50000))["status"] == "fail"

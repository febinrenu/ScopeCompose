"""Anything that can reach a test set must be able to run on the evaluation-only tier."""

from __future__ import annotations

import json

import benchmark.mining.build_batch as bb
import benchmark.mining.split_constructor as sc
from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient
from api_budget.costlog import CostLog


def _client(tmp_path, reply):
    backend = MockBackend(scripted={"": json.dumps(reply)})
    return LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                     backend_override=backend), backend


def _models(tmp_path):
    return [r.model for r in CostLog(tmp_path / "l.jsonl").read()]


def test_query_writing_defaults_to_the_pilot_tier_and_can_run_on_eval(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, {"query": "Can I use the airport lounge at the terminal?"})
    monkeypatch.setattr(bb, "QUERY_TIER", "BULK")
    bb.write_query("A rule about lounges.", "An exception for members.", client=c)
    monkeypatch.setattr(bb, "QUERY_TIER", "EVAL")
    bb.write_query("A rule about lounges.", "An exception for members.", client=c)
    log = list(CostLog(tmp_path / "l.jsonl").read())
    assert [r.tier for r in log] == ["BULK", "EVAL"]
    assert [r.step for r in log] == ["build_batch_query", "build_batch_query_eval"]


def test_split_construction_defaults_to_the_pilot_tier_and_can_run_on_eval(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, {"statement_a": "x", "statement_b": "y"})
    cand = sc.SplitCandidate(doc_id="d", provider="p", kind="k", url="u",
                             source="Fees are charged, unless the holder is a student.", cue="unless")
    monkeypatch.setattr(sc, "SPLIT_TIER", "BULK")
    sc.split_sentence(cand, client=c)
    monkeypatch.setattr(sc, "SPLIT_TIER", "EVAL")
    sc.split_sentence(cand, client=c)
    assert [r.tier for r in CostLog(tmp_path / "l.jsonl").read()] == ["BULK", "EVAL"]


def test_both_clis_expose_the_tier_flag():
    import subprocess, sys

    for mod in ("benchmark.mining.build_batch", "benchmark.mining.split_constructor",
                "benchmark.mining.llm_proposer"):
        out = subprocess.run([sys.executable, "-m", mod, "--help"], capture_output=True, text=True)
        assert "--tier" in out.stdout, mod

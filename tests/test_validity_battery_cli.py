"""The validity-battery driver: split guard, refusal of unscorable instances, and the JSON it writes."""

from __future__ import annotations

import json
import re

import pytest

from contract.mock import generate
from detection.nli import NLIScores
from experiments.validity_battery import run, scorable, select, to_json
from metrics.case_accuracy import NOT_DETERMINED, ReaderOutput
from metrics.validity import FILLER, NULL_ANSWER, OUTPUTS, V1_CRITERIA_SHA


class ConstNLI:
    def score_batch(self, pairs):
        return [NLIScores(0.95, 0.05, 0.0) for _ in pairs]


def reader(*, query, answer, persona):
    if answer == NULL_ANSWER:
        return ReaderOutput(NOT_DETERMINED, "")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    pool = [s for s in sentences if s != FILLER.split(". ")[0] + "."]
    if "-tier" in persona:
        cands = [s for s in pool if re.search(r"waived|fee-free|may be eligible|no fee", s, re.I)]
    else:
        cands = [s for s in pool if re.search(r"\d+%|incur|fee", s, re.I) and "waived" not in s]
    s = (cands or pool)[0]
    return ReaderOutput(s, s)


def dev_instances(n=12):
    out = [i.model_copy(update={"split": "dev"}) for i in generate(120, seed=0)
           if i.personas and i.gold_scoped_answer and i.selection_answer][:n]
    assert len(out) == n
    return out


def test_the_test_split_is_refused_outright():
    with pytest.raises(ValueError, match="never on the test split"):
        select(dev_instances(), split="test")
    with pytest.raises(ValueError, match="never on the test split"):
        run(dev_instances(), reader, ConstNLI(), split="test")


def test_only_the_requested_split_is_scored():
    mixed = dev_instances(6)
    mixed[0] = mixed[0].model_copy(update={"split": "train"})
    assert len(select(mixed, split="dev")) == 5 and len(select(mixed, split="train")) == 1
    assert run(mixed, reader, ConstNLI(), split="dev").n_instances == 5


def test_an_empty_split_is_an_error():
    with pytest.raises(ValueError, match="no instances in split"):
        run(dev_instances(3), reader, ConstNLI(), split="train")


def test_unscorable_instances_are_named_not_skipped():
    insts = dev_instances(4)
    insts[1] = insts[1].model_copy(update={"selection_answer": None})
    ok, bad = scorable(insts)
    assert bad == [insts[1].instance_id] and len(ok) == 3
    with pytest.raises(ValueError, match=insts[1].instance_id):
        run(insts, reader, ConstNLI())


def test_the_json_carries_every_output_every_criterion_and_the_hash():
    res = run(dev_instances(), reader, ConstNLI())
    out = to_json(res)
    json.dumps(out)                                           # serialisable
    assert out["criteria_sha256"] == V1_CRITERIA_SHA and out["n_instances"] == 12
    assert list(out["outputs"]) == list(OUTPUTS)
    assert set(out["outputs"]["gold"]) == {"legacy_pr", "cca_strict", "cca_lenient", "scoped_precision"}
    assert [c["name"] for c in out["criteria"]] == [c.name for c in res.criteria()]
    assert out["nli_threshold_tuned"] is False and out["v1_passed"] is res.v1_passed
    assert out["outputs"]["gold"]["cca_strict"] == 1.0 and out["outputs"]["null"]["cca_strict"] == 0.0

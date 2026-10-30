"""The matrix runner: resumable, one model per cell, same inputs for every system."""

from __future__ import annotations

import json

import pytest

from api_budget import BackendError
from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient
from api_budget.costlog import CostLog
from contract.gold import GoldInstance
from contract.mock import generate
from experiments.run_matrix import (
    Cell,
    CellInterrupted,
    CellMismatch,
    items,
    load_answers,
    load_instances,
    main,
    read_done,
    run_cell,
    select,
)
from experiments.staging import FreezeError


def instances(n=3) -> list[GoldInstance]:
    """Mock refinement instances with personas. They are LLM-authored, hence stratum S5, split 'train'."""
    out = [i for i in generate(120, seed=0) if i.personas][:n]
    assert len(out) == n
    return out


def cell(**kw) -> Cell:
    base = dict(system="B0", backbone="local", population="S5", setting="underspecified", split="train")
    base.update(kw)
    return Cell(**base)


def client(tmp_path, backend=None):
    b = backend or MockBackend()
    return LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                     backend_override=b, max_retries=1, base_backoff=0.0), b


# --------------------------------------------------------------------------- #
# Cells and selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kw", [{"system": "B9"}, {"backbone": "huge"}, {"population": "S9"}, {"setting": "x"}])
def test_an_invalid_cell_is_rejected(kw):
    with pytest.raises(ValueError):
        cell(**kw)


def test_a_cell_names_itself_uniquely_and_resolves_its_pinned_model():
    c = cell()
    assert c.name == "B0__local__S5__underspecified__train"
    assert c.model == "scopecompose-qwen7b"
    assert cell(backbone="primary").model == "openai/gpt-oss-120b"


def test_select_filters_by_split_and_population_and_sorts():
    insts = instances(3)
    mixed = [insts[2].model_copy(update={"split": "dev"})] + insts[:2]
    got = select(mixed, cell())
    assert [i.instance_id for i in got] == sorted(i.instance_id for i in insts[:2])
    assert select(mixed, cell(population="S1")) == [], "mock probes are S5, never S1"


def test_items_per_setting():
    inst = instances(1)[0]
    (one,) = items(inst, cell())
    assert one == (inst.instance_id, None, inst.query)
    per = items(inst, cell(setting="persona"))
    assert [k for k, _, _ in per] == [f"{inst.instance_id}|{p.persona_id}" for p in inst.personas]
    assert all(q.startswith(inst.query) and p.statement in q for (_, _, q), p in zip(per, inst.personas))


# --------------------------------------------------------------------------- #
# Running, resuming, refusing
# --------------------------------------------------------------------------- #


def test_a_run_writes_one_record_per_instance_with_the_model_and_metering(tmp_path):
    c, backend = client(tmp_path)
    insts = instances(3)
    s = run_cell(cell(), insts, tmp_path / "runs", client=c)
    assert s["ran"] == 3 and s["skipped"] == 0 and s["calls"] == 3 == len(backend.calls)
    recs = list(read_done(tmp_path / "runs" / f"{cell().name}.jsonl").values())
    assert len(recs) == 3
    for r in recs:
        assert r["model"] == "scopecompose-qwen7b" and r["system"] == "B0" and r["setting"] == "underspecified"
        assert r["prompt_tokens"] > 0 and r["text"]


def test_a_rerun_skips_everything_and_makes_no_calls(tmp_path):
    c, backend = client(tmp_path)
    insts = instances(3)
    run_cell(cell(), insts, tmp_path / "runs", client=c)
    before = len(backend.calls)
    s = run_cell(cell(), insts, tmp_path / "runs", client=c)
    assert s["ran"] == 0 and s["skipped"] == 3 and len(backend.calls) == before


def test_a_cell_resumes_from_a_partial_file(tmp_path):
    c, backend = client(tmp_path)
    insts = instances(3)
    run_cell(cell(), insts[:1], tmp_path / "runs", client=c)
    backend.calls.clear()
    s = run_cell(cell(), insts, tmp_path / "runs", client=c)
    assert s["ran"] == 2 and s["skipped"] == 1 and len(backend.calls) == 2


def test_a_cell_file_from_a_different_model_or_system_is_refused(tmp_path):
    c, _ = client(tmp_path)
    insts = instances(2)
    run_cell(cell(), insts, tmp_path / "runs", client=c)
    path = tmp_path / "runs" / f"{cell().name}.jsonl"
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    recs[0]["model"] = "some-other-model"
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    with pytest.raises(CellMismatch, match="never mixes models"):
        run_cell(cell(), insts, tmp_path / "runs", client=c)


def test_a_token_budget_interrupts_the_cell_and_keeps_what_was_done(tmp_path):
    c, _ = client(tmp_path)
    insts = instances(3)
    with pytest.raises(CellInterrupted, match="token budget"):
        run_cell(cell(), insts, tmp_path / "runs", client=c, token_budget=1)
    done = read_done(tmp_path / "runs" / f"{cell().name}.jsonl")
    assert len(done) == 1, "the first item ran before the budget was checked; nothing is lost"
    s = run_cell(cell(), insts, tmp_path / "runs", client=c)
    assert s["ran"] == 2 and len(read_done(tmp_path / "runs" / f"{cell().name}.jsonl")) == 3


def test_repeated_backend_errors_interrupt_without_writing_partial_answers(tmp_path):
    c, _ = client(tmp_path, MockBackend(fail_with=BackendError("down")))
    with pytest.raises(CellInterrupted, match="backend errors"):
        run_cell(cell(), instances(3), tmp_path / "runs", client=c, max_errors=2)
    assert read_done(tmp_path / "runs" / f"{cell().name}.jsonl") == {}


def test_the_persona_setting_runs_one_answer_per_persona_with_the_persona_in_the_query(tmp_path):
    c, backend = client(tmp_path)
    inst = instances(1)[0]
    s = run_cell(cell(setting="persona"), [inst], tmp_path / "runs", client=c)
    assert s["ran"] == len(inst.personas) == len(backend.calls)
    sent = [call["messages"][0]["content"] for call in backend.calls]
    assert all(p.statement in m for p, m in zip(inst.personas, sent))
    done = read_done(tmp_path / "runs" / f"{cell(setting='persona').name}.jsonl")
    assert {r["persona_id"] for r in done.values()} == {p.persona_id for p in inst.personas}


def test_b3_is_given_gold_conflict_groups(tmp_path):
    c, _ = client(tmp_path)
    insts = instances(2)
    run_cell(cell(system="B3"), insts, tmp_path / "runs", client=c)
    recs = read_done(tmp_path / "runs" / f"{cell(system='B3').name}.jsonl").values()
    assert all(r["trace"]["decisions"] for r in recs), "every conditional instance has a typed conflict group"


def test_load_answers_maps_items_to_text(tmp_path):
    c, _ = client(tmp_path)
    insts = instances(2)
    run_cell(cell(), insts, tmp_path / "runs", client=c)
    ans = load_answers(tmp_path / "runs" / f"{cell().name}.jsonl")
    assert set(ans) == {i.instance_id for i in insts} and all(ans.values())


def test_the_cli_refuses_a_test_split_run_before_the_freeze(tmp_path):
    data = tmp_path / "t.jsonl"
    data.write_text(json.dumps(instances(1)[0].model_dump(mode="json")) + "\n", encoding="utf-8")
    with pytest.raises(FreezeError):
        main(["--data", str(data), "--system", "B0", "--backbone", "local", "--population", "S1",
              "--split", "test", "--out", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists(), "nothing was run, nothing was written"


def test_load_instances_round_trips(tmp_path):
    insts = instances(2)
    p = tmp_path / "i.jsonl"
    p.write_text("\n".join(json.dumps(i.model_dump(mode="json")) for i in insts) + "\n", encoding="utf-8")
    assert [i.instance_id for i in load_instances(p)] == [i.instance_id for i in insts]

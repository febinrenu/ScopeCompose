"""The studio service: orchestration, tracing, glue, ingest, and the API.

All offline: heuristic NLI, rule descriptors, cue extraction, template
generation, and the mock LLM backend where a call is exercised.
"""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.costlog import CostLog
from contract.gold import Applicability, AttributeKind, Branch, ScopeAttribute
from contract.models import Passage
from service import branches as B
from service import ingest
from service.events import Tracer, use_tracer
from service.orchestrator import RunRequest, benchmark, engines, run_query

MISSED = "wp1_fin_imp_003"  # conditional refinement; the offline stage-1 scorer misses it

STAGES = ["setup", "ingest", "retrieve", "detect", "type", "policy", "scope", "extract",
          "branches", "compose", "generate", "verify", "arena", "reader", "score"]


def _run(**kw):
    tr = Tracer("test")
    summary = run_query(RunRequest(**kw), tr)
    return summary, tr.events


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def test_every_stage_opens_and_closes_in_order():
    _, events = _run(source="benchmark", record_id=MISSED)
    starts = [e.stage for e in events if e.kind == "start"]
    ends = [e.stage for e in events if e.kind == "end"]
    assert starts == STAGES and ends == STAGES
    assert [e.seq for e in events] == list(range(len(events)))
    assert all(a.t_ms <= b.t_ms for a, b in zip(events, events[1:]))
    assert not [e for e in events if e.kind == "error"]


def test_override_composes_and_says_so():
    summary, events = _run(source="benchmark", record_id=MISSED, detector_policy="override")
    warn = [e for e in events if e.stage == "policy" and e.kind == "warn"]
    assert warn and warn[0].payload["kind"] == "gold"
    assert summary["override"]["kind"] == "gold"
    assert summary["resolution"] == "composed"
    assert summary["faithful"] is True
    origin = [e for e in events if e.stage == "branches" and e.title == "branches"][0].payload["origin"]
    assert origin.startswith("oracle")


def test_as_run_answers_with_the_rule_alone():
    summary, events = _run(source="benchmark", record_id=MISSED, detector_policy="as_run")
    assert summary["override"] is None
    assert summary["resolution"] == "pass_through"
    assert any(e.kind == "warn" and e.stage == "policy" for e in events)


def test_branches_are_exactly_what_the_operator_composed():
    summary, _ = _run(source="benchmark", record_id=MISSED)
    inst = benchmark()[MISSED]
    rec = inst.to_query_record(include_gold_pairs=True)
    composed = engines(False).operator.compose(rec, list(inst.gold_branches))
    assert [b["id"] for b in summary["branches"]] == [b.branch_id for b in composed.branches]


def test_selection_is_judged_wrong_for_the_exception_reader():
    # the clause matcher must not read "no monthly fee applies" into "a monthly fee applies"
    _, events = _run(source="benchmark", record_id=MISSED,
                     reader={"scheme": "graduate", "years_since_registration": 1})
    v = {x["system"]: x["right"] for x in
         [e for e in events if e.stage == "reader" and e.kind == "data"][0].payload["verdicts"]}
    assert v["scopecompose"] is True and v["selection"] is False


def test_inline_passages_run_end_to_end():
    summary, events = _run(source="passages", query="Is there a fee for international payments?",
                           detector_policy="override", passages=[
                               {"text": "A 3% fee applies to all international card payments.",
                                "source_type": "official_policy"},
                               {"text": "Premium cardholders pay no fee on international payments, "
                                        "unless the payment is a cash withdrawal.",
                                "source_type": "product_terms"}])
    assert summary["answer"]
    assert any(e.stage == "retrieve" and e.title == "decision" and e.payload["mode"] == "given" for e in events)


def test_request_validation():
    with pytest.raises(ValueError):
        RunRequest(source="benchmark").validate()
    with pytest.raises(ValueError):
        RunRequest(source="passages", query="q", passages=[{"text": "only one"}]).validate()
    with pytest.raises(ValueError):
        RunRequest(source="benchmark", record_id="x", mode="turbo").validate()


# --------------------------------------------------------------------------- #
# Glue, readers, ingest
# --------------------------------------------------------------------------- #


def _exc(values):
    return Branch(branch_id="b1", condition="premium tier", outcome="no fee",
                  applicability=Applicability(descriptor="premium", attributes=[
                      ScopeAttribute(name="card_tier", kind=AttributeKind.CATEGORICAL, values=values)]),
                  supporting_passage="p1")


def test_default_branch_from_the_rule_passage():
    p = Passage(id="p0", text="A 3% fee applies to international payments. It is charged monthly.")
    b = B.default_branch(p)
    assert b.is_default and b.supporting_passage == "p0"
    assert b.outcome == "a 3% fee applies to international payments"


def test_reader_membership():
    default = B.default_branch(Passage(id="p0", text="A fee applies."))
    exc = _exc(["premium"])
    assert B.satisfies(default, {}) is True
    assert B.satisfies(exc, {"card_tier": "premium"}) is True
    assert B.satisfies(exc, {"card_tier": "standard"}) is False
    assert B.branch_for_reader([default, exc], {"card_tier": "premium"}).branch_id == "b1"
    assert B.branch_for_reader([default, exc], {"card_tier": "standard"}).branch_id == "b0"
    free = Branch(branch_id="b2", condition="students", outcome="no fee",
                  applicability=Applicability(descriptor="students"), supporting_passage="p1")
    assert B.satisfies(free, {"x": 1}) is None  # undecidable, not guessed


def test_chunking_keeps_paragraphs_apart():
    text = ("A monthly fee of 12.00 applies to all current accounts.\n\n"
            "Graduate Scheme accounts pay no monthly fee for three years.")
    chunks = ingest.chunk_text(text)
    assert len(chunks) == 2


def test_ingest_metadata():
    ps = ingest.passages_from_text("Updated 3 March 2024. Fees apply to all transfers made abroad.",
                                   doc_id="d1", name="https://www.gov.uk/fees")
    assert ps[0].source_type.value == "government_guidance"
    assert ps[0].date == "2024-03-03"
    assert ps[0].id == "d1:0" and ps[0].document_id == "d1"


def test_html_to_text_drops_chrome():
    text, title = ingest.html_to_text(
        "<html><head><title>Fees</title><script>x()</script></head>"
        "<body><nav>menu</nav><p>A fee applies.</p><p>Unless you are a student.</p></body></html>")
    assert title == "Fees" and "menu" not in text and "x()" not in text
    assert "A fee applies." in text and "Unless you are a student." in text


# --------------------------------------------------------------------------- #
# Tracing client
# --------------------------------------------------------------------------- #


def test_tracing_client_reports_calls_to_the_running_trace():
    from service.tracing_client import TracingLLMClient

    client = TracingLLMClient(backend_override=MockBackend(), cache=RequestCache(enabled=False),
                              cost_log=CostLog(enabled=False))
    tr = Tracer("t")
    client.complete(messages=[{"role": "user", "content": "outside any run"}], tier="BULK", step="x")
    assert not tr.events
    with use_tracer(tr):
        client.complete(messages=[{"role": "user", "content": "hello"}], tier="BULK", step="b1_test")
    llm = [e for e in tr.events if e.kind == "llm"]
    assert len(llm) == 1 and llm[0].payload["step"] == "b1_test" and "hello" in llm[0].payload["prompt"]
    assert tr.llm["calls"] == 1


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient

    from service.app import create_app
    from service.store import Store

    with TestClient(create_app(Store(tmp_path / "studio.sqlite"))) as c:
        yield c


def _stream(client, url):
    events = []
    with client.stream("GET", url) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                events.append(ev)
                if ev["stage"] == "run" and ev["kind"] in ("end", "error"):
                    break
    return events


def test_api_health_and_corpus(client):
    h = client.get("/v1/health").json()
    assert h["ok"] and h["benchmark_records"] == len(benchmark())
    rows = client.get("/v1/corpus", params={"branchy": True}).json()
    assert rows and all(r["gold_branches"] for r in rows)
    assert client.get(f"/v1/corpus/{MISSED}").json()["instance_id"] == MISSED
    assert client.get("/v1/corpus/nope").status_code == 404


def test_api_run_streams_to_the_end_and_is_stored(client):
    r = client.post("/v1/runs", json={"source": "benchmark", "record_id": MISSED})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    events = _stream(client, r.json()["events_url"])
    assert events[-1]["kind"] == "end" and events[-1]["payload"]["summary"]["resolution"] == "composed"
    assert {e["stage"] for e in events} >= set(STAGES)
    stored = client.get(f"/v1/runs/{run_id}/events.json").json()
    assert len(stored["events"]) == len(events)
    assert client.get(f"/v1/runs/{run_id}").json()["status"] == "done"
    assert any(x["id"] == run_id for x in client.get("/v1/runs").json())
    # a late subscriber replays the stored run
    assert len(_stream(client, f"/v1/runs/{run_id}/events")) == len(events)
    assert client.delete(f"/v1/runs/{run_id}").json()["deleted"] is True


def test_api_rejects_bad_requests(client):
    assert client.post("/v1/runs", json={"source": "benchmark"}).status_code == 422
    assert client.post("/v1/runs", json={"source": "benchmark", "record_id": "x", "mode": "warp"}).status_code == 422


def test_api_resolve_with_documents(client):
    r = client.post("/v1/resolve", json={
        "query": "Is there a fee for international payments?", "detector_policy": "override",
        "documents": ["A 3% fee applies to all international card payments.",
                      "Premium cardholders pay no fee on international payments."]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] and body["branches"] and body["run_id"].startswith("resolve_")


def test_api_collections_upload_run_and_audit(client):
    c = client.post("/v1/collections", json={"name": "Card fees"}).json()
    cid = c["id"]
    doc = client.post(f"/v1/collections/{cid}/documents", json={
        "name": "terms.txt",
        "text": ("A 3% fee applies to all international card payments.\n\n"
                 "Premium cardholders pay no fee on international card payments.\n\n"
                 "Cash withdrawals abroad cost 2.50 each.\n\n"
                 "Statements are sent monthly by email.")}).json()
    assert doc["passages"] == 4
    up = client.post(f"/v1/collections/{cid}/upload",
                     files=[("files", ("faq.md", b"Students pay no international payment fee until graduation.",
                                       "text/markdown"))]).json()
    assert up["added"][0]["passages"] == 1
    assert client.get(f"/v1/collections/{cid}").json()["passages"] == 5

    r = client.post("/v1/runs", json={"source": "collection", "collection_id": cid, "k": 3,
                                      "query": "Is there a fee on international card payments?"})
    events = _stream(client, r.json()["events_url"])
    assert events[-1]["kind"] == "end"
    ranks = [e for e in events if e["stage"] == "retrieve" and e["title"].startswith("rankings")]
    assert ranks and ranks[0]["payload"]["bm25"]

    a = client.post(f"/v1/collections/{cid}/audit", json={"analyse_top": 3}).json()
    aev = _stream(client, a["events_url"])
    assert aev[-1]["kind"] == "end"
    assert [e for e in aev if e["stage"] == "analyse" and e["title"] == "finding"]


def test_api_eval_streams_items_and_metrics(client):
    r = client.post("/v1/evals", json={"split": "dev", "limit": 3})
    events = _stream(client, r.json()["events_url"])
    items = [e for e in events if e["title"] == "item"]
    metrics = [e for e in events if e["title"] == "metrics"]
    assert len(items) == 3 and len(metrics) == 3
    assert metrics[-1]["payload"]["n"] == 3
    assert events[-1]["payload"]["summary"]["n"] == 3


def test_api_costs_shape(client):
    c = client.get("/v1/costs").json()
    assert set(c) == {"total", "by_step"}

"""The ScopeCompose API.

Everything the studio does goes through here, and everything here is usable
without the studio: start a run and stream its events, resolve a question in
one call, manage document collections, audit a collection for hidden
conditional conflicts, and evaluate on the benchmark.

Interactive docs at ``/docs`` (Swagger) and ``/redoc``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from service import ingest, orchestrator
from service.events import Event, Tracer, use_tracer
from service.store import Store

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent
COST_LOG = Path(os.environ.get("CONFLICT_RAG_COSTLOG") or ROOT / "logs" / "api_cost.jsonl")


# --------------------------------------------------------------------------- #
# Runs: worker threads in, events out (store + live subscribers)
# --------------------------------------------------------------------------- #


@dataclass
class _Live:
    loop: asyncio.AbstractEventLoop
    events: list[dict[str, Any]] = field(default_factory=list)
    queues: list[asyncio.Queue] = field(default_factory=list)
    done: bool = False


def _new_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{secrets.token_hex(3)}"


class RunManager:
    def __init__(self, store: Store, workers: int = 2):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="run")
        self.live: dict[str, _Live] = {}

    def start(self, kind: str, request: dict[str, Any], work: Callable[[Tracer], dict[str, Any]]) -> str:
        run_id = _new_id(kind)
        self.store.create_run(run_id, request, kind=kind)
        state = _Live(loop=asyncio.get_running_loop())
        self.live[run_id] = state

        def sink(ev: Event) -> None:
            d = ev.to_dict()
            self.store.add_event(ev)
            state.events.append(d)
            for q in list(state.queues):
                state.loop.call_soon_threadsafe(q.put_nowait, d)

        def job() -> None:
            tracer = Tracer(run_id, sink=sink)
            tracer.emit("run", "start", kind, {"kind": kind, "request": request})
            try:
                with use_tracer(tracer):
                    summary = work(tracer)
                summary = {**summary, "llm": dict(tracer.llm),
                           "stage_ms": {k: round(v, 1) for k, v in tracer.stage_ms.items()}}
                tracer.emit("run", "end", "done", {"summary": summary,
                                                   "ms": tracer.events[-1].t_ms if tracer.events else 0})
                self.store.finish_run(run_id, "done", summary)
            except Exception as exc:
                tracer.emit("run", "error", f"{type(exc).__name__}: {exc}"[:400], {})
                self.store.finish_run(run_id, "error", {"error": f"{type(exc).__name__}: {exc}"[:400]})
            finally:
                state.done = True

        self.pool.submit(job)
        return run_id

    async def stream(self, run_id: str, after: int = -1) -> AsyncIterator[dict[str, Any]]:
        state = self.live.get(run_id)
        if state is None or state.done:
            for ev in self.store.events(run_id, after):
                yield ev
            return
        q: asyncio.Queue = asyncio.Queue()
        state.queues.append(q)
        try:
            last = after
            for ev in list(state.events):
                if ev["seq"] > last:
                    last = ev["seq"]
                    yield ev
            while True:
                if state.done and q.empty():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {"keepalive": True}
                    continue
                if ev["seq"] > last:
                    last = ev["seq"]
                    yield ev
                if ev["stage"] == "run" and ev["kind"] in ("end", "error"):
                    break
        finally:
            state.queues.remove(q)


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #


class PassageIn(BaseModel):
    text: str = Field(..., min_length=1)
    source_type: str | None = Field(default=None, examples=["official_policy"])
    date: str | None = Field(default=None, examples=["2024-04"])
    url: str | None = None


class RunIn(BaseModel):
    """What to run. Exactly one source: a benchmark record, a collection, or passages."""

    query: str = Field(default="", examples=["Is there a monthly fee on my current account?"])
    source: str = Field(default="benchmark", description="benchmark | collection | passages")
    record_id: str | None = Field(default=None, examples=["wp1_fin_imp_003"])
    collection_id: str | None = None
    passages: list[PassageIn] = Field(default_factory=list)
    mode: str = Field(default="offline", description="offline (deterministic, no keys) | live (models + LLM)")
    detector_policy: str = Field(default="override", description="as_run | override (continue when stage 1 misses, labelled)")
    generation: str = Field(default="template", description="template | llm (live only)")
    reader: dict[str, Any] = Field(default_factory=dict, description="attribute -> value, e.g. {\"card_tier\": \"premium\"}")
    k: int = Field(default=5, ge=2, le=10)
    long_context_baseline: bool = Field(default=False, description="also run the long-context LLM baseline (live only)")


class ResolveIn(RunIn):
    documents: list[str] = Field(default_factory=list, description="raw texts; chunked into passages")


class CollectionIn(BaseModel):
    name: str = Field(..., min_length=1, examples=["Card fees 2024"])
    description: str = ""


class TextDocIn(BaseModel):
    name: str = Field(default="pasted text")
    text: str | None = None
    url: str | None = None
    source_type: str | None = None


class AuditIn(BaseModel):
    mode: str = "offline"
    max_pairs: int = Field(default=300, ge=1, le=5000)
    analyse_top: int = Field(default=12, ge=1, le=100)
    min_overlap: float = Field(default=0.1, ge=0.0, le=1.0)
    topic: str | None = None


class EvalIn(BaseModel):
    split: str = Field(default="dev", description="all | train | dev | test")
    limit: int | None = Field(default=25, ge=1, le=500)
    mode: str = "offline"
    detector_policy: str = "as_run"
    branchy_only: bool = Field(default=False, description="only instances with gold branches")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def _request(body: BaseModel) -> orchestrator.RunRequest:
    d = body.model_dump()
    d["passages"] = [p for p in d.get("passages") or []]
    try:
        return orchestrator.RunRequest.from_dict(d)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None


def _sse(gen: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    async def body() -> AsyncIterator[bytes]:
        yield b"retry: 2000\n\n"
        async for ev in gen:
            if ev.get("keepalive"):
                yield b": keepalive\n\n"
            else:
                yield f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def create_app(store: Store | None = None) -> FastAPI:
    app = FastAPI(
        title="ScopeCompose API",
        version=VERSION,
        description=(
            "Conditional-conflict-aware answering over retrieved documents. "
            "Every run streams its stages as events: retrieval, conflict detection, "
            "scope analysis, condition extraction, composition, generation and verification. "
            "Offline mode is deterministic and needs no keys; live mode loads the NLI and "
            "retrieval models and calls the LLM tiers configured in config/models.yaml."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\]|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+"
                           r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?|https://[a-z0-9-]+\.github\.io",
        allow_methods=["*"], allow_headers=["*"],
    )
    app.state.store = store or Store()
    app.state.runs = RunManager(app.state.store)

    def runs() -> RunManager:
        return app.state.runs

    def st() -> Store:
        return app.state.store

    # ---- meta -------------------------------------------------------------------

    @app.get("/v1/health", tags=["meta"], summary="Status, and what live mode needs")
    def health() -> dict[str, Any]:
        import settings

        groq = settings.models().providers.get("groq")
        has = lambda m: importlib.util.find_spec(m) is not None  # noqa: E731
        return {
            "ok": True, "version": VERSION,
            "engines_loaded": sorted("live" if k else "offline" for k in orchestrator._engines),
            "benchmark_records": len(orchestrator.benchmark()),
            "live": {"llm_configured": bool(groq and groq.configured), "torch": has("torch"),
                     "transformers": has("transformers"), "sentence_transformers": has("sentence_transformers")},
            "live_ready": bool(groq and groq.configured) and has("torch") and has("transformers"),
        }

    @app.get("/v1/corpus", tags=["benchmark"], summary="Benchmark records, for picking a run")
    def corpus(q: str = "", branchy: bool = False) -> list[dict[str, Any]]:
        out = []
        for i, inst in sorted(orchestrator.benchmark().items()):
            if branchy and not inst.gold_branches:
                continue
            if q and q.lower() not in (inst.query + " " + i).lower():
                continue
            out.append({"id": i, "query": inst.query, "domain": inst.domain.value, "split": inst.split,
                        "construction": inst.construction.value, "gold_type": inst.gold_conflict_type.value,
                        "gold_relation": inst.gold_scope_relation.value if inst.gold_scope_relation else None,
                        "gold_branches": len(inst.gold_branches)})
        return out

    @app.get("/v1/corpus/{record_id}", tags=["benchmark"], summary="One benchmark record, with gold")
    def corpus_record(record_id: str) -> dict[str, Any]:
        inst = orchestrator.benchmark().get(record_id)
        if inst is None:
            raise HTTPException(404, f"no record {record_id!r}")
        return inst.model_dump(mode="json")

    # ---- runs ---------------------------------------------------------------------

    @app.post("/v1/runs", tags=["runs"], status_code=202, summary="Start a run; stream it from events_url")
    async def start_run(body: RunIn) -> dict[str, Any]:
        req = _request(body)
        run_id = runs().start("run", body.model_dump(), lambda tr: orchestrator.run_query(req, tr, store=st()))
        return {"run_id": run_id, "events_url": f"/v1/runs/{run_id}/events"}

    @app.get("/v1/runs", tags=["runs"], summary="Run history")
    def list_runs(limit: int = 50, offset: int = 0, q: str = "", kind: str | None = None) -> list[dict[str, Any]]:
        return st().list_runs(limit=min(limit, 500), offset=offset, q=q, kind=kind)

    @app.get("/v1/runs/{run_id}", tags=["runs"], summary="One run: request, status, summary")
    def get_run(run_id: str) -> dict[str, Any]:
        r = st().get_run(run_id)
        if r is None:
            raise HTTPException(404, f"no run {run_id!r}")
        return r

    @app.get("/v1/runs/{run_id}/events", tags=["runs"], summary="Server-sent events: stored first, then live")
    async def run_events(run_id: str, request: Request, after: int = -1) -> StreamingResponse:
        if st().get_run(run_id) is None:
            raise HTTPException(404, f"no run {run_id!r}")
        last = request.headers.get("last-event-id")
        return _sse(runs().stream(run_id, int(last) if last and last.isdigit() else after))

    @app.get("/v1/runs/{run_id}/events.json", tags=["runs"], summary="Every event of a run, as JSON")
    def run_events_json(run_id: str) -> dict[str, Any]:
        r = st().get_run(run_id)
        if r is None:
            raise HTTPException(404, f"no run {run_id!r}")
        return {"run": r, "events": st().events(run_id)}

    @app.delete("/v1/runs/{run_id}", tags=["runs"], summary="Delete a run and its events")
    def delete_run(run_id: str) -> dict[str, Any]:
        return {"deleted": st().delete_run(run_id)}

    @app.post("/v1/resolve", tags=["product"], summary="Answer a question in one call")
    async def resolve(body: ResolveIn) -> dict[str, Any]:
        """Synchronous: returns the scoped answer, its branches, citations and checks.

        Give ``documents`` (raw texts), ``passages``, a ``collection_id`` or a ``record_id``."""
        d = body.model_dump()
        docs = d.pop("documents", []) or []
        if docs:
            ps = []
            for i, text in enumerate(docs):
                ps += [{"text": p.text, "source_type": p.source_type.value, "date": p.date}
                       for p in ingest.passages_from_text(text, doc_id=f"d{i}")]
            d.update(source="passages", passages=ps)
        try:
            req = orchestrator.RunRequest.from_dict(d)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from None
        run_id = _new_id("resolve")
        st().create_run(run_id, d, kind="resolve")
        tracer = Tracer(run_id, sink=st().add_event)
        try:
            summary = await asyncio.get_running_loop().run_in_executor(
                runs().pool, lambda: orchestrator.run_query(req, tracer, store=st()))
        except Exception as exc:
            st().finish_run(run_id, "error", {"error": str(exc)[:400]})
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from None
        st().finish_run(run_id, "done", summary)
        return {"run_id": run_id, **summary}

    # ---- collections ---------------------------------------------------------------

    @app.post("/v1/collections", tags=["collections"], status_code=201, summary="Create a document collection")
    def create_collection(body: CollectionIn) -> dict[str, Any]:
        cid = re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-")[:32] or "collection"
        cid = f"{cid}-{secrets.token_hex(2)}"
        return st().create_collection(cid, body.name, body.description)

    @app.get("/v1/collections", tags=["collections"], summary="All collections")
    def list_collections() -> list[dict[str, Any]]:
        return st().list_collections()

    @app.get("/v1/collections/{cid}", tags=["collections"], summary="One collection and its documents")
    def get_collection(cid: str) -> dict[str, Any]:
        c = st().get_collection(cid)
        if c is None:
            raise HTTPException(404, f"no collection {cid!r}")
        return c

    @app.get("/v1/collections/{cid}/passages", tags=["collections"], summary="Every passage in a collection")
    def collection_passages(cid: str) -> list[dict[str, Any]]:
        if st().get_collection(cid) is None:
            raise HTTPException(404, f"no collection {cid!r}")
        return [p.model_dump(mode="json") for p in st().passages(cid)]

    @app.delete("/v1/collections/{cid}", tags=["collections"], summary="Delete a collection")
    def delete_collection(cid: str) -> dict[str, Any]:
        return {"deleted": st().delete_collection(cid)}

    def _add(cid: str, name: str, text: str, *, url: str | None, kind: str, source_type: str | None):
        from contract.models import SourceType

        if st().get_collection(cid) is None:
            raise HTTPException(404, f"no collection {cid!r}")
        doc_id = f"d{secrets.token_hex(3)}"
        stype = SourceType(source_type) if source_type in SourceType._value2member_map_ else None
        ps = ingest.passages_from_text(text, doc_id=doc_id, name=name, source_url=url, source_type=stype)
        if not ps:
            raise HTTPException(422, "no usable text found in that document")
        return st().add_document(cid, doc_id, name=name, url=url, kind=kind, chars=len(text), passages=ps)

    @app.post("/v1/collections/{cid}/documents", tags=["collections"], status_code=201,
              summary="Add a document: JSON {text} or {url}")
    def add_text_document(cid: str, body: TextDocIn) -> dict[str, Any]:
        if body.url:
            try:
                text, title = ingest.fetch_url(body.url)
            except Exception as exc:
                raise HTTPException(400, f"could not fetch {body.url}: {exc}") from None
            return _add(cid, title or body.url, text, url=body.url, kind="url", source_type=body.source_type)
        if not (body.text or "").strip():
            raise HTTPException(422, "give text or url")
        return _add(cid, body.name, body.text or "", url=None, kind="text", source_type=body.source_type)

    @app.post("/v1/collections/{cid}/upload", tags=["collections"], status_code=201,
              summary="Upload files (.txt, .md, .pdf, .html)")
    async def upload(cid: str, files: list[UploadFile] = File(...), source_type: str | None = Form(None)):
        added = []
        for f in files:
            data = await f.read()
            name = f.filename or "upload"
            low = name.lower()
            if low.endswith(".pdf"):
                text = ingest.read_pdf(data)
            elif low.endswith((".html", ".htm")):
                text = ingest.html_to_text(data.decode("utf-8", "replace"))[0]
            else:
                text = data.decode("utf-8", "replace")
            added.append(_add(cid, name, text, url=None, kind=low.rsplit(".", 1)[-1], source_type=source_type))
        return {"added": added}

    @app.post("/v1/collections/{cid}/audit", tags=["product"], status_code=202,
              summary="Scan a collection for hidden conditional conflicts")
    async def audit(cid: str, body: AuditIn) -> dict[str, Any]:
        from service.audit import run_audit

        coll = st().get_collection(cid)
        if coll is None:
            raise HTTPException(404, f"no collection {cid!r}")
        passages = st().passages(cid)
        if len(passages) < 2:
            raise HTTPException(422, "the collection needs at least two passages")
        params = body.model_dump()
        run_id = runs().start("audit", {"collection_id": cid, "query": f"audit: {coll['name']}", **params},
                              lambda tr: run_audit(passages, params, tr))
        return {"run_id": run_id, "events_url": f"/v1/runs/{run_id}/events"}

    # ---- evaluation ---------------------------------------------------------------

    @app.post("/v1/evals", tags=["evaluation"], status_code=202, summary="Evaluate on a benchmark split, streamed")
    async def start_eval(body: EvalIn) -> dict[str, Any]:
        from service.evals import SPLITS, run_eval

        if body.split not in SPLITS:
            raise HTTPException(422, f"split must be one of {SPLITS}")
        params = body.model_dump()
        run_id = runs().start("eval", {"query": f"eval: {body.split} ×{body.limit or 'all'}", **params},
                              lambda tr: run_eval(params, tr))
        return {"run_id": run_id, "events_url": f"/v1/runs/{run_id}/events"}

    # ---- costs --------------------------------------------------------------------

    @app.get("/v1/costs", tags=["meta"], summary="LLM spend and latency, by pipeline step and model")
    def costs() -> dict[str, Any]:
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        total = {"calls": 0, "cost_usd": 0.0, "tokens": 0, "cached": 0}
        if COST_LOG.exists():
            for line in COST_LOG.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                key = (r.get("step", "?"), r.get("model", "?"))
                a = rows.setdefault(key, {"step": key[0], "model": key[1], "calls": 0, "tokens": 0,
                                          "cost_usd": 0.0, "cached": 0, "latency_s": 0.0, "errors": 0})
                a["calls"] += 1
                toks = int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
                a["tokens"] += toks
                a["cost_usd"] += float(r.get("cost_usd") or 0.0)
                a["cached"] += int(bool(r.get("cached")))
                a["latency_s"] += float(r.get("latency_s") or 0.0)
                a["errors"] += int(bool(r.get("error")))
                total["calls"] += 1
                total["tokens"] += toks
                total["cost_usd"] += float(r.get("cost_usd") or 0.0)
                total["cached"] += int(bool(r.get("cached")))
        out = sorted(rows.values(), key=lambda a: -a["calls"])
        for a in out:
            a["mean_latency_s"] = a.pop("latency_s") / a["calls"] if a["calls"] else 0.0
        return {"total": total, "by_step": out}

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"detail": str(exc).strip("'\"")}, status_code=404)

    return app

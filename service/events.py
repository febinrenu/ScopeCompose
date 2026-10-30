"""Run events: the one shape every consumer reads.

A run is a sequence of events. The studio draws from them, the store keeps
them, replay plays them back, and the API's synchronous endpoint folds them
into a result -- so there is exactly one description of what happened.

    {run_id, seq, t_ms, stage, kind, title, payload}

``kind`` is one of ``start | data | log | llm | warn | end | error``. A stage
opens with ``start`` and closes with ``end`` (carrying ``ms``); in between it
emits ``data`` (what it computed), ``log`` lines and, through the tracing LLM
client, one ``llm`` event per model call.
"""

from __future__ import annotations

import dataclasses
import enum
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

KINDS = ("start", "data", "log", "llm", "warn", "end", "error")


@dataclass
class Event:
    run_id: str
    seq: int
    t_ms: float
    stage: str
    kind: str
    title: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "seq": self.seq, "t_ms": round(self.t_ms, 2),
                "stage": self.stage, "kind": self.kind, "title": self.title,
                "payload": self.payload}


def jsonable(obj: Any) -> Any:
    """Plain JSON from pydantic models, dataclasses, enums, tuples and sets."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, enum.Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return jsonable(obj.model_dump(mode="json"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k if not isinstance(k, tuple) else "|".join(map(str, k))): jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return str(obj)


class Tracer:
    """Collects one run's events and hands each to a sink as it happens.

    Thread-confined: a run executes in one worker thread, and the tracing LLM
    client finds the tracer for that thread through :func:`current`.
    """

    def __init__(self, run_id: str, sink: Callable[[Event], None] | None = None):
        self.run_id = run_id
        self.sink = sink
        self.events: list[Event] = []
        self._t0 = time.perf_counter()
        self._seq = 0
        self._lock = threading.Lock()
        self.stage_ms: dict[str, float] = {}
        self.llm = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "cost_usd": 0.0, "cached": 0, "fell_back": 0, "errors": 0}

    def emit(self, stage: str, kind: str, title: str, payload: dict[str, Any] | None = None) -> Event:
        assert kind in KINDS, kind
        with self._lock:
            ev = Event(self.run_id, self._seq, (time.perf_counter() - self._t0) * 1000.0,
                       stage, kind, title, jsonable(payload or {}))
            self._seq += 1
            self.events.append(ev)
        if self.sink is not None:
            self.sink(ev)
        return ev

    def data(self, stage: str, title: str, **payload: Any) -> Event:
        return self.emit(stage, "data", title, payload)

    def log(self, stage: str, message: str, level: str = "info") -> Event:
        return self.emit(stage, "log", message, {"level": level})

    def warn(self, stage: str, title: str, **payload: Any) -> Event:
        return self.emit(stage, "warn", title, payload)

    @contextmanager
    def stage(self, name: str, title: str, **payload: Any) -> Iterator[None]:
        """Open a stage; closes with its duration, or with the error that ended it."""
        t = time.perf_counter()
        self.emit(name, "start", title, payload)
        try:
            yield
        except Exception as exc:
            ms = (time.perf_counter() - t) * 1000.0
            self.stage_ms[name] = ms
            self.emit(name, "error", f"{type(exc).__name__}: {exc}"[:400], {"ms": ms})
            raise
        ms = (time.perf_counter() - t) * 1000.0
        self.stage_ms[name] = self.stage_ms.get(name, 0.0) + ms
        self.emit(name, "end", title, {"ms": ms})

    def record_llm(self, *, prompt_tokens: int, completion_tokens: int, cost_usd: float,
                   cached: bool, fell_back: bool, error: bool = False) -> None:
        with self._lock:
            self.llm["calls"] += 1
            self.llm["prompt_tokens"] += prompt_tokens
            self.llm["completion_tokens"] += completion_tokens
            self.llm["cost_usd"] += cost_usd
            self.llm["cached"] += int(cached)
            self.llm["fell_back"] += int(fell_back)
            self.llm["errors"] += int(error)


# -- the tracer for the current thread --------------------------------------- #

_local = threading.local()


def current() -> Tracer | None:
    return getattr(_local, "tracer", None)


def current_stage() -> str:
    return getattr(_local, "stage", "run")


@contextmanager
def use_tracer(tracer: Tracer) -> Iterator[Tracer]:
    prev = current()
    _local.tracer = tracer
    try:
        yield tracer
    finally:
        _local.tracer = prev


@contextmanager
def in_stage(tracer: Tracer, name: str, title: str, **payload: Any) -> Iterator[None]:
    """A stage that LLM calls made inside it are attributed to."""
    prev = current_stage()
    _local.stage = name
    try:
        with tracer.stage(name, title, **payload):
            yield
    finally:
        _local.stage = prev

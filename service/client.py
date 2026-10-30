"""A small Python client for the ScopeCompose API.

    from service.client import ScopeCompose

    sc = ScopeCompose("http://localhost:8008")
    r = sc.resolve("Is there a fee on international payments?",
                   documents=["A 3% fee applies to all international payments.",
                              "Premium cardholders pay no fee on international payments."])
    print(r["answer"])

    for ev in sc.stream(sc.start(source="benchmark", record_id="wp1_fin_imp_003")):
        print(ev["stage"], ev["kind"], ev["title"])

Only needs ``httpx``.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx


class ScopeCompose:
    def __init__(self, base_url: str = "http://localhost:8008", *, timeout: float = 300.0):
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base, timeout=timeout)

    # -- product ------------------------------------------------------------- #

    def resolve(self, query: str, *, documents: list[str] | None = None, passages: list[dict] | None = None,
                collection_id: str | None = None, record_id: str | None = None, mode: str = "offline",
                detector_policy: str = "override", reader: dict[str, Any] | None = None, **extra: Any) -> dict:
        """Answer a question in one call."""
        body: dict[str, Any] = {"query": query, "mode": mode, "detector_policy": detector_policy,
                                "reader": reader or {}, **extra}
        if documents:
            body["documents"] = documents
        elif passages:
            body.update(source="passages", passages=passages)
        elif collection_id:
            body.update(source="collection", collection_id=collection_id)
        elif record_id:
            body.update(source="benchmark", record_id=record_id)
        return self._json(self.http.post("/v1/resolve", json=body))

    # -- runs ------------------------------------------------------------------ #

    def start(self, **request: Any) -> str:
        """Start a run (same fields as POST /v1/runs) and return its id."""
        return self._json(self.http.post("/v1/runs", json=request))["run_id"]

    def stream(self, run_id: str, *, after: int = -1) -> Iterator[dict]:
        """Every event of a run, as it happens, until the run ends."""
        with self.http.stream("GET", f"/v1/runs/{run_id}/events", params={"after": after}, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    yield ev
                    if ev["stage"] == "run" and ev["kind"] in ("end", "error"):
                        return

    def run(self, run_id: str) -> dict:
        return self._json(self.http.get(f"/v1/runs/{run_id}"))

    def history(self, *, q: str = "", kind: str | None = None, limit: int = 50) -> list[dict]:
        params: dict[str, Any] = {"q": q, "limit": limit}
        if kind:
            params["kind"] = kind
        return self._json(self.http.get("/v1/runs", params=params))

    # -- collections -------------------------------------------------------------- #

    def create_collection(self, name: str, description: str = "") -> dict:
        return self._json(self.http.post("/v1/collections", json={"name": name, "description": description}))

    def add_text(self, collection_id: str, text: str, *, name: str = "text") -> dict:
        return self._json(self.http.post(f"/v1/collections/{collection_id}/documents", json={"name": name, "text": text}))

    def add_url(self, collection_id: str, url: str) -> dict:
        return self._json(self.http.post(f"/v1/collections/{collection_id}/documents", json={"url": url}))

    def upload(self, collection_id: str, *paths: str) -> dict:
        files = [("files", (p.replace("\\", "/").rsplit("/", 1)[-1], open(p, "rb"))) for p in paths]
        try:
            return self._json(self.http.post(f"/v1/collections/{collection_id}/upload", files=files))
        finally:
            for _, (_, fh) in files:
                fh.close()

    def audit(self, collection_id: str, **params: Any) -> str:
        return self._json(self.http.post(f"/v1/collections/{collection_id}/audit", json=params))["run_id"]

    def evaluate(self, **params: Any) -> str:
        return self._json(self.http.post("/v1/evals", json=params))["run_id"]

    def health(self) -> dict:
        return self._json(self.http.get("/v1/health"))

    @staticmethod
    def _json(r: httpx.Response) -> Any:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text
            raise RuntimeError(f"HTTP {r.status_code}: {detail}")
        return r.json()

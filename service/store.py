"""Persistence: runs, their events, and document collections (SQLite).

One file (``logs/studio.sqlite`` by default; ``SCOPECOMPOSE_STUDIO_DB`` to
move it). Every event of every run is kept, which is what makes history,
replay, comparison and export work without re-running anything.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from contract.models import Passage
from service.events import Event

DEFAULT_DB = Path(__file__).resolve().parent.parent / "logs" / "studio.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, created REAL NOT NULL, finished REAL,
  status TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'run',
  query TEXT, request TEXT NOT NULL, summary TEXT
);
CREATE TABLE IF NOT EXISTS events (
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, body TEXT NOT NULL,
  PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS collections (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, name TEXT, url TEXT, kind TEXT,
  created REAL NOT NULL, chars INTEGER, passages TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_created ON runs(created DESC);
CREATE INDEX IF NOT EXISTS docs_coll ON documents(collection_id);
"""


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("SCOPECOMPOSE_STUDIO_DB") or DEFAULT_DB)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # -- runs ------------------------------------------------------------------ #

    def create_run(self, run_id: str, request: dict[str, Any], *, kind: str = "run") -> None:
        self._exec("INSERT INTO runs (id, created, status, kind, query, request) VALUES (?,?,?,?,?,?)",
                   (run_id, time.time(), "running", kind, request.get("query"), json.dumps(request)))

    def finish_run(self, run_id: str, status: str, summary: dict[str, Any] | None) -> None:
        self._exec("UPDATE runs SET finished=?, status=?, summary=? WHERE id=?",
                   (time.time(), status, json.dumps(summary or {}), run_id))

    def add_event(self, ev: Event) -> None:
        self._exec("INSERT OR REPLACE INTO events (run_id, seq, body) VALUES (?,?,?)",
                   (ev.run_id, ev.seq, json.dumps(ev.to_dict())))

    def events(self, run_id: str, after: int = -1) -> list[dict[str, Any]]:
        rows = self._all("SELECT body FROM events WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after))
        return [json.loads(r["body"]) for r in rows]

    @staticmethod
    def _run_row(r: sqlite3.Row) -> dict[str, Any]:
        return {"id": r["id"], "created": r["created"], "finished": r["finished"], "status": r["status"],
                "kind": r["kind"], "query": r["query"], "request": json.loads(r["request"] or "{}"),
                "summary": json.loads(r["summary"]) if r["summary"] else None}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM runs WHERE id=?", (run_id,))
        return self._run_row(rows[0]) if rows else None

    def list_runs(self, *, limit: int = 50, offset: int = 0, q: str = "", kind: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM runs WHERE 1=1", []
        if q:
            sql += " AND (query LIKE ? OR id LIKE ?)"
            args += [f"%{q}%", f"%{q}%"]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY created DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [self._run_row(r) for r in self._all(sql, tuple(args))]

    def delete_run(self, run_id: str) -> bool:
        self._exec("DELETE FROM events WHERE run_id=?", (run_id,))
        return self._exec("DELETE FROM runs WHERE id=?", (run_id,)).rowcount > 0

    # -- collections ----------------------------------------------------------- #

    def create_collection(self, cid: str, name: str, description: str = "") -> dict[str, Any]:
        self._exec("INSERT INTO collections (id, name, description, created) VALUES (?,?,?,?)",
                   (cid, name, description, time.time()))
        return self.get_collection(cid) or {}

    def get_collection(self, cid: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM collections WHERE id=?", (cid,))
        if not rows:
            return None
        r = rows[0]
        docs = self.documents(cid)
        return {"id": r["id"], "name": r["name"], "description": r["description"], "created": r["created"],
                "documents": docs, "passages": sum(d["passages"] for d in docs)}

    def list_collections(self) -> list[dict[str, Any]]:
        out = []
        for r in self._all("SELECT id FROM collections ORDER BY created DESC"):
            c = self.get_collection(r["id"])
            if c:
                c.pop("documents", None)
                out.append(c)
        return out

    def delete_collection(self, cid: str) -> bool:
        self._exec("DELETE FROM documents WHERE collection_id=?", (cid,))
        return self._exec("DELETE FROM collections WHERE id=?", (cid,)).rowcount > 0

    def add_document(self, cid: str, doc_id: str, *, name: str, url: str | None, kind: str,
                     chars: int, passages: list[Passage]) -> dict[str, Any]:
        self._exec("INSERT INTO documents (id, collection_id, name, url, kind, created, chars, passages)"
                   " VALUES (?,?,?,?,?,?,?,?)",
                   (doc_id, cid, name, url, kind, time.time(), chars,
                    json.dumps([p.model_dump(mode="json") for p in passages])))
        return {"id": doc_id, "name": name, "url": url, "kind": kind, "chars": chars, "passages": len(passages)}

    def documents(self, cid: str) -> list[dict[str, Any]]:
        rows = self._all("SELECT id, name, url, kind, created, chars, passages FROM documents"
                         " WHERE collection_id=? ORDER BY created", (cid,))
        return [{"id": r["id"], "name": r["name"], "url": r["url"], "kind": r["kind"], "created": r["created"],
                 "chars": r["chars"], "passages": len(json.loads(r["passages"]))} for r in rows]

    def passages(self, cid: str) -> list[Passage]:
        rows = self._all("SELECT passages FROM documents WHERE collection_id=? ORDER BY created", (cid,))
        return [Passage.model_validate(p) for r in rows for p in json.loads(r["passages"])]

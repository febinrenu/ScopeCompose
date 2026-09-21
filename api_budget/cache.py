"""On-disk cache for LLM requests.

Built in week one, not bolted on later, for a specific reason: development
re-runs the same pipeline over the same instances dozens of times. Without a
cache, every debugging iteration costs real rate-limit budget, and the second
run of a script is as expensive as the first. With one, it is free.

The key is a hash of everything that could change the response. Anything
omitted from the key is a correctness bug -- a cache that returns a response
generated under different settings is worse than no cache, because it is
silently wrong rather than slow.

Storage is SQLite: one file, no server, safe across processes, and easy to
delete when a prompt template changes and the old entries stop being valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path(
    os.environ.get("CONFLICT_RAG_CACHE", "api_budget/cache.sqlite")
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key               TEXT PRIMARY KEY,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    request_json      TEXT NOT NULL,
    response_text     TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_model ON responses(model);
CREATE INDEX IF NOT EXISTS idx_responses_created ON responses(created_at);
"""


@dataclass(frozen=True)
class CachedResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    created_at: float


def request_key(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    system: str | None = None,
    response_format: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stable hash of a request.

    ``sort_keys=True`` matters: Python dict ordering is insertion-ordered, so
    two semantically identical requests built in different orders would
    otherwise hash differently and silently miss the cache.
    """
    payload = {
        "provider": provider,
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system": system,
        "response_format": response_format,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RequestCache:
    """A SQLite-backed request cache.

    Thread-safe via a lock plus one connection per thread. Not designed for
    heavy concurrency -- this is a research pipeline, not a web service.
    """

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH, *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._local = threading.local()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # -- core operations ----------------------------------------------------- #

    def get(self, key: str) -> CachedResponse | None:
        if not self.enabled:
            return None
        with self._lock:
            row = self._connect().execute(
                "SELECT response_text, model, provider, prompt_tokens, completion_tokens, "
                "created_at FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return CachedResponse(
            text=row[0], model=row[1], provider=row[2],
            prompt_tokens=row[3], completion_tokens=row[4], created_at=row[5],
        )

    def put(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        request: dict[str, Any],
        text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(key, provider, model, request_json, response_text, prompt_tokens, "
                " completion_tokens, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    key, provider, model,
                    json.dumps(request, sort_keys=True, ensure_ascii=False),
                    text, prompt_tokens, completion_tokens, time.time(),
                ),
            )
            conn.commit()

    # -- maintenance --------------------------------------------------------- #

    def stats(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "hits": self.hits, "misses": self.misses}
        with self._lock:
            conn = self._connect()
            n, = conn.execute("SELECT COUNT(*) FROM responses").fetchone()
            by_model = dict(
                conn.execute(
                    "SELECT model, COUNT(*) FROM responses GROUP BY model ORDER BY 2 DESC"
                ).fetchall()
            )
            tokens = conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0) "
                "FROM responses"
            ).fetchone()
        total = self.hits + self.misses
        return {
            "enabled": True,
            "path": str(self.path),
            "entries": n,
            "by_model": by_model,
            "cached_prompt_tokens": tokens[0],
            "cached_completion_tokens": tokens[1],
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }

    def clear(self, *, model: str | None = None) -> int:
        """Delete entries. Scope to one model where possible.

        Needed whenever a prompt template changes: old entries are still
        returnable but no longer describe the prompt the code now sends, which
        is exactly the silently-wrong case this cache must not cause.
        """
        if not self.enabled:
            return 0
        with self._lock:
            conn = self._connect()
            if model:
                cur = conn.execute("DELETE FROM responses WHERE model = ?", (model,))
            else:
                cur = conn.execute("DELETE FROM responses")
            conn.commit()
            return cur.rowcount

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> RequestCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

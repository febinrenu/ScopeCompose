/*  Talking to the engine server, and the three ways a run arrives.
 *
 *    LiveSource    the server runs the real pipeline and streams events (SSE)
 *    ReplaySource  a stored run, played back at a chosen speed (any events)
 *    local runner  the browser's verified engines (local.ts), when no server
 *
 *  All three hand the console the same RunEvent objects.
 */
import type { RunEvent } from "./model";

const KEY = "sc.server";

/** Where the engine server is: a saved override, else this host on port 8008
 *  (so a phone that opened the site from the laptop finds the laptop's server). */
export function serverBase(): string {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved) return saved.replace(/\/+$/, "");
  } catch { /* storage may be unavailable */ }
  const h = location.hostname;
  const local = /^(localhost|127\.0\.0\.1|\[::1\]|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(h);
  return `http://${local ? h : "localhost"}:8008`;
}

export function setServerBase(url: string | null) {
  try {
    if (url) localStorage.setItem(KEY, url.replace(/\/+$/, ""));
    else localStorage.removeItem(KEY);
  } catch { /* ignore */ }
}

export interface Health {
  ok: boolean; version: string; engines_loaded: string[]; benchmark_records: number;
  live: { llm_configured: boolean; torch: boolean; transformers: boolean; sentence_transformers: boolean };
  live_ready: boolean;
}

async function call<T>(path: string, init?: RequestInit, timeout = 15000): Promise<T> {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeout);
  try {
    const r = await fetch(serverBase() + path, { ...init, signal: ctl.signal });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error((body as { detail?: string }).detail ? String((body as { detail: unknown }).detail) : `HTTP ${r.status}`);
    return body as T;
  } finally {
    clearTimeout(t);
  }
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  health: () => call<Health>("/v1/health", undefined, 2500),
  corpus: (branchy = false) => call<any[]>(`/v1/corpus${branchy ? "?branchy=true" : ""}`),
  startRun: (body: unknown) => call<{ run_id: string; events_url: string }>("/v1/runs", json(body)),
  runs: (q = "", kind = "") => call<any[]>(`/v1/runs?limit=100&q=${encodeURIComponent(q)}${kind ? `&kind=${kind}` : ""}`),
  run: (id: string) => call<any>(`/v1/runs/${encodeURIComponent(id)}`),
  runEvents: (id: string) => call<{ run: any; events: RunEvent[] }>(`/v1/runs/${encodeURIComponent(id)}/events.json`, undefined, 30000),
  deleteRun: (id: string) => call<any>(`/v1/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  resolve: (body: unknown) => call<any>("/v1/resolve", json(body), 300000),
  collections: () => call<any[]>("/v1/collections"),
  collection: (id: string) => call<any>(`/v1/collections/${encodeURIComponent(id)}`),
  passages: (id: string) => call<any[]>(`/v1/collections/${encodeURIComponent(id)}/passages`),
  createCollection: (name: string, description = "") => call<any>("/v1/collections", json({ name, description })),
  deleteCollection: (id: string) => call<any>(`/v1/collections/${encodeURIComponent(id)}`, { method: "DELETE" }),
  addText: (id: string, body: { name?: string; text?: string; url?: string; source_type?: string }) =>
    call<any>(`/v1/collections/${encodeURIComponent(id)}/documents`, json(body), 60000),
  upload: (id: string, files: File[], sourceType = "") => {
    const fd = new FormData();
    files.forEach(f => fd.append("files", f));
    if (sourceType) fd.append("source_type", sourceType);
    return call<any>(`/v1/collections/${encodeURIComponent(id)}/upload`, { method: "POST", body: fd }, 120000);
  },
  audit: (id: string, body: unknown) => call<{ run_id: string; events_url: string }>(`/v1/collections/${encodeURIComponent(id)}/audit`, json(body)),
  startEval: (body: unknown) => call<{ run_id: string; events_url: string }>("/v1/evals", json(body)),
  costs: () => call<any>("/v1/costs"),
};

// --------------------------------------------------------------------------- //
// Sources
// --------------------------------------------------------------------------- //

export interface Sink { event: (ev: RunEvent) => void; done: () => void }

/** A run on the server, streamed as it happens. */
export class LiveSource {
  private es: EventSource | null = null;
  private last = -1;
  constructor(private runId: string) {}

  static async start(body: unknown, endpoint: "run" | "eval" | "audit" = "run", collection = ""): Promise<LiveSource> {
    const r = endpoint === "eval" ? await api.startEval(body)
      : endpoint === "audit" ? await api.audit(collection, body)
      : await api.startRun(body);
    return new LiveSource(r.run_id);
  }

  get id() { return this.runId; }

  attach(sink: Sink) {
    const url = `${serverBase()}/v1/runs/${encodeURIComponent(this.runId)}/events`;
    const es = new EventSource(url);
    this.es = es;
    es.onmessage = m => {
      const ev = JSON.parse(m.data) as RunEvent;
      if (ev.seq <= this.last) return;
      this.last = ev.seq;
      sink.event(ev);
      if (ev.stage === "run" && (ev.kind === "end" || ev.kind === "error")) { es.close(); sink.done(); }
    };
    es.onerror = () => {
      // EventSource reconnects by itself (resuming from Last-Event-ID); give up only when closed
      if (es.readyState === EventSource.CLOSED) sink.done();
    };
  }

  stop() { this.es?.close(); this.es = null; }
}

/** Plays stored events back, keeping their rhythm (long gaps shortened). */
export class ReplaySource {
  private timer = 0;
  private i = 0;
  private stopped = false;
  /** `pace`: a minimum pause before each stage starts, so a run that took a few
   *  milliseconds can still be watched. Recorded timings are never altered. */
  constructor(private events: RunEvent[], private speed = 1, private maxGap = 700, private pace = 0) {}

  attach(sink: Sink) {
    const step = () => {
      if (this.stopped) return;
      if (this.i >= this.events.length) { sink.done(); return; }
      const ev = this.events[this.i++];
      sink.event(ev);
      const next = this.events[this.i];
      let gap = next ? Math.min(this.maxGap, Math.max(0, next.t_ms - ev.t_ms)) / this.speed : 0;
      if (next && next.kind === "start" && next.stage !== "run") gap = Math.max(gap, this.pace / this.speed);
      this.timer = window.setTimeout(step, gap);
    };
    step();
  }

  stop() { this.stopped = true; clearTimeout(this.timer); }
}

export function download(name: string, text: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

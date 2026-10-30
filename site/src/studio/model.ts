/*  The run model: one event shape, folded into one state.
 *
 *  Every source -- the engine server's stream, the browser engines, a stored
 *  run being replayed -- produces the same events. `apply` folds an event
 *  into the state every panel reads. It is pure in the sense that matters:
 *  the same events always give the same state, which is what makes replay and
 *  scrubbing exact.
 */

export type Kind = "start" | "data" | "log" | "llm" | "warn" | "end" | "error";

export interface RunEvent {
  run_id: string;
  seq: number;
  t_ms: number;
  stage: string;
  kind: Kind;
  title: string;
  payload: Record<string, any>;
}

export type StageStatus = "pending" | "running" | "done" | "skipped" | "warn" | "error";

export interface StageState {
  id: string;
  status: StageStatus;
  t0: number | null;
  t1: number | null;
  ms: number | null;
  title: string;
  data: { title: string; payload: Record<string, any>; t: number }[];
  logs: number;
  llm: number;
  note: string;
}

export interface LogLine { t: number; stage: string; level: string; text: string }

export interface LlmCall {
  t: number; stage: string; step: string; model: string; provider: string; tier: string;
  prompt_tokens: number; completion_tokens: number; cost_usd: number; cached: boolean;
  fell_back: boolean; ms: number; error?: string; system?: string; prompt?: string; response?: string;
}

export interface RunState {
  runId: string;
  kind: string;
  status: "idle" | "running" | "done" | "error";
  t: number;
  stages: Record<string, StageState>;
  order: string[];
  logs: LogLine[];
  llm: LlmCall[];
  contract: { title: string; record: any; t: number }[];
  counters: { calls: number; tokens: number; cost: number; cached: number };
  request: any;
  summary: any;
  error: string;
  events: RunEvent[];
}

/** The pipeline as drawn, in order. `setup` and `run` are bookkeeping, not nodes. */
export const PIPELINE = [
  "ingest", "retrieve", "detect", "type", "policy", "scope", "extract",
  "branches", "compose", "generate", "verify", "arena", "reader", "score",
] as const;

export const STAGE_LABEL: Record<string, string> = {
  setup: "Setup", ingest: "Ingest", retrieve: "Retrieve", detect: "Detect", type: "Type",
  policy: "Policy", scope: "Scope", extract: "Extract", branches: "Branches", compose: "Compose",
  generate: "Generate", verify: "Verify", arena: "Baselines", reader: "Reader", score: "Score",
  eval: "Evaluate", candidates: "Candidates", screen: "Screen", analyse: "Analyse", run: "Run",
};

export const STAGE_SUB: Record<string, string> = {
  ingest: "documents", retrieve: "BM25·dense·RRF", detect: "stage 1 · 2", type: "A3 · 5 classes",
  policy: "detector miss", scope: "A4 · relation", extract: "B1 probe·gate", branches: "rule + exceptions",
  compose: "B2 operator", generate: "B3 answer", verify: "faithfulness", arena: "same input",
  reader: "for a reader", score: "against gold",
};

export function emptyState(): RunState {
  return {
    runId: "", kind: "run", status: "idle", t: 0, stages: {}, order: [], logs: [], llm: [], contract: [],
    counters: { calls: 0, tokens: 0, cost: 0, cached: 0 }, request: null, summary: null, error: "", events: [],
  };
}

function stage(s: RunState, id: string): StageState {
  let st = s.stages[id];
  if (!st) {
    st = { id, status: "pending", t0: null, t1: null, ms: null, title: STAGE_LABEL[id] || id,
           data: [], logs: 0, llm: 0, note: "" };
    s.stages[id] = st;
    s.order.push(id);
  }
  return st;
}

/** Fold one event into the state (in place) and return it. */
export function apply(s: RunState, ev: RunEvent): RunState {
  s.events.push(ev);
  s.t = Math.max(s.t, ev.t_ms);
  if (!s.runId) s.runId = ev.run_id;

  if (ev.stage === "run") {
    if (ev.kind === "start") {
      s.status = "running"; s.kind = ev.payload.kind || ev.title; s.request = ev.payload.request ?? null;
    } else if (ev.kind === "end") {
      s.status = "done"; s.summary = ev.payload.summary ?? null;
    } else if (ev.kind === "error") {
      s.status = "error"; s.error = ev.title;
    } else if (ev.kind === "data" && ev.title === "request") {
      s.request = ev.payload.request ?? s.request;
    }
    return s;
  }
  if (s.status === "idle") s.status = "running";

  if (ev.stage === "contract") {
    if (ev.kind === "data") s.contract.push({ title: ev.title, record: ev.payload.record, t: ev.t_ms });
    return s;
  }

  const st = stage(s, ev.stage);
  switch (ev.kind) {
    case "start":
      st.status = "running"; st.t0 = ev.t_ms; st.title = ev.title;
      break;
    case "end":
      st.t1 = ev.t_ms; st.ms = ev.payload.ms ?? (st.t0 !== null ? ev.t_ms - st.t0 : null);
      if (st.status === "running") st.status = st.data.some(d => d.title === "skipped") ? "skipped" : "done";
      break;
    case "error":
      st.status = "error"; st.t1 = ev.t_ms; st.note = ev.title;
      s.logs.push({ t: ev.t_ms, stage: ev.stage, level: "error", text: ev.title });
      break;
    case "warn":
      st.status = "warn"; st.note = ev.title;
      st.data.push({ title: ev.title, payload: { ...ev.payload, _warn: true }, t: ev.t_ms });
      s.logs.push({ t: ev.t_ms, stage: ev.stage, level: "warn", text: ev.title });
      break;
    case "data":
      st.data.push({ title: ev.title, payload: ev.payload, t: ev.t_ms });
      if (ev.title === "skipped") st.note = ev.payload.reason || "skipped";
      break;
    case "log":
      st.logs++;
      s.logs.push({ t: ev.t_ms, stage: ev.stage, level: ev.payload.level || "info", text: ev.title });
      break;
    case "llm": {
      st.llm++;
      const p = ev.payload;
      const call: LlmCall = {
        t: ev.t_ms, stage: ev.stage, step: p.step || ev.title, model: p.model || "", provider: p.provider || "",
        tier: p.tier || "", prompt_tokens: p.prompt_tokens || 0, completion_tokens: p.completion_tokens || 0,
        cost_usd: p.cost_usd || 0, cached: !!p.cached, fell_back: !!p.fell_back, ms: p.ms || 0, error: p.error,
        system: p.system, prompt: p.prompt, response: p.response,
      };
      s.llm.push(call);
      s.counters.calls++;
      s.counters.tokens += call.prompt_tokens + call.completion_tokens;
      s.counters.cost += call.cost_usd;
      s.counters.cached += call.cached ? 1 : 0;
      s.logs.push({ t: ev.t_ms, stage: ev.stage, level: call.error ? "error" : "llm",
                    text: `${call.step} → ${call.model || call.tier}${call.cached ? " (cache)" : ""}` +
                          (call.error ? ` failed: ${call.error}` : ` · ${call.prompt_tokens}+${call.completion_tokens} tok · ${Math.round(call.ms)} ms`) });
      break;
    }
  }
  return s;
}

export function fold(events: RunEvent[], upTo = Infinity): RunState {
  const s = emptyState();
  for (const ev of events) {
    if (ev.seq > upTo) break;
    apply(s, ev);
  }
  return s;
}

/** The latest data payload a stage emitted under a title. */
export function latest(s: RunState, stageId: string, title?: string): Record<string, any> | null {
  const st = s.stages[stageId];
  if (!st) return null;
  for (let i = st.data.length - 1; i >= 0; i--) if (!title || st.data[i].title === title) return st.data[i].payload;
  return null;
}

export function allData(s: RunState, stageId: string, title?: string): Record<string, any>[] {
  return (s.stages[stageId]?.data || []).filter(d => !title || d.title === title).map(d => d.payload);
}

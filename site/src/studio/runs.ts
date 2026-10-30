/*  History: every run the server has made, kept with every event.
 *
 *  Open one in the console (instantly, or watch it again), compare two side
 *  by side, export the events as JSON or the run as a standalone HTML report,
 *  and see where the LLM budget actually went.
 */
import { esc, human } from "../ui/dom";
import { api, download } from "./api";
import type { StudioCtx } from "./console";
import { allData, fold, latest, PIPELINE, STAGE_LABEL, type RunEvent, type RunState } from "./model";

const ago = (t: number) => {
  const s = Date.now() / 1000 - t;
  return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago` : s < 86400 ? `${Math.floor(s / 3600)} h ago` : new Date(t * 1000).toLocaleDateString();
};

export class RunsView {
  private sel: string[] = [];
  private cache = new Map<string, RunState>();

  constructor(private host: HTMLElement, private ctx: StudioCtx) {
    host.innerHTML = `<div class="rv">
      <div class="rv-bar"><input class="inp" placeholder="search runs…" data-k="q"><select class="sel sm" data-k="kind">
        <option value="">all kinds</option><option>run</option><option>resolve</option><option>eval</option><option>audit</option></select>
        <span class="muted">tick two runs to compare them</span></div>
      <div class="rv-grid"><div class="tscroll" data-lenis-prevent><table class="data mini"><thead><tr><th></th><th>when</th><th>kind</th><th>query</th><th>status</th><th>outcome</th><th>time</th><th>LLM</th></tr></thead><tbody data-k="rows"></tbody></table></div>
        <aside data-k="side" class="rv-side"></aside></div>
      <section data-k="cmp" class="rv-cmp"></section>
      <section class="rv-costs"><h5>LLM spend, by pipeline step</h5><div data-k="costs"></div></section>
    </div>`;
    let t = 0;
    this.q("q").addEventListener("input", () => { clearTimeout(t); t = window.setTimeout(() => this.load(), 200); });
    this.q("kind").addEventListener("change", () => this.load());
    ctx.onHealth(() => this.load());
  }

  private q<T extends HTMLElement = HTMLElement>(k: string): T { return this.host.querySelector(`[data-k="${k}"]`) as T; }

  show(id?: string) { this.load().then(() => { if (id) this.open(id); }); }

  private async load() {
    if (!this.ctx.online) {
      this.q("rows").innerHTML = `<tr><td colspan="8">The history lives on the engine server. Start it with <code>start.bat studio</code>; meanwhile the console's recorded runs can be watched without one.</td></tr>`;
      this.q("costs").innerHTML = "";
      return;
    }
    const rows = await api.runs(this.q<HTMLInputElement>("q").value, this.q<HTMLSelectElement>("kind").value).catch(() => []);
    this.q("rows").innerHTML = rows.map(r => {
      const s = r.summary || {};
      const ms = Object.values(s.stage_ms || {}).reduce((a: number, b: any) => a + Number(b || 0), 0) as number;
      const outcome = r.kind === "eval" ? `${s.n ?? "?"} instances` : r.kind === "audit" ? `${s.high ?? 0} high-risk pairs`
        : [s.relation, s.resolution && human(s.resolution)].filter(Boolean).join(" · ");
      return `<tr data-id="${esc(r.id)}" class="${this.sel.includes(r.id) ? "picked" : ""}">
        <td><input type="checkbox" data-pick="${esc(r.id)}" ${this.sel.includes(r.id) ? "checked" : ""} aria-label="compare"></td>
        <td title="${new Date(r.created * 1000).toLocaleString()}">${ago(r.created)}</td><td>${esc(r.kind)}</td>
        <td class="qcol">${esc(r.query || "—")}</td><td><span class="chip ${r.status === "done" ? "ok" : r.status === "error" ? "bad" : "warn"}">${esc(r.status)}</span></td>
        <td>${esc(outcome || "—")}</td><td>${ms ? (ms / 1000).toFixed(2) + " s" : "—"}</td><td>${s.llm?.calls ?? 0}</td></tr>`;
    }).join("") || `<tr><td colspan="8" class="muted">No runs yet. Run something in the console.</td></tr>`;
    this.q("rows").querySelectorAll<HTMLTableRowElement>("tr[data-id]").forEach(tr => tr.addEventListener("click", e => {
      if ((e.target as HTMLElement).matches("[data-pick]")) return;
      this.open(tr.dataset.id!);
    }));
    this.q("rows").querySelectorAll<HTMLInputElement>("[data-pick]").forEach(cb => cb.addEventListener("change", () => {
      const id = cb.dataset.pick!;
      this.sel = cb.checked ? [...this.sel.filter(x => x !== id), id].slice(-2) : this.sel.filter(x => x !== id);
      this.load();
      this.compareRuns();
    }));
    this.costs();
  }

  private async state(id: string): Promise<{ s: RunState; run: any; events: RunEvent[] }> {
    const r = await api.runEvents(id);
    const s = fold(r.events as RunEvent[]);
    this.cache.set(id, s);
    return { s, run: r.run, events: r.events as RunEvent[] };
  }

  private async open(id: string) {
    const side = this.q("side");
    side.innerHTML = `<p class="muted">loading…</p>`;
    let d: Awaited<ReturnType<RunsView["state"]>>;
    try { d = await this.state(id); } catch (e) { side.innerHTML = `<p class="muted">${esc((e as Error).message)}</p>`; return; }
    const { s, run, events } = d;
    const a = latest(s, "generate", "answer");
    const max = Math.max(1, ...PIPELINE.map(k => s.stages[k]?.ms || 0));
    side.innerHTML = `<h4>${esc(run.query || run.id)}</h4><p class="muted"><code>${esc(run.id)}</code> · ${esc(run.kind)} · ${esc(run.status)}</p>
      ${a ? `<blockquote class="answer">${esc(a.text).replace(/\(per (p\d+)\)/g, '<span class="cite">$1</span>')}</blockquote>` : ""}
      <ul class="hbars">${PIPELINE.filter(k => s.stages[k]).map(k => `<li><span>${STAGE_LABEL[k]}</span><i style="width:${((s.stages[k].ms || 0) / max) * 100}%"></i><b>${Math.round(s.stages[k].ms || 0)} ms</b></li>`).join("")}</ul>
      <p class="row">${run.kind === "run" || run.kind === "resolve" ? `<a class="btn ghost sm" href="#/studio/run/${encodeURIComponent(id)}">Open in console</a>` : ""}
        <button class="btn ghost sm" data-x="json">Export events (JSON)</button>
        <button class="btn ghost sm" data-x="html">Export report (HTML)</button>
        <button class="btn ghost sm danger" data-x="del">Delete</button></p>`;
    side.querySelector('[data-x="json"]')!.addEventListener("click", () => download(`${id}.json`, JSON.stringify({ run, events }, null, 2)));
    side.querySelector('[data-x="html"]')!.addEventListener("click", () => download(`${id}.html`, report(run, s), "text/html"));
    side.querySelector('[data-x="del"]')!.addEventListener("click", async () => {
      if (!confirm("Delete this run and its events?")) return;
      await api.deleteRun(id); side.innerHTML = ""; this.load();
    });
  }

  private async compareRuns() {
    const host = this.q("cmp");
    if (this.sel.length !== 2) { host.innerHTML = ""; return; }
    const [a, b] = await Promise.all(this.sel.map(id => this.cache.get(id) ? Promise.resolve({ s: this.cache.get(id)! }) : this.state(id)));
    const col = (s: RunState) => {
      const ans = latest(s, "generate", "answer"), sc = latest(s, "score", "preservation");
      const ours = sc?.rows?.find((r: any) => r.system === "scopecompose");
      return `<div><h5>${esc(s.request?.query || s.runId)}</h5><p class="muted">${esc(s.request?.mode || "")} · ${esc(s.request?.detector_policy || "")}</p>
        <blockquote class="answer">${esc(ans?.text || "—")}</blockquote>
        <dl class="kv"><dt>relation</dt><dd>${esc(s.summary?.relation || "—")}</dd><dt>resolution</dt><dd>${esc(human(s.summary?.resolution || "—"))}</dd>
        <dt>faithful</dt><dd>${s.summary?.faithful ? "yes" : "no"}</dd><dt>PR vs gold</dt><dd>${ours?.PR !== undefined && ours?.PR !== null ? Math.round(ours.PR * 100) + "%" : "—"}</dd>
        <dt>LLM</dt><dd>${s.counters.calls} calls · ${s.counters.tokens.toLocaleString()} tokens · $${s.counters.cost.toFixed(4)}</dd>
        <dt>total</dt><dd>${(s.t / 1000).toFixed(2)} s</dd></dl></div>`;
    };
    const max = Math.max(1, ...PIPELINE.flatMap(k => [a.s.stages[k]?.ms || 0, b.s.stages[k]?.ms || 0]));
    host.innerHTML = `<h4>Comparison</h4><div class="cmp2">${col(a.s)}${col(b.s)}</div>
      <ul class="cmp-bars">${PIPELINE.filter(k => a.s.stages[k] || b.s.stages[k]).map(k => `<li><span>${STAGE_LABEL[k]}</span>
        <i class="a" style="width:${((a.s.stages[k]?.ms || 0) / max) * 100}%"></i><i class="b" style="width:${((b.s.stages[k]?.ms || 0) / max) * 100}%"></i></li>`).join("")}</ul>`;
  }

  private async costs() {
    const c = await api.costs().catch(() => null);
    if (!c) return;
    this.q("costs").innerHTML = `<p><b>${c.total.calls.toLocaleString()}</b> calls · <b>${c.total.tokens.toLocaleString()}</b> tokens · <b>$${c.total.cost_usd.toFixed(4)}</b> · ${c.total.cached.toLocaleString()} served from cache</p>
      <table class="data mini"><thead><tr><th>step</th><th>model</th><th>calls</th><th>tokens</th><th>cached</th><th>mean latency</th><th>cost</th></tr></thead><tbody>${
        c.by_step.slice(0, 20).map((r: any) => `<tr><td><code>${esc(r.step)}</code></td><td>${esc(r.model)}</td><td>${r.calls}</td><td>${r.tokens.toLocaleString()}</td><td>${r.cached}</td><td>${r.mean_latency_s.toFixed(2)} s</td><td>$${r.cost_usd.toFixed(4)}</td></tr>`).join("")}</tbody></table>`;
  }
}

/** A standalone, self-contained HTML report of one run. */
export function report(run: any, s: RunState): string {
  const a = latest(s, "generate", "answer"), f = latest(s, "verify", "faithfulness"), c = latest(s, "compose", "composition");
  const b = latest(s, "branches", "branches"), sc = latest(s, "score", "preservation"), pol = s.stages.policy?.data.find(d => d.payload._warn)?.payload;
  const scope = allData(s, "scope").filter(d => d.relation);
  const css = `body{font:15px/1.55 system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#14161b}h1{font-size:26px;margin:0 0 4px}
    .m{color:#666;font-size:13px}blockquote{margin:14px 0;padding:14px 18px;background:#f4f2ec;border-left:3px solid #0a8;font-size:17px}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}td,th{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}
    .chip{display:inline-block;padding:1px 8px;border:1px solid #bbb;border-radius:20px;font-size:12px;margin-right:4px}.warn{border-color:#c80;color:#a60}
    h2{font-size:17px;margin:28px 0 8px;border-top:1px solid #ddd;padding-top:16px}code{font-size:12px}`;
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ScopeCompose run ${esc(run.id)}</title><style>${css}</style></head><body>
  <h1>${esc(run.query || run.id)}</h1><p class="m">ScopeCompose run <code>${esc(run.id)}</code> · ${new Date(run.created * 1000).toLocaleString()} ·
  mode ${esc(s.request?.mode || "")} · detector policy ${esc(s.request?.detector_policy || "")} · ${(s.t / 1000).toFixed(2)} s · ${s.counters.calls} LLM calls ($${s.counters.cost.toFixed(4)})</p>
  ${pol ? `<p><span class="chip warn">${esc(pol.badge || pol.policy)}</span> ${esc(s.stages.policy.note)}</p>` : ""}
  <h2>Answer</h2><blockquote>${esc(a?.text || "—")}</blockquote>
  <p>${c ? `<span class="chip">${esc(c.resolution)}</span> ${esc(c.reason)}` : ""} ${f ? `<span class="chip">${f.ok ? "faithful" : "unfaithful"}</span>` : ""}</p>
  ${scope.map(d => `<h2>Scope relation: ${esc(d.relation)}</h2><p>${esc(d.rationale)}</p><p class="m">applicability ${esc(d.set_relation)} · outcomes ${d.outcomes_agree ? "agree" : "disagree"} · evidence ${esc(d.evidence)}${d.gold_relation ? ` · gold ${esc(d.gold_relation)}` : ""}</p>`).join("")}
  ${b ? `<h2>Branches (${esc(b.origin)})</h2><table><tr><th>id</th><th>applies to</th><th>outcome</th><th>passage</th></tr>${b.branches.map((x: any) => `<tr><td>${esc(x.id)}</td><td>${esc(x.default ? "all cases" : x.applies_to)}</td><td>${esc(x.outcome)}</td><td>${esc(x.passage || "")}</td></tr>`).join("")}</table>` : ""}
  ${sc ? `<h2>Against gold</h2><table><tr><th>system</th><th>PR</th><th>SR</th><th>instrument</th></tr>${sc.rows.map((r: any) => `<tr><td>${esc(r.name)}</td><td>${r.PR === null ? "—" : Math.round(r.PR * 100) + "%"}</td><td>${r.SR === null ? "—" : Math.round(r.SR * 100) + "%"}</td><td>${esc(r.instrument)}</td></tr>`).join("")}</table>` : ""}
  <h2>Stages</h2><table><tr><th>stage</th><th>status</th><th>time</th><th>LLM calls</th></tr>${s.order.map(k => s.stages[k]).map(st => `<tr><td>${esc(STAGE_LABEL[st.id] || st.id)}</td><td>${esc(st.status)}${st.note ? ` · ${esc(st.note)}` : ""}</td><td>${st.ms !== null ? Math.round(st.ms) + " ms" : ""}</td><td>${st.llm || ""}</td></tr>`).join("")}</table>
  ${s.llm.length ? `<h2>LLM calls</h2><table><tr><th>step</th><th>model</th><th>tokens</th><th>latency</th><th>cache</th></tr>${s.llm.map(x => `<tr><td><code>${esc(x.step)}</code></td><td>${esc(x.model)}</td><td>${x.prompt_tokens}+${x.completion_tokens}</td><td>${Math.round(x.ms)} ms</td><td>${x.cached ? "yes" : ""}</td></tr>`).join("")}</table>` : ""}
  <p class="m">Generated by ScopeCompose Studio from the run's stored events.</p></body></html>`;
}

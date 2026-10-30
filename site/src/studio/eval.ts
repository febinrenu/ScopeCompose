/*  Batch evaluation, live.
 *
 *  The engine server runs the pipeline over a benchmark split and streams one
 *  item per instance; without a server the browser's verified engines do the
 *  same over the same records (oracle branches, labelled). Either way the
 *  numbers here are recomputed from the items as they arrive -- one
 *  aggregation, whatever the source.
 */
import { esc, human } from "../ui/dom";
import { corpus as browserCorpus } from "../data/corpus";
import { engine } from "../engine/bridge";
import { LiveSource, download } from "./api";
import type { StudioCtx } from "./console";
import { localRun } from "./local";
import { fold } from "./model";

interface Item {
  n: number; id: string; query: string; gold_type: string; gold_relation: string | null; detected: boolean;
  override: any; relation: string | null; resolution: string; faithful: boolean | null;
  scores: Record<string, { PR: number | null; SR: number | null }> | null; ms: number; answer?: string;
}

const RELS = ["refinement", "disjoint", "redundant", "opposed", "none"];
const pct = (x: number | null | undefined) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${(x * 100).toFixed(1)}%`);

export class EvalView {
  private items: Item[] = [];
  private total = 0;
  private stop: (() => void) | null = null;
  private cost = 0;
  private calls = 0;

  constructor(private host: HTMLElement, private ctx: StudioCtx) {
    host.innerHTML = `<div class="ev">
      <div class="ev-ctl">
        <label class="field"><span>Split</span><select class="sel" data-k="split"><option>dev</option><option>test</option><option>train</option><option>all</option></select></label>
        <label class="field"><span>Limit</span><input class="inp" type="number" min="1" max="500" value="25" data-k="limit"></label>
        <label class="field"><span>Mode</span><select class="sel" data-k="mode"><option value="offline">offline</option><option value="live">live</option></select></label>
        <label class="field"><span>Detector miss</span><select class="sel" data-k="policy"><option value="as_run">as-run</option><option value="override">override</option></select></label>
        <label class="tog"><input type="checkbox" data-k="branchy"> only records with gold branches</label>
        <button class="btn run" data-k="go">Evaluate ▸</button>
        <span class="ev-src" data-k="src"></span>
      </div>
      <div class="ev-prog"><i data-k="bar"></i><span data-k="prog"></span></div>
      <div class="ev-tiles" data-k="tiles"></div>
      <div class="ev-grid">
        <section><h5>Relation confusion (gold → predicted)</h5><div data-k="conf"></div></section>
        <section><h5>Detector</h5><div data-k="det"></div><h5>Routing</h5><div data-k="route"></div></section>
      </div>
      <section><h5>Instances <button class="btn ghost sm" data-k="csv">CSV</button> <button class="btn ghost sm" data-k="json">JSON</button></h5>
        <div class="tscroll" data-lenis-prevent><table class="data mini"><thead><tr><th>#</th><th>record</th><th>gold</th><th>detected</th><th>A4 relation</th><th>resolution</th><th>PR (ours)</th><th>PR (selection)</th><th>faithful</th><th>ms</th></tr></thead><tbody data-k="rows"></tbody></table></div></section>
    </div>`;
    this.q("go").addEventListener("click", () => this.start());
    this.q("csv").addEventListener("click", () => this.export("csv"));
    this.q("json").addEventListener("click", () => this.export("json"));
    ctx.onHealth(() => this.syncSrc());
    this.syncSrc();
    this.render();
  }

  private q<T extends HTMLElement = HTMLElement>(k: string): T { return this.host.querySelector(`[data-k="${k}"]`) as T; }

  private syncSrc() {
    const live = this.q<HTMLSelectElement>("mode").querySelector('option[value="live"]') as HTMLOptionElement;
    live.disabled = !(this.ctx.online && this.ctx.health?.live_ready);
    this.q("src").innerHTML = this.ctx.online
      ? `<span class="chip ok">engine server</span> the full pipeline runs on every instance`
      : `<span class="chip warn">browser engines</span> no server: stage 1, scope algebra, composition and scoring run here, on gold branches (oracle)`;
  }

  show() { this.syncSrc(); }

  private async start() {
    if (this.stop) { this.stop(); this.stop = null; this.q("go").textContent = "Evaluate ▸"; return; }
    this.items = []; this.cost = 0; this.calls = 0;
    const params = {
      split: this.q<HTMLSelectElement>("split").value, limit: Number(this.q<HTMLInputElement>("limit").value) || 25,
      mode: this.q<HTMLSelectElement>("mode").value, detector_policy: this.q<HTMLSelectElement>("policy").value,
      branchy_only: this.q<HTMLInputElement>("branchy").checked,
    };
    this.q("go").textContent = "Stop ■";
    const done = () => { this.stop = null; this.q("go").textContent = "Evaluate ▸"; this.render(); };
    if (this.ctx.online) {
      try {
        const src = await LiveSource.start(params, "eval");
        this.stop = () => src.stop();
        src.attach({
          event: ev => {
            if (ev.stage === "eval" && ev.kind === "start") this.total = ev.payload.total || 0;
            if (ev.title === "item") { this.items.push(ev.payload as Item); this.render(); }
            if (ev.title === "metrics") { this.cost = ev.payload.llm?.cost_usd || 0; this.calls = ev.payload.llm?.calls || 0; }
            if (ev.kind === "warn") this.items.push({ n: this.items.length + 1, id: ev.payload.id, query: ev.payload.error, gold_type: "error",
              gold_relation: null, detected: false, override: null, relation: null, resolution: "error", faithful: null, scores: null, ms: 0 });
          },
          done,
        });
      } catch (e) { this.q("prog").textContent = `The server refused: ${(e as Error).message}`; done(); }
      return;
    }
    // browser engines, paced so the dashboard fills visibly
    const rows = browserCorpus().filter(r => (params.split === "all" || r.sp === params.split) && (!params.branchy_only || (r.b || []).length >= 2))
      .sort((a, b) => a.id.localeCompare(b.id)).slice(0, params.limit);
    this.total = rows.length;
    let i = 0, alive = true;
    this.stop = () => { alive = false; };
    const step = () => {
      if (!alive || i >= rows.length) { done(); return; }
      const rec = rows[i++];
      const t = performance.now();
      const st = fold(localRun(rec, { policy: params.detector_policy as "as_run" | "override", reader: {} }));
      const det = (st.stages.detect?.data.find(d => d.title === "verdicts")?.payload.pairs || []).some((p: any) => p.is_conflict);
      const score = st.stages.score?.data.find(d => d.title === "preservation")?.payload;
      this.items.push({
        n: i, id: rec.id, query: rec.q, gold_type: rec.ty, gold_relation: rec.rel, detected: det,
        override: st.summary?.override ?? null, relation: st.summary?.relation ?? null,
        resolution: st.summary?.resolution ?? "pass_through",
        faithful: st.stages.verify?.data.find(d => d.title === "faithfulness")?.payload.ok ?? null,
        scores: score ? Object.fromEntries(score.rows.map((r: any) => [r.system, { PR: r.PR, SR: r.SR }])) : null,
        ms: performance.now() - t,
      });
      this.render();
      setTimeout(step, 60);
    };
    step();
  }

  private metrics() {
    const it = this.items.filter(x => x.gold_type !== "error");
    let tp = 0, fp = 0, fn = 0, tn = 0;
    const routing: Record<string, number> = {};
    const conf: Record<string, Record<string, number>> = {};
    const sums: Record<string, { PR: number; SR: number; n: number; pres: number; tot: number }> = {};
    let faithful = 0;
    for (const x of it) {
      const g = x.gold_type !== "no_conflict";
      if (g && x.detected) tp++; else if (!g && x.detected) fp++; else if (g) fn++; else tn++;
      routing[x.resolution] = (routing[x.resolution] || 0) + 1;
      const gr = x.gold_relation || "none", pr = x.relation || "none";
      (conf[gr] ??= {})[pr] = (conf[gr][pr] || 0) + 1;
      if (x.faithful) faithful++;
      for (const [sys, sc] of Object.entries(x.scores || {})) {
        if (sc.PR === null || sc.PR === undefined) continue;
        const a = (sums[sys] ??= { PR: 0, SR: 0, n: 0, pres: 0, tot: 0 });
        a.PR += sc.PR; a.SR += sc.SR ?? 0; a.n++;
      }
    }
    const P = tp + fp ? tp / (tp + fp) : null, R = tp + fn ? tp / (tp + fn) : null;
    return { tp, fp, fn, tn, P, R, F1: P && R ? (2 * P * R) / (P + R) : null, routing, conf, sums, faithful, n: it.length };
  }

  private render() {
    const m = this.metrics();
    const done = this.items.length, total = this.total || done;
    this.q("bar").style.width = `${total ? (done / total) * 100 : 0}%`;
    this.q("prog").textContent = total ? `${done} / ${total} instances${this.stop ? " · running" : done ? " · done" : ""}` : "Choose a split and press Evaluate.";
    const interval = (k: number, n: number) => {
      if (!n) return "";
      try { const w = engine().wilson(k, n); return ` <span class="muted">[${(w.low * 100).toFixed(0)}–${(w.high * 100).toFixed(0)}%]</span>`; } catch { return ""; }
    };
    const sys = (k: string, label: string, note = "") => {
      const a = m.sums[k];
      return `<div class="tile"><span class="mono">${label}</span><b>${a ? pct(a.PR / a.n) : "—"}</b><span>PR · SR ${a ? pct(a.SR / a.n) : "—"} · n ${a?.n || 0}${note ? `<small>${note}</small>` : ""}</span></div>`;
    };
    this.q("tiles").innerHTML = `
      <div class="tile"><span class="mono">detector precision</span><b>${pct(m.P)}</b><span>${m.tp} TP · ${m.fp} FP${interval(m.tp, m.tp + m.fp)}</span></div>
      <div class="tile"><span class="mono">detector recall</span><b>${pct(m.R)}</b><span>${m.tp} of ${m.tp + m.fn} conflicts${interval(m.tp, m.tp + m.fn)}</span></div>
      <div class="tile"><span class="mono">detector F1</span><b>${pct(m.F1)}</b><span>${m.tn} true negatives</span></div>
      ${sys("scopecompose", "ScopeCompose")}${sys("selection", "Selection")}${sys("concat", "Standard RAG", "gameable: repeats every passage, resolves nothing")}
      <div class="tile"><span class="mono">faithful answers</span><b>${m.n ? pct(m.faithful / m.n) : "—"}</b><span>${m.faithful} of ${m.n}</span></div>
      <div class="tile"><span class="mono">LLM</span><b>${this.calls}</b><span>calls · $${this.cost.toFixed(4)}</span></div>`;
    const cols = RELS;
    this.q("conf").innerHTML = `<table class="conf"><thead><tr><th></th>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${
      RELS.filter(r => m.conf[r]).map(r => `<tr><th>${r}</th>${cols.map(c => {
        const v = m.conf[r]?.[c] || 0;
        const tot = Object.values(m.conf[r] || {}).reduce((a, b) => a + b, 0) || 1;
        return `<td class="${r === c ? "diag" : ""}" style="--a:${(v / tot).toFixed(3)}">${v || ""}</td>`;
      }).join("")}</tr>`).join("") || `<tr><td colspan="6" class="muted">fills in as instances finish</td></tr>`}</tbody></table>`;
    this.q("det").innerHTML = `<table class="conf two"><thead><tr><th></th><th>detected</th><th>missed</th></tr></thead><tbody>
      <tr><th>gold conflict</th><td class="diag" style="--a:${m.tp / Math.max(1, m.tp + m.fn)}">${m.tp}</td><td style="--a:${m.fn / Math.max(1, m.tp + m.fn)}">${m.fn}</td></tr>
      <tr><th>no conflict</th><td style="--a:${m.fp / Math.max(1, m.fp + m.tn)}">${m.fp}</td><td class="diag" style="--a:${m.tn / Math.max(1, m.fp + m.tn)}">${m.tn}</td></tr></tbody></table>`;
    const rmax = Math.max(1, ...Object.values(m.routing));
    this.q("route").innerHTML = `<ul class="hbars">${Object.entries(m.routing).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
      `<li><span>${esc(human(k))}</span><i style="width:${(v / rmax) * 100}%"></i><b>${v}</b></li>`).join("")}</ul>`;
    this.q("rows").innerHTML = this.items.map(x => `<tr class="${x.gold_type === "error" ? "err" : ""}">
      <td>${x.n}</td><td title="${esc(x.query)}"><code>${esc(x.id)}</code></td>
      <td>${esc(x.gold_type)}${x.gold_relation ? ` · ${esc(x.gold_relation)}` : ""}</td>
      <td>${x.detected ? "✓" : x.gold_type !== "no_conflict" ? '<span class="miss">missed</span>' : "—"}${x.override ? ' <span class="chip warn">override</span>' : ""}</td>
      <td>${esc(x.relation || "—")}</td><td>${esc(human(x.resolution))}</td>
      <td>${pct(x.scores?.scopecompose?.PR)}</td><td>${pct(x.scores?.selection?.PR)}</td>
      <td>${x.faithful === null ? "—" : x.faithful ? "✓" : "✗"}</td><td>${Math.round(x.ms)}</td></tr>`).join("");
  }

  private export(kind: "csv" | "json") {
    if (kind === "json") { download("scopecompose-eval.json", JSON.stringify({ metrics: this.metrics(), items: this.items }, null, 2)); return; }
    const head = ["n", "id", "gold_type", "gold_relation", "detected", "override", "relation", "resolution", "PR_scopecompose", "PR_selection", "SR_scopecompose", "faithful", "ms"];
    const rows = this.items.map(x => [x.n, x.id, x.gold_type, x.gold_relation ?? "", x.detected, x.override ? x.override.kind : "", x.relation ?? "",
      x.resolution, x.scores?.scopecompose?.PR ?? "", x.scores?.selection?.PR ?? "", x.scores?.scopecompose?.SR ?? "", x.faithful ?? "", Math.round(x.ms)]);
    download("scopecompose-eval.csv", [head, ...rows].map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n"), "text/csv");
  }
}

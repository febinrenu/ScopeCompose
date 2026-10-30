/*  The live console: ask, watch every stage run, inspect anything.
 *
 *  Source, in order of preference: the engine server (the real pipeline,
 *  streamed), else the browser's verified engines (benchmark records only),
 *  or a recorded run played back. The console says which, always.
 */
import { esc, human } from "../ui/dom";
import { corpus as browserCorpus } from "../data/corpus";
import { LiveSource, ReplaySource, api, type Health } from "./api";
import { Dock } from "./dock";
import { PipelineGraph } from "./graph";
import { renderInspector } from "./inspector";
import { localRun } from "./local";
import { apply, emptyState, latest, PIPELINE, type RunEvent, type RunState } from "./model";

export interface StudioCtx {
  health: Health | null;
  online: boolean;
  recorded: { id: string; title: string; note: string; file: string; mode: string }[];
  onHealth: (fn: () => void) => void;
}

type Src = "benchmark" | "collection" | "passages";

interface Running { stop: () => void }

/** Releases a burst of events at a watchable pace: at least `gap` ms before each
 *  stage starts. Events that arrive slower than that pass straight through, so a
 *  live run (seconds per stage) is shown exactly as it happens. Recorded timings
 *  (`t_ms`) are never touched. */
class Pacer {
  private q: RunEvent[] = [];
  private timer = 0;
  private lastStart = 0;
  private finished: (() => void) | null = null;
  constructor(private out: (ev: RunEvent) => void, private gap = 130) {}

  push = (ev: RunEvent) => { this.q.push(ev); if (!this.timer) this.drain(); };

  finish(cb: () => void) { this.finished = cb; if (!this.q.length && !this.timer) cb(); }

  stop() { clearTimeout(this.timer); this.q = []; }

  private drain = () => {
    this.timer = 0;
    while (this.q.length) {
      const ev = this.q[0];
      if (ev.kind === "start" && ev.stage !== "run") {
        const wait = this.lastStart + this.gap - performance.now();
        if (wait > 0) { this.timer = window.setTimeout(this.drain, wait); return; }
        this.lastStart = performance.now();
      }
      this.q.shift();
      this.out(ev);
    }
    if (this.finished) { const f = this.finished; this.finished = null; f(); }
  };
}

export class Console {
  private s: RunState = emptyState();
  private graph: PipelineGraph;
  private dock: Dock;
  private ins: HTMLElement;
  private pinned: string | null = null;
  private running: Running | null = null;
  private raf = 0;
  private tick = 0;
  private startedAt = 0;
  private source: Src = "benchmark";
  private mode: "offline" | "live" = "offline";
  private policy: "override" | "as_run" = "override";
  private generation: "template" | "llm" = "template";
  private records: { id: string; query: string; gold_branches: number; gold_type: string; gold_relation: string | null; split: string }[] = [];
  private reader: Record<string, unknown> = {};

  constructor(private host: HTMLElement, private ctx: StudioCtx) {
    host.innerHTML = `
    <div class="sc">
      <div class="sc-top">
        <span class="srv" data-k="srv"><i></i><b>checking the engine server…</b></span>
        <div class="seg sm" role="radiogroup" aria-label="Mode" data-k="mode">
          <button role="radio" data-v="offline" aria-checked="true">Offline</button><button role="radio" data-v="live" aria-checked="false">Live</button></div>
        <div class="seg sm" role="radiogroup" aria-label="When the detector misses" data-k="policy">
          <button role="radio" data-v="override" aria-checked="true">Override</button><button role="radio" data-v="as_run" aria-checked="false">As-run</button></div>
        <div class="seg sm" role="radiogroup" aria-label="Generation" data-k="gen">
          <button role="radio" data-v="template" aria-checked="true">Template</button><button role="radio" data-v="llm" aria-checked="false">LLM</button></div>
        <label class="tog" title="Also run the long-context LLM baseline (live)"><input type="checkbox" data-k="lc"> long-context baseline</label>
        <span class="sc-counters" data-k="counters"></span>
        <button class="btn run" data-k="run" title="Run (Ctrl+Enter)">Run ▸</button>
      </div>
      <div class="sc-main">
        <aside class="sc-in" data-lenis-prevent>
          <div class="seg sm wide" role="radiogroup" aria-label="Source" data-k="src">
            <button role="radio" data-v="benchmark" aria-checked="true">Benchmark</button>
            <button role="radio" data-v="collection" aria-checked="false">My documents</button>
            <button role="radio" data-v="passages" aria-checked="false">Paste</button></div>
          <div data-k="srcbody"></div>
          <label class="field"><span>Question</span><textarea class="inp" rows="2" data-k="q" placeholder="Ask about these documents…"></textarea></label>
          <div data-k="reader" class="sc-reader"></div>
          <div data-k="recorded" class="sc-rec"></div>
        </aside>
        <section class="sc-center">
          <div class="sc-graph" data-k="graph"></div>
          <div class="sc-result" data-k="result"></div>
          <ol class="sc-feed" data-k="feed" aria-hidden="true"></ol>
        </section>
        <aside class="sc-ins" data-k="ins" aria-live="polite" data-lenis-prevent></aside>
      </div>
      <div class="sc-dock" data-k="dock"></div>
    </div>`;

    this.graph = new PipelineGraph(id => { this.pinned = id; this.renderInspector(); });
    this.k("graph").appendChild(this.graph.el);
    this.dock = new Dock(this.k("dock"));
    this.ins = this.k("ins");

    this.seg("mode", v => { this.mode = v as "offline" | "live"; this.syncTop(); });
    this.seg("policy", v => { this.policy = v as "override" | "as_run"; });
    this.seg("gen", v => { this.generation = v as "template" | "llm"; });
    this.seg("src", v => { this.source = v as Src; this.renderSource(); });
    this.k("run").addEventListener("click", () => this.run());
    host.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); this.run(); } });

    ctx.onHealth(() => { this.syncTop(); this.loadRecords(); });
    this.syncTop();
    this.loadRecords();
    this.renderRecorded();
    this.paint();
  }

  private k<T extends HTMLElement = HTMLElement>(key: string): T {
    return this.host.querySelector(`[data-k="${key}"]`) as T;
  }

  private seg(key: string, on: (v: string) => void) {
    const g = this.k(key);
    g.querySelectorAll<HTMLButtonElement>("button").forEach(b => b.addEventListener("click", () => {
      if (b.disabled) return;
      g.querySelectorAll("button").forEach(x => x.setAttribute("aria-checked", String(x === b)));
      on(b.dataset.v!);
    }));
  }

  private setSeg(key: string, v: string) {
    this.k(key).querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String((b as HTMLElement).dataset.v === v)));
  }

  // ---- server status ---------------------------------------------------------------

  private syncTop() {
    const h = this.ctx.health, online = this.ctx.online;
    const srv = this.k("srv");
    srv.dataset.state = online ? (h?.live_ready ? "live" : "on") : "off";
    (srv.querySelector("b") as HTMLElement).textContent = online
      ? `engine server online · ${h?.live_ready ? "live ready" : "offline only"}`
      : "no engine server · browser engines";
    srv.title = online ? `ScopeCompose API ${h?.version} · LLM ${h?.live.llm_configured ? "configured" : "not configured"} · torch ${h?.live.torch ? "yes" : "no"}`
      : "Start it with `start.bat studio` (or python -m service). Until then, runs use the browser's verified engines on benchmark records.";
    const live = this.k("mode").querySelector<HTMLButtonElement>('[data-v="live"]')!;
    live.disabled = !(online && h?.live_ready);
    if (live.disabled && this.mode === "live") { this.mode = "offline"; this.setSeg("mode", "offline"); }
    const llm = this.k("gen").querySelector<HTMLButtonElement>('[data-v="llm"]')!;
    llm.disabled = this.mode !== "live";
    if (llm.disabled && this.generation === "llm") { this.generation = "template"; this.setSeg("gen", "template"); }
    this.k<HTMLInputElement>("lc").disabled = this.mode !== "live";
    for (const v of ["collection", "passages"]) {
      const b = this.k("src").querySelector<HTMLButtonElement>(`[data-v="${v}"]`)!;
      b.disabled = !online;
      b.title = online ? "" : "needs the engine server";
    }
    if (!online && this.source !== "benchmark") { this.source = "benchmark"; this.setSeg("src", "benchmark"); this.renderSource(); }
  }

  // ---- inputs ------------------------------------------------------------------------

  private async loadRecords() {
    try {
      this.records = this.ctx.online ? await api.corpus()
        : browserCorpus().map(r => ({ id: r.id, query: r.q, gold_branches: (r.b || []).length, gold_type: r.ty, gold_relation: r.rel, split: r.sp }));
    } catch {
      this.records = browserCorpus().map(r => ({ id: r.id, query: r.q, gold_branches: (r.b || []).length, gold_type: r.ty, gold_relation: r.rel, split: r.sp }));
    }
    if (this.source === "benchmark") this.renderSource();
  }

  private renderSource() {
    const body = this.k("srcbody");
    if (this.source === "benchmark") {
      const prev = body.querySelector<HTMLSelectElement>("[data-k=rec]")?.value || "wp1_fin_imp_003";
      const only = body.querySelector<HTMLInputElement>("[data-k=branchy]")?.checked ?? true;
      const rows = this.records.filter(r => !only || r.gold_branches >= 2);
      body.innerHTML = `<label class="field"><span>Benchmark record (${rows.length})</span>
        <select class="sel" data-k="rec">${rows.map(r => `<option value="${esc(r.id)}" ${r.id === prev ? "selected" : ""}>${esc(r.id)} — ${esc(r.query.slice(0, 48))}${r.query.length > 48 ? "…" : ""}</option>`).join("")}</select></label>
        <label class="tog"><input type="checkbox" data-k="branchy" ${only ? "checked" : ""}> only records with gold branches</label>
        <p class="rec-meta" data-k="recmeta"></p>`;
      const sel = body.querySelector<HTMLSelectElement>("[data-k=rec]")!;
      const meta = () => {
        const r = this.records.find(x => x.id === sel.value);
        this.k<HTMLTextAreaElement>("q").value = r?.query || "";
        body.querySelector("[data-k=recmeta]")!.innerHTML = r ? `gold: <span class="chip ${r.gold_type === "conditional" ? "warn" : ""}">${esc(r.gold_type)}</span> ${r.gold_relation ? `<span class="chip">${esc(r.gold_relation)}</span>` : ""} · ${r.gold_branches} branches · ${esc(r.split)}` : "";
      };
      sel.addEventListener("change", meta);
      body.querySelector("[data-k=branchy]")!.addEventListener("change", () => this.renderSource());
      meta();
    } else if (this.source === "collection") {
      body.innerHTML = `<label class="field"><span>Collection</span><select class="sel" data-k="coll"><option>loading…</option></select></label>
        <label class="field"><span>Passages to retrieve</span><input class="inp" type="number" min="2" max="10" value="5" data-k="kk"></label>
        <p class="rec-meta"><a href="#/studio/docs">Add or manage documents →</a></p>`;
      api.collections().then(cs => {
        const sel = body.querySelector<HTMLSelectElement>("[data-k=coll]")!;
        sel.innerHTML = cs.length ? cs.map(c => `<option value="${esc(c.id)}">${esc(c.name)} · ${c.passages} passages</option>`).join("")
          : `<option value="">no collections yet: add documents first</option>`;
      }).catch(() => {});
      this.k<HTMLTextAreaElement>("q").value = "";
    } else {
      body.innerHTML = `<div data-k="pastes" class="pastes">${[0, 1].map(i => this.pasteRow(i)).join("")}</div>
        <button class="btn ghost sm" data-k="addp">+ passage</button>`;
      body.querySelector("[data-k=addp]")!.addEventListener("click", () => {
        const host = body.querySelector("[data-k=pastes]")!;
        host.insertAdjacentHTML("beforeend", this.pasteRow(host.children.length));
      });
      this.k<HTMLTextAreaElement>("q").value = "";
    }
  }

  private pasteRow(i: number) {
    const ex = [
      ["A 3% fee applies to all international card payments.", "official_policy"],
      ["Premium cardholders pay no fee on international card payments.", "product_terms"],
    ][i] || ["", "unknown"];
    return `<div class="paste"><span class="pid">p${i}</span><textarea class="inp" rows="3" placeholder="passage text">${esc(ex[0])}</textarea>
      <select class="sel sm">${["official_policy", "government_guidance", "product_terms", "faq", "amendment", "third_party", "unknown"].map(t => `<option ${t === ex[1] ? "selected" : ""}>${t}</option>`).join("")}</select></div>`;
  }

  private body(): Record<string, unknown> | null {
    const q = this.k<HTMLTextAreaElement>("q").value.trim();
    const base = { mode: this.mode, detector_policy: this.policy, generation: this.generation,
                   long_context_baseline: this.k<HTMLInputElement>("lc").checked, reader: this.reader };
    if (this.source === "benchmark") {
      const id = this.k("srcbody").querySelector<HTMLSelectElement>("[data-k=rec]")?.value;
      return id ? { ...base, source: "benchmark", record_id: id, query: q } : null;
    }
    if (this.source === "collection") {
      const cid = this.k("srcbody").querySelector<HTMLSelectElement>("[data-k=coll]")?.value;
      const k = Number(this.k("srcbody").querySelector<HTMLInputElement>("[data-k=kk]")?.value || 5);
      return cid && q ? { ...base, source: "collection", collection_id: cid, query: q, k } : null;
    }
    const passages = [...this.k("srcbody").querySelectorAll(".paste")].map(r => ({
      text: (r.querySelector("textarea") as HTMLTextAreaElement).value.trim(),
      source_type: (r.querySelector("select") as HTMLSelectElement).value,
    })).filter(p => p.text);
    return q && passages.length >= 2 ? { ...base, source: "passages", passages, query: q } : null;
  }

  // ---- running ------------------------------------------------------------------------

  private reset() {
    this.running?.stop();
    this.running = null;
    this.s = emptyState();
    this.graph.reset();
    this.pinned = null;
    this.startedAt = performance.now();
    clearInterval(this.tick);
    this.tick = window.setInterval(() => this.counters(), 100);
  }

  private onEvent = (ev: RunEvent) => {
    apply(this.s, ev);
    if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.paint(); });
  };

  private onDone = () => {
    clearInterval(this.tick);
    this.running = null;
    this.k("run").textContent = "Run ▸";
    this.paint();
    this.renderReader();
  };

  async run() {
    if (this.running) { this.running.stop(); this.onDone(); return; }
    const body = this.body();
    if (!body) { this.flash(this.source === "passages" ? "Give a question and at least two passages." : "Choose a source and ask a question."); return; }
    this.reset();
    this.k("run").textContent = "Stop ■";
    if (this.ctx.online) {
      try {
        const src = await LiveSource.start(body);
        const pacer = new Pacer(this.onEvent);
        this.running = { stop: () => { src.stop(); pacer.stop(); } };
        src.attach({ event: pacer.push, done: () => pacer.finish(this.onDone) });
        history.replaceState(null, "", `#/studio/run/${encodeURIComponent(src.id)}`);
      } catch (e) {
        this.flash(`The server refused the run: ${(e as Error).message}`);
        this.onDone();
      }
      return;
    }
    const rec = browserCorpus().find(r => r.id === body.record_id);
    if (!rec) { this.flash("Without the engine server, only benchmark records can run."); this.onDone(); return; }
    const events = localRun(rec, { policy: this.policy, reader: this.reader });
    const src = new ReplaySource(events, 1, 700, 160);
    this.running = src;
    src.attach({ event: this.onEvent, done: this.onDone });
  }

  /** Show a stored run: animated (watch it again) or all at once. */
  play(events: RunEvent[], animate = true) {
    this.reset();
    if (!animate) { for (const ev of events) apply(this.s, ev); this.onDone(); this.graph.update(this.s, false); return; }
    this.k("run").textContent = "Stop ■";
    const src = new ReplaySource(events, 1.4, 900, 120);
    this.running = src;
    src.attach({ event: this.onEvent, done: this.onDone });
  }

  async openRun(id: string, animate = false) {
    try {
      const r = await api.runEvents(id);
      this.play(r.events as RunEvent[], animate);
    } catch (e) {
      this.flash(`Could not load run ${id}: ${(e as Error).message}`);
    }
  }

  private flash(msg: string) {
    const r = this.k("result");
    r.innerHTML = `<p class="sc-flash">${esc(msg)}</p>`;
  }

  // ---- painting ----------------------------------------------------------------------

  private counters() {
    const s = this.s;
    const el = this.k("counters");
    const elapsed = this.running ? (performance.now() - this.startedAt) / 1000 : s.t / 1000;
    el.innerHTML = `<span><b>${elapsed.toFixed(1)}</b> s</span><span><b>${s.counters.calls}</b> LLM calls</span>
      <span><b>${s.counters.tokens.toLocaleString()}</b> tokens</span><span><b>$${s.counters.cost.toFixed(4)}</b></span>
      <span class="st st-${s.status}">${esc(s.status === "idle" ? "ready" : s.status)}</span>`;
  }

  private current(): string {
    if (this.pinned) return this.pinned;
    const running = PIPELINE.filter(id => this.s.stages[id]?.status === "running");
    if (running.length) return running[running.length - 1];
    const seen = PIPELINE.filter(id => this.s.stages[id]);
    if (this.s.status === "done") return this.s.stages.generate ? "generate" : seen[seen.length - 1] || "setup";
    return seen[seen.length - 1] || "setup";
  }

  private renderInspector() {
    const id = this.current();
    renderInspector(this.ins, this.s, id);
    if (!this.pinned) this.graph.el.querySelectorAll(".pg-node").forEach((g, i) => g.classList.toggle("sel", PIPELINE[i] === id));
  }

  private paint() {
    this.graph.update(this.s);
    this.renderInspector();
    this.dock.render(this.s);
    this.counters();
    this.renderResult();
    this.renderFeed();
  }

  /** The last few things that happened, large, as they happen. */
  private renderFeed() {
    const lines = this.s.logs.slice(-7);
    this.k("feed").innerHTML = lines.map((l, i) => `<li class="lv-${esc(l.level)} ${i === lines.length - 1 && this.running ? "new" : ""}" style="--o:${(0.35 + (0.65 * (i + 1)) / lines.length).toFixed(2)}">
      <time>${(l.t / 1000).toFixed(2)}s</time><b>${esc(l.stage)}</b><span>${esc(l.text)}</span></li>`).join("");
  }

  private renderResult() {
    const s = this.s;
    const r = this.k("result");
    if (s.status === "idle") {
      r.innerHTML = `<p class="sc-hint">Pick a source, ask, and press <b>Run</b>. Every stage reports as it runs; click any node to see exactly what it computed.</p>`;
      return;
    }
    const a = latest(s, "generate", "answer");
    const comp = latest(s, "compose", "composition");
    const f = latest(s, "verify", "faithfulness");
    const pol = s.stages.policy?.data.find(d => d.payload._warn)?.payload;
    const rel = s.summary?.relation ?? latest(s, "scope")?.relation;
    const mode = s.request?.mode || "";
    r.innerHTML = `<div class="res">
      <div class="res-a">${a ? `<blockquote>${esc(a.text).replace(/\(per (p\d+)\)/g, '<span class="cite">$1</span>')}</blockquote>` : `<p class="muted">${s.status === "error" ? esc(s.error) : "Working…"}</p>`}</div>
      <div class="res-c">
        ${mode ? `<span class="chip ${mode === "live" ? "ok" : ""}">${esc(mode)}</span>` : ""}
        ${rel ? `<span class="chip">${esc(rel)}</span>` : ""}
        ${comp ? `<span class="chip solid">${esc(human(comp.resolution))}</span>` : ""}
        ${f ? `<span class="chip ${f.ok ? "ok" : "bad"}">${f.ok ? "faithful" : "unfaithful"}</span>` : ""}
        ${pol && pol.policy === "override" ? `<span class="chip warn">${esc(pol.badge || "override")}</span>` : ""}
        ${pol && pol.policy === "as_run" ? `<span class="chip warn">detector missed (as-run)</span>` : ""}
        ${s.status === "done" && s.runId && this.ctx.online ? `<a class="chip link" href="#/studio/runs/${encodeURIComponent(s.runId)}">report ↗</a>` : ""}
      </div></div>`;
  }

  private renderReader() {
    const host = this.k("reader");
    const dims = latest(this.s, "branches", "branches")?.dims || latest(this.s, "reader")?.dims || [];
    if (!dims.length) { host.innerHTML = ""; return; }
    host.innerHTML = `<h5>Who is reading?</h5>${dims.map((d: any) => d.values && d.values.length
      ? `<label class="field"><span>${esc(human(d.name))}</span><select class="sel" data-dim="${esc(d.name)}"><option value="">— none of these —</option>${d.values.map((v: string) => `<option ${String(this.reader[d.name]) === v ? "selected" : ""}>${esc(v)}</option>`).join("")}</select></label>`
      : `<label class="field"><span>${esc(human(d.name))}</span><input class="inp" type="number" data-dim="${esc(d.name)}" value="${esc(this.reader[d.name] ?? "")}" placeholder="${d.min !== undefined ? "≥ " + d.min : ""}${d.max !== undefined ? " ≤ " + d.max : ""}"></label>`).join("")}
      <button class="btn ghost sm" data-k="rerun">Run again for this reader</button>`;
    host.querySelectorAll<HTMLInputElement | HTMLSelectElement>("[data-dim]").forEach(el => el.addEventListener("change", () => {
      const v = el.value;
      if (v === "") delete this.reader[el.dataset.dim!]; else this.reader[el.dataset.dim!] = el.tagName === "INPUT" ? Number(v) : v;
    }));
    host.querySelector("[data-k=rerun]")!.addEventListener("click", () => this.run());
  }

  private renderRecorded() {
    const host = this.k("recorded");
    if (!this.ctx.recorded.length) { host.innerHTML = ""; return; }
    host.innerHTML = `<h5>Recorded runs</h5><p class="muted">Real runs of the engine server, stored with every event, to watch without a server.</p>
      <ul>${this.ctx.recorded.map(r => `<li><button class="linkish" data-rec="${esc(r.file)}"><span class="chip ${r.mode === "live" ? "ok" : ""}">${esc(r.mode)}</span> ${esc(r.title)}</button><span class="muted">${esc(r.note)}</span></li>`).join("")}</ul>`;
    host.querySelectorAll<HTMLButtonElement>("[data-rec]").forEach(b => b.addEventListener("click", async () => {
      try {
        const data = await (await fetch(`./data/runs/${b.dataset.rec}`)).json();
        this.play(data.events as RunEvent[], true);
      } catch { this.flash("Could not load that recording."); }
    }));
  }

  refreshRecorded() { this.renderRecorded(); }

  /** Arrive with a collection chosen (from "Ask a question" on the documents page). */
  useCollection(id: string) {
    if (!this.ctx.online) return;
    this.source = "collection";
    this.setSeg("src", "collection");
    this.renderSource();
    const pick = () => {
      const sel = this.k("srcbody").querySelector<HTMLSelectElement>("[data-k=coll]");
      if (sel && [...sel.options].some(o => o.value === id)) sel.value = id;
      else setTimeout(pick, 150);
    };
    pick();
    this.k<HTMLTextAreaElement>("q").focus();
  }
}

/*  The dock: the run's instruments, in tabs.
 *
 *    Log        every line, coloured by level, filterable, following the tail
 *    Timeline   a waterfall of stages and the LLM calls inside them
 *    LLM        each model call: model, tokens, latency, cache, prompt, reply
 *    Contract   the QueryRecord as it is filled in, changed lines highlighted
 *    Events     the raw event stream, as the API sends it
 */
import { esc } from "../ui/dom";
import { STAGE_LABEL, type RunState } from "./model";

export type DockTab = "log" | "timeline" | "llm" | "contract" | "events";

const fmt = (ms: number) => (ms < 1000 ? `${ms.toFixed(ms < 10 ? 1 : 0)} ms` : `${(ms / 1000).toFixed(2)} s`);

export class Dock {
  tab: DockTab = "log";
  filter = "";
  follow = true;
  contractIdx = -1;
  private body: HTMLElement;
  private tabs: HTMLElement;
  private last: RunState | null = null;

  constructor(readonly el: HTMLElement) {
    el.innerHTML = `<div class="dock-bar" role="tablist" aria-label="Run instruments">
        ${(["log", "timeline", "llm", "contract", "events"] as DockTab[]).map(t =>
          `<button role="tab" data-tab="${t}" aria-selected="${t === this.tab}">${t === "llm" ? "LLM calls" : t[0].toUpperCase() + t.slice(1)}<span class="n"></span></button>`).join("")}
        <span class="dock-tools"><input class="dock-filter" placeholder="filter…" aria-label="Filter the log"><label><input type="checkbox" class="dock-follow" checked> follow</label></span>
      </div><div class="dock-body" tabindex="0" data-lenis-prevent></div>`;
    this.body = el.querySelector(".dock-body") as HTMLElement;
    this.tabs = el.querySelector(".dock-bar") as HTMLElement;
    this.tabs.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach(b => b.addEventListener("click", () => {
      this.tab = b.dataset.tab as DockTab;
      this.tabs.querySelectorAll("[data-tab]").forEach(x => x.setAttribute("aria-selected", String(x === b)));
      if (this.last) this.render(this.last, true);
    }));
    (el.querySelector(".dock-filter") as HTMLInputElement).addEventListener("input", e => {
      this.filter = (e.target as HTMLInputElement).value.toLowerCase();
      if (this.last) this.render(this.last, true);
    });
    (el.querySelector(".dock-follow") as HTMLInputElement).addEventListener("change", e => {
      this.follow = (e.target as HTMLInputElement).checked;
    });
    this.body.addEventListener("click", e => {
      const b = (e.target as HTMLElement).closest<HTMLElement>("[data-snap]");
      if (b && this.last) { this.contractIdx = Number(b.dataset.snap); this.render(this.last, true); }
    });
  }

  render(s: RunState, force = false) {
    this.last = s;
    const counts: Record<DockTab, number> = { log: s.logs.length, timeline: s.order.length, llm: s.llm.length,
                                              contract: s.contract.length, events: s.events.length };
    this.tabs.querySelectorAll<HTMLElement>("[data-tab]").forEach(b => {
      (b.querySelector(".n") as HTMLElement).textContent = counts[b.dataset.tab as DockTab] ? String(counts[b.dataset.tab as DockTab]) : "";
    });
    const atBottom = this.body.scrollTop + this.body.clientHeight >= this.body.scrollHeight - 40;
    let html = "";
    switch (this.tab) {
      case "log": html = this.log(s); break;
      case "timeline": html = this.timeline(s); break;
      case "llm": html = this.llmCalls(s); break;
      case "contract": html = this.contract(s); break;
      case "events": html = this.events(s); break;
    }
    // keep open <details> open across re-renders of the LLM table
    const open = new Set([...this.body.querySelectorAll<HTMLDetailsElement>("details[open]")].map(d => d.dataset.k));
    this.body.innerHTML = html;
    open.forEach(k => { const d = this.body.querySelector<HTMLDetailsElement>(`details[data-k="${k}"]`); if (d) d.open = true; });
    if ((this.follow && (atBottom || force)) && (this.tab === "log" || this.tab === "events")) this.body.scrollTop = this.body.scrollHeight;
  }

  private log(s: RunState) {
    const lines = s.logs.filter(l => !this.filter || (l.text + l.stage).toLowerCase().includes(this.filter));
    if (!lines.length) return `<p class="dock-empty">No log lines yet.</p>`;
    return `<ol class="term">${lines.map(l => `<li class="lv-${esc(l.level)}"><time>${(l.t / 1000).toFixed(3)}</time><b>${esc(STAGE_LABEL[l.stage] || l.stage)}</b><span>${esc(l.text)}</span></li>`).join("")}</ol>`;
  }

  private timeline(s: RunState) {
    const stages = s.order.map(id => s.stages[id]).filter(st => st.t0 !== null);
    if (!stages.length) return `<p class="dock-empty">The waterfall fills in as stages run.</p>`;
    const end = Math.max(s.t, ...stages.map(st => st.t1 ?? s.t), 1);
    const x = (t: number) => `${((t / end) * 100).toFixed(3)}%`;
    const rows = stages.map(st => {
      const t0 = st.t0 ?? 0, t1 = st.t1 ?? s.t;
      const calls = s.llm.filter(c => c.stage === st.id);
      return `<li class="wf-${st.status}"><span class="wf-l">${esc(STAGE_LABEL[st.id] || st.id)}</span>
        <span class="wf-track"><i style="left:${x(t0)};width:max(2px, ${x(t1 - t0)})"></i>
        ${calls.map(c => `<b class="wf-call ${c.cached ? "cached" : ""}" title="${esc(c.step)} · ${esc(c.model)} · ${fmt(c.ms)}" style="left:${x(Math.max(0, c.t - c.ms))};width:max(2px, ${x(c.ms)})"></b>`).join("")}</span>
        <span class="wf-ms">${st.ms !== null ? fmt(st.ms) : "…"}</span></li>`;
    }).join("");
    return `<ol class="wf">${rows}</ol><p class="dock-note">Total ${fmt(end)}. Bars: stage time. Ticks: LLM calls inside it (hollow = served from cache).</p>`;
  }

  private llmCalls(s: RunState) {
    if (!s.llm.length) return `<p class="dock-empty">No LLM calls in this run. Offline mode never calls a model; live mode calls one for stage 2, descriptors, extraction and (optionally) generation.</p>`;
    return `<table class="data mini llm-t"><thead><tr><th>#</th><th>stage</th><th>step</th><th>model</th><th>tokens in / out</th><th>latency</th><th>cost</th><th></th></tr></thead><tbody>${
      s.llm.map((c, i) => `<tr class="${c.error ? "err" : ""}"><td>${i + 1}</td><td>${esc(STAGE_LABEL[c.stage] || c.stage)}</td><td><code>${esc(c.step)}</code></td>
        <td>${esc(c.model || c.tier)}</td><td>${c.prompt_tokens} / ${c.completion_tokens}</td><td>${fmt(c.ms)}</td>
        <td>$${c.cost_usd.toFixed(5)}</td><td>${c.cached ? '<span class="chip ok">cache</span>' : ""}${c.fell_back ? '<span class="chip warn">fallback</span>' : ""}${c.error ? '<span class="chip bad">error</span>' : ""}</td></tr>
        <tr class="llm-x"><td></td><td colspan="7"><details data-k="${i}"><summary>prompt and response</summary>
          ${c.system ? `<h6>system</h6><pre>${esc(c.system)}</pre>` : ""}<h6>prompt</h6><pre>${esc(c.prompt || "")}</pre>
          <h6>${c.error ? "error" : "response"}</h6><pre>${esc(c.error || c.response || "")}</pre></details></td></tr>`).join("")}</tbody></table>`;
  }

  private contract(s: RunState) {
    if (!s.contract.length) return `<p class="dock-empty">The QueryRecord (the frozen contract both halves of the pipeline agree on) appears here as each stage fills it in.</p>`;
    const i = this.contractIdx < 0 || this.contractIdx >= s.contract.length ? s.contract.length - 1 : this.contractIdx;
    const cur = JSON.stringify(s.contract[i].record, null, 2).split("\n");
    const prev = i > 0 ? new Set(JSON.stringify(s.contract[i - 1].record, null, 2).split("\n")) : new Set<string>();
    return `<div class="snaps">${s.contract.map((c, k) => `<button data-snap="${k}" class="${k === i ? "on" : ""}">${esc(c.title)}</button>`).join("")}</div>
      <pre class="json">${cur.map(l => `<span class="${i > 0 && !prev.has(l) ? "chg" : ""}">${esc(l)}</span>`).join("\n")}</pre>`;
  }

  private events(s: RunState) {
    const evs = s.events.filter(e => !this.filter || JSON.stringify(e).toLowerCase().includes(this.filter)).slice(-400);
    return `<ol class="term ev">${evs.map(e => `<li class="k-${e.kind}"><time>${(e.t_ms / 1000).toFixed(3)}</time><b>#${e.seq} ${esc(e.stage)}.${esc(e.kind)}</b><span>${esc(e.title)} <code>${esc(JSON.stringify(e.payload).slice(0, 220))}</code></span></li>`).join("")}</ol>`;
  }
}

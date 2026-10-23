/*  Act 08: the research, as an editorial paper surface.
 *  Counts are read from the corpus; quoted findings carry their source.
 */
import { corpus, countBy, byId, PROTAGONIST_ID } from "../data/corpus";
import { CONTRACT_SCHEMA, CONTRACT_SOURCE, CONTRACT_VERSION, KAPPA, TIER1_PILOT, type Fact } from "../data/facts";
import type { CorpusRecord } from "../engine/types";
import { $, $$, esc, human } from "../ui/dom";

/* ---------------------------------------------------------------- metrics */

/** Wording from metrics/preservation.py's module docstring. */
const METRICS = [
  { abbr: "PR", name: "Preservation Rate", formula: "preserved branches ÷ gold branches",
    body: "The headline metric. A judged branch is exactly one of preserved, suppressed or distorted.", src: "metrics/preservation.py:9-10" },
  { abbr: "SR", name: "Suppression Rate", formula: "suppressed branches ÷ gold branches",
    body: "The critical safety metric: a valid, separately-scoped branch of the truth silently dropped — which answer-correctness cannot see.", src: "metrics/preservation.py:12-18" },
  { abbr: "HCR", name: "Hallucinated-Condition Rate", formula: "conditions asserted with no passage support ÷ instances",
    body: "Guards against fabricating content. Implemented as instances with a predicted branch citing a passage that does not exist.", src: "metrics/preservation.py:20-23, 363-373" },
  { abbr: "SCR", name: "Spurious-Condition Rate", formula: "conditional structure imposed on instances that have none ÷ no-conflict instances",
    body: "Guards the opposite failure: a caveat-happy model inventing structure to game PR. This is why distractor instances are mandatory.", src: "metrics/preservation.py:25-29, 376-395" },
  { abbr: "F1", name: "Detection", formula: "pair-level precision and recall, F1 = 2PR / (P + R)",
    body: "With the headline confusion cell reported on its own: conditional read as factual, and factual read as conditional.", src: "metrics/classification.py:36-58, 233-261" },
  { abbr: "OI", name: "Order invariance", formula: "1 − violations ÷ pairs compared",
    body: "Relation, roles, set relation, agreement and confidence must survive every permutation of the passages — and non-positional roles prove the pass is not vacuous.", src: "scope/order_invariance.py:58-77" },
];

function renderMetrics() {
  const host = $("#metrics");
  if (!host) return;
  host.innerHTML = METRICS.map(m => `
    <div class="metric">
      <div class="abbr">${m.abbr}</div>
      <div class="name">${esc(m.name)}</div>
      <p class="formula">${esc(m.formula)}</p>
      <p>${esc(m.body)}</p>
      <p class="src muted">${esc(m.src)}</p>
    </div>`).join("");
}

/* --------------------------------------------------------------- contract */

function contractExample(r: CorpusRecord) {
  return {
    contract_version: CONTRACT_VERSION,
    query_id: r.id,
    query: r.q,
    domain: r.dom,
    construction: r.con,
    separation: r.sep,
    passages: r.p.map(p => ({ id: p.id, text: p.t, source_type: p.st ?? null, date: p.dt ?? null, source_url: p.u ?? null, document_id: p.d ?? null })),
    conflict_pairs: r.p.length >= 2 ? [{
      doc_i: r.p[0].id, doc_j: r.p[1].id, is_conflict: r.ty !== "no_conflict", type: r.ty,
      scope_relation: r.rel, confidence: null, decided_by: "gold annotation",
    }] : [],
    multi_exception_flags: { nested: false, crossed: false },
  };
}

function jsonHtml(v: unknown, indent = 0): string {
  const pad = "  ".repeat(indent);
  if (v === null) return '<span class="t">null</span>';
  if (typeof v === "string") return `<span class="t">"${esc(v)}"</span>`;
  if (typeof v !== "object") return `<span class="t">${esc(v)}</span>`;
  if (Array.isArray(v)) return "[\n" + v.map(x => pad + "  " + jsonHtml(x, indent + 1)).join(",\n") + "\n" + pad + "]";
  return "{\n" + Object.entries(v as Record<string, unknown>).map(([k, x]) =>
    `${pad}  <span class="k">"${esc(k)}"</span>: ${jsonHtml(x, indent + 1)}`).join(",\n") + "\n" + pad + "}";
}

function schemaTree(): string {
  const props = (CONTRACT_SCHEMA.properties || {}) as Record<string, { type?: string; description?: string; $ref?: string; items?: { $ref?: string }; anyOf?: { $ref?: string; type?: string }[]; enum?: string[] }>;
  const defs = (CONTRACT_SCHEMA.$defs || {}) as Record<string, { enum?: string[]; properties?: Record<string, { type?: string; description?: string }>; description?: string }>;
  const refName = (x?: string) => (x || "").split("/").pop() || "";
  const typeOf = (p: (typeof props)[string]) =>
    p.type || (p.$ref ? refName(p.$ref) : p.items?.$ref ? refName(p.items.$ref) + "[]" : p.anyOf ? p.anyOf.map(a => a.type || refName(a.$ref)).join(" | ") : "object");
  const required = new Set(CONTRACT_SCHEMA.required || []);
  return Object.entries(props).map(([k, p]) => {
    const ref = p.$ref ? refName(p.$ref) : p.items?.$ref ? refName(p.items.$ref) : p.anyOf?.find(a => a.$ref) ? refName(p.anyOf.find(a => a.$ref)!.$ref) : "";
    const def = ref ? defs[ref] : undefined;
    const inner = def?.enum
      ? `<div class="d">one of: ${def.enum.map(e => `<span class="t">${esc(e)}</span>`).join(", ")}</div>`
      : def?.properties
        ? Object.entries(def.properties).map(([dk, dp]) => `<div><span class="k">${esc(dk)}</span> <span class="t">${esc(dp.type || "")}</span></div>`).join("")
        : "";
    const head = `<span class="k">${esc(k)}</span> <span class="t">${esc(typeOf(p))}</span>${required.has(k) ? "" : ' <span class="d">optional</span>'}`;
    return inner ? `<details><summary>${head}</summary>${p.description ? `<div class="d">${esc(p.description)}</div>` : ""}${inner}</details>`
      : `<div>${head}${p.description ? ` <span class="d">— ${esc(p.description)}</span>` : ""}</div>`;
  }).join("");
}

function renderContract() {
  const host = $("#contract");
  if (!host) return;
  const rec = byId(PROTAGONIST_ID);
  host.innerHTML = `
    <div class="ct-head"><span>QueryRecord · v${CONTRACT_VERSION}</span>
      <span class="ct-tabs"><button type="button" data-ct="schema" aria-pressed="true">Schema</button><button type="button" data-ct="record" aria-pressed="false">Real record</button></span></div>
    <div class="ct-body" id="ctBody"></div>`;
  const body = $("#ctBody", host)!;
  const show = (which: string) => {
    $$<HTMLButtonElement>("[data-ct]", host).forEach(b => b.setAttribute("aria-pressed", String(b.dataset.ct === which)));
    body.innerHTML = which === "schema"
      ? schemaTree() + `<p class="d" style="margin-top:12px">${esc(CONTRACT_SOURCE)}</p>`
      : rec ? `<pre>${jsonHtml(contractExample(rec))}</pre><p class="d">${esc(rec.id)} from the corpus, in contract shape. Branch annotations live in the gold record, not the contract.</p>` : "";
  };
  $$<HTMLButtonElement>("[data-ct]", host).forEach(b => b.addEventListener("click", () => show(b.dataset.ct!)));
  show("schema");
}

/* ------------------------------------------------------------- the corpus */

function renderComposition(tier: "all" | "natural" | "split") {
  const host = $("#compTable");
  if (!host) return;
  const rows = tier === "all" ? corpus() : corpus().filter(r => r.con === tier);
  const ty = countBy("ty", rows), sep = countBy("sep", rows), rel = countBy("rel", rows), sp = countBy("sp", rows), prov = countBy("prov", rows);
  const withB = rows.filter(r => (r.b || []).length).length;
  const nB = rows.reduce((s, r) => s + (r.b || []).length, 0);
  const line = (k: string, v: number, n: string) => `<tr><td>${k}</td><td class="num">${v}</td><td class="n">${n}</td></tr>`;
  const all = corpus().length;
  const bar = (k: string, v: number, cls = "") => `<div class="row"><span>${k}</span><div class="bar"><i class="${cls}" style="width:${(v / all * 100).toFixed(2)}%"></i></div><span class="n">${v}</span></div>`;
  const chart = `<div class="chart" aria-hidden="true">
    ${bar("instances", rows.length)}
    ${bar("conditional", ty.conditional || 0, "c")}
    ${bar("no conflict", ty.no_conflict || 0)}
    ${bar("with branches", withB, "c")}
    ${bar("cross-document", sep.cross_document || 0)}
    ${bar("opposed + redundant", (rel.opposed || 0) + (rel.redundant || 0), "o")}
  </div>`;
  // bars grow from their previous widths, so the tier toggle visibly reflows
  const prev = Array.from(host.querySelectorAll<HTMLElement>(".chart .bar i")).map(i => i.style.width);
  host.innerHTML = chart + `<div class="tscroll"><table><thead><tr><th>${tier === "all" ? "All instances" : tier === "natural" ? "Tier 1 · natural" : "Tier 2 · constructed"}</th><th class="num">n</th><th>what it means</th></tr></thead><tbody>
    ${line("instances", rows.length, "every row below partitions this")}
    ${line("conditional conflicts", ty.conditional || 0, "the class this project addresses")}
    ${line("no conflict (distractors)", ty.no_conflict || 0, "guards against a detector that always fires; SCR's denominator")}
    ${line("factual / temporal / opinion", (ty.factual || 0) + (ty.temporal || 0) + (ty.opinion || 0), "routed to prior work")}
    ${line("with a scope relation", (rel.refinement || 0) + (rel.disjoint || 0) + (rel.redundant || 0) + (rel.opposed || 0), `refinement ${rel.refinement || 0} · disjoint ${rel.disjoint || 0} · redundant ${rel.redundant || 0} · opposed ${rel.opposed || 0} — redundant and opposed are thin`)}
    ${line("cross-document separation", sep.cross_document || 0, "the hard, scarce case")}
    ${line("same-guide separation", sep.same_guide || 0, "reported as its own row, never blended")}
    ${line("synthetic split", sep.synthetic_split || 0, "one document's rule and exception split in two")}
    ${line("with gold branch structure", withB, "the denominator for PR / SR / HCR / SCR")}
    ${line("gold branches", nB, "what the branch-scored metrics count")}
    ${line("train · dev · test", sp.train || 0, `${sp.train || 0} · ${sp.dev || 0} · ${sp.test || 0}`)}
    ${line("two independent annotators", prov["two-annotator"] || 0, "κ is measured on this class only")}
    ${line("model-proposed + adjudicated", prov["model-proposed+adjudicated"] || 0, "never pooled with the above")}
  </tbody></table></div>
  ${tier === "natural" && !withB ? `<p class="note"><b>No Tier 1 instance carries branch annotation yet.</b> Every branch-scored number on this page is measured on Tier 2 constructed splits — the most load-bearing limitation of the work so far.</p>` : ""}`;
  const bars = Array.from(host.querySelectorAll<HTMLElement>(".chart .bar i"));
  const want = bars.map(b => b.style.width);
  bars.forEach((b, i) => (b.style.width = prev[i] ?? "0%"));
  requestAnimationFrame(() => requestAnimationFrame(() => bars.forEach((b, i) => (b.style.width = want[i]))));
}

function renderTierToggle() {
  const btns = $$<HTMLButtonElement>(".tier-toggle [role=radio]");
  const set = (b: HTMLButtonElement) => {
    btns.forEach(x => { x.setAttribute("aria-checked", String(x === b)); x.tabIndex = x === b ? 0 : -1; });
    renderComposition(b.dataset.tier as "all" | "natural" | "split");
  };
  btns.forEach((b, i) => {
    b.addEventListener("click", () => set(b));
    b.addEventListener("keydown", e => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      const n = btns[(i + (e.key === "ArrowRight" ? 1 : btns.length - 1)) % btns.length];
      n.focus(); set(n);
    });
  });
  btns.forEach((b, i) => (b.tabIndex = i === 0 ? 0 : -1));
  renderComposition("all");
}

/* --------------------------------------------------------------- findings */

const factHtml = (f: Fact) => `<div class="fact"><div class="v">${esc(f.value)}</div><p class="l">${esc(f.label)}</p>
  ${f.caveat ? `<p class="c">${esc(f.caveat)}</p>` : ""}<p class="s">${esc(f.source)}</p></div>`;

function renderFindings() {
  const host = $("#findings");
  if (!host) return;
  host.innerHTML = `
    <div class="finding"><h4>1 · Natural cross-document conflicts are scarce — and we know why</h4>
      <div class="facts">${TIER1_PILOT.map(factHtml).join("")}</div>
      <p class="note">Organisations state their carve-outs beside the rule, in the same document. That is why the
        cross-document case the literature leaves open is also the hardest to collect, and why the benchmark has a constructed tier.</p></div>
    <div class="finding"><h4>2 · The annotation gate, and what it cost</h4>
      <div class="facts">${KAPPA.map(factHtml).join("")}</div></div>
    <div class="finding"><h4>3 · Negative controls</h4>
      <div class="refuse">
        <div><span class="mono">Refused</span><pre># metrics — prevalence_estimate()
ValueError: the corpus was built by targeted
mining, not random sampling. A prevalence
figure from it would describe the search,
not the world.</pre><p>The tempting sentence was a prevalence percentage. The function raises rather than return a number the design cannot support.</p></div>
        <div><span class="mono">Permitted</span><p style="margin-top:10px">“Of 91 document pairings selected for likely conditional structure, 31 yielded candidate conflicts (0.34 per pairing). This is a yield rate under targeted search and <b>is not an estimate of prevalence in retrieval</b>.”</p><p class="note">Same data. The second sentence is true.</p></div>
      </div></div>`;
}

/* --------------------------------------------------------------- explorer */

function renderExplorer() {
  const host = $("#explorer");
  if (!host) return;
  const C = corpus();
  const FILTERS: [string, keyof CorpusRecord, string][] = [["fTy", "ty", "type"], ["fRel", "rel", "relation"], ["fCon", "con", "tier"], ["fSp", "sp", "split"], ["fDom", "dom", "domain"]];
  host.innerHTML = `
    <div class="ex-filters">
      <label class="sr-only" for="fq">Search the corpus</label>
      <input id="fq" type="search" placeholder="Search questions and passages">
      ${FILTERS.map(([id, key, label]) => `<label class="sr-only" for="${id}">${label}</label><select id="${id}"><option value="">${label}: all</option>
        ${[...new Set(C.map(r => r[key]).filter(Boolean))].sort().map(v => `<option value="${esc(v)}">${esc(human(String(v)))}</option>`).join("")}</select>`).join("")}
      <span class="ex-count" id="exCount" aria-live="polite"></span>
    </div>
    <div class="ex-list" id="exList" data-lenis-prevent></div>
    <div class="ex-detail" id="exDetail" data-lenis-prevent></div>`;
  let selected = C[0].id;
  const list = $("#exList", host)!, detail = $("#exDetail", host)!, count = $("#exCount", host)!;
  const q = $<HTMLInputElement>("#fq", host)!;
  const matching = () => {
    const s = q.value.trim().toLowerCase();
    return C.filter(r => {
      for (const [id, key] of FILTERS) { const v = ($<HTMLSelectElement>("#" + id, host)!).value; if (v && String(r[key]) !== v) return false; }
      if (!s) return true;
      return r.q.toLowerCase().includes(s) || r.id.toLowerCase().includes(s) || r.p.some(p => p.t.toLowerCase().includes(s));
    });
  };
  const showDetail = (r: CorpusRecord) => {
    detail.innerHTML = `<p style="font-size:17px;margin:0 0 6px">“${esc(r.q)}”</p>
      <p class="mono muted" style="margin:0 0 14px">${esc(r.id)}</p>
      <dl class="kv" style="margin-bottom:16px">
        <dt>type</dt><dd>${esc(r.ty)}</dd><dt>scope relation</dt><dd>${esc(r.rel || "— (not a conditional conflict)")}</dd>
        <dt>tier</dt><dd>${r.con === "natural" ? "Tier 1 · natural" : "Tier 2 · constructed"} · ${esc(human(r.sep))}</dd>
        <dt>split</dt><dd>${esc(r.sp)}</dd><dt>domain</dt><dd>${esc(human(r.dom))}</dd><dt>annotation</dt><dd>${esc(r.prov)}</dd>
      </dl>
      ${r.p.map(p => `<div class="psg" style="margin-bottom:8px;background:none"><span class="pid">${esc(p.id)}${p.d ? " · " + esc(p.d) : ""}${p.dt ? " · " + esc(p.dt) : ""}</span><p style="color:inherit">${esc(p.t)}</p>${p.u ? `<p class="mono muted" style="font-size:10px;text-transform:none">${esc(p.u)}</p>` : ""}</div>`).join("")}
      ${(r.b || []).length ? `<div class="branches" style="margin-top:12px">${r.b!.map(b => `<div class="branch ${b.def ? "def" : "exc"}"><span class="bcond">${b.def ? "Default" : "If · " + esc(b.cond || b.desc)}</span><span class="bout">${esc(b.out)}</span><div class="bmeta">from ${esc(b.sup || "—")}${b.exp ? " · " + esc(b.exp) : ""}</div></div>`).join("")}</div>`
        : `<p class="note"><b>No branch structure.</b> Detection labels only — which is why branch-scored metrics have a smaller denominator.</p>`}`;
  };
  const render = () => {
    const rows = matching();
    count.textContent = `${rows.length} of ${C.length}`;
    list.innerHTML = rows.length ? rows.slice(0, 400).map(r => `<button class="ex-row" type="button" data-id="${esc(r.id)}" ${r.id === selected ? 'aria-current="true"' : ""}>
      <span class="rid">${esc(r.id)}</span><div class="rq">${esc(r.q)}</div>
      <div class="tags"><span class="tag">${esc(r.ty)}</span>${r.rel ? `<span class="tag t-rel">${esc(r.rel)}</span>` : ""}<span class="tag">${r.con === "natural" ? "tier 1" : "tier 2"}</span><span class="tag">${esc(r.sp)}</span>${(r.b || []).length ? `<span class="tag">${r.b!.length} branches</span>` : ""}</div></button>`).join("")
      : `<p class="note" style="padding:14px">No instances match these filters.</p>`;
  };
  list.addEventListener("click", e => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>(".ex-row");
    if (!b) return;
    selected = b.dataset.id!;
    $$(".ex-row", list).forEach(x => x.toggleAttribute("aria-current", x === b));
    b.setAttribute("aria-current", "true");
    const r = C.find(x => x.id === selected);
    if (r) showDetail(r);
  });
  let t = 0;
  q.addEventListener("input", () => { clearTimeout(t); t = window.setTimeout(render, 140); });
  FILTERS.forEach(([id]) => $("#" + id, host)!.addEventListener("change", render));
  render(); showDetail(C[0]);
}

function renderBigStats() {
  const host = $("#bigStats");
  if (!host) return;
  const C = corpus();
  const branches = C.reduce((n, r) => n + (r.b || []).length, 0);
  const cond = C.filter(r => r.ty === "conditional").length;
  const items: [string, string][] = [[String(C.length), "benchmark instances"], [String(cond), "conditional conflicts"],
    [String(branches), "gold branches"], [KAPPA[0].value, "κ, conflict type (pilot 9)"]];
  host.innerHTML = items.map(([v, k]) => `<div class="bs"><div class="v" data-scramble>${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join("");
}

export function renderResearch() {
  renderBigStats();
  renderContract();
  renderMetrics();
  renderTierToggle();
  renderFindings();
  renderExplorer();
}

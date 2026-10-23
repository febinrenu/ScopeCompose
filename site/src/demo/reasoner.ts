/*  Act 06: try the reasoning.
 *
 *  Three kinds of input, always labelled: an illustrative example, a real
 *  benchmark record, or two branches you type. Whatever the input, every
 *  decision below is made by the verified engines, in order:
 *    screenPair -> (type) -> compare -> outcome agreement -> combine ->
 *    routeFor -> composeBranches -> renderTemplate -> checkFaithfulness,
 *  and the textual judge scores both answers.
 *
 *  What does NOT run here, and says so: stage-2 LLM escalation, the trained
 *  typing head, and B1 extraction, which all need a model provider. The
 *  branches are therefore gold (benchmark), authored (illustrative) or yours.
 */
import { FEE, RELATION_EXAMPLES, type Example } from "../data/examples";
import { branchy } from "../data/corpus";
import { audit, inside, readerDims, runChain, screen, selectionOf, type ChainResult, type Dim } from "../engine/pipeline";
import type { Branch, CorpusRecord, Passage } from "../engine/types";
import { $, $$, esc, fx, human } from "../ui/dom";
import { markAnswer } from "./story";
import { BranchTree } from "./tree";

type Mode = "illustrative" | "benchmark" | "own";

interface Input {
  kind: Mode;
  q: string;
  p: Passage[];
  b: Branch[];
  ty: string;
  oa?: boolean;
  provenance: string[];
  rec?: CorpusRecord;
}

const ILLUSTRATIVE: Example[] = [FEE, RELATION_EXAMPLES.disjoint, RELATION_EXAMPLES.redundant, RELATION_EXAMPLES.opposed];

const OWN_DEFAULT = {
  q: "Can I bring my partner on a student visa?",
  aOut: "dependants may not accompany the student", aDesc: "all cases", aAttr: "", aVal: "",
  bOut: "a partner may accompany the student as a dependant", bDesc: "postgraduate research students",
  bAttr: "course_level", bVal: "postgraduate_research",
};

export class Reasoner {
  private mode: Mode = "illustrative";
  private illKey = "fee";
  private recId = "";
  private own = { ...OWN_DEFAULT };
  private reader: Record<string, unknown> = {};
  private host: HTMLElement;
  private steps!: HTMLElement;
  private readerHost!: HTMLElement;
  private controls!: HTMLElement;
  private recs: CorpusRecord[];
  private tree!: BranchTree;

  constructor(host: HTMLElement) {
    this.host = host;
    this.recs = branchy();
    this.recId = this.recs.find(r => r.id === "wp1_fin_imp_003")?.id || this.recs[0].id;
    this.layout();
    this.renderControls();
    this.update(true);
  }

  private layout() {
    this.host.innerHTML = `
      <div class="rz-input">
        <h3 id="rzInputH">Input</h3>
        <div class="seg" role="radiogroup" aria-labelledby="rzInputH">
          <button role="radio" aria-checked="true" data-mode="illustrative">Illustrative</button>
          <button role="radio" aria-checked="false" data-mode="benchmark">Benchmark record</button>
          <button role="radio" aria-checked="false" data-mode="own">Your own</button>
        </div>
        <div id="rzControls"></div>
        <div class="rz-reader">
          <h3>Who is reading?</h3>
          <div class="rz-reader-dims" id="rzReader"></div>
        </div>
        <div id="rzTree"></div>
      </div>
      <div class="rz-out">
        <ol class="rz-steps" id="rzSteps"></ol>
      </div>`;
    this.steps = $("#rzSteps", this.host)!;
    this.readerHost = $("#rzReader", this.host)!;
    this.controls = $("#rzControls", this.host)!;
    this.tree = new BranchTree($("#rzTree", this.host)!);
    $$<HTMLButtonElement>("[data-mode]", this.host).forEach(b => {
      b.addEventListener("click", () => this.setMode(b.dataset.mode as Mode));
      b.addEventListener("keydown", e => {
        const all = $$<HTMLButtonElement>("[data-mode]", this.host);
        const i = all.indexOf(b);
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        const n = all[(i + (e.key === "ArrowRight" ? 1 : all.length - 1)) % all.length];
        n.focus(); this.setMode(n.dataset.mode as Mode);
      });
    });
  }

  private setMode(m: Mode) {
    if (m === this.mode) return;
    this.mode = m;
    $$<HTMLButtonElement>("[data-mode]", this.host).forEach(b => {
      const on = b.dataset.mode === m;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    });
    this.renderControls();
    this.update(true);
  }

  private renderControls() {
    const c = this.controls;
    if (this.mode === "illustrative") {
      c.innerHTML = `<label class="field"><span>Example</span>
        <select class="sel" id="rzIll">${ILLUSTRATIVE.map(e =>
          `<option value="${e.key}" ${e.key === this.illKey ? "selected" : ""}>${esc(e.label)}</option>`).join("")}</select></label>
        <div class="rz-prov"><span class="chip warn">illustrative input</span><span class="chip ok">real engine</span></div>`;
      $("#rzIll", c)!.addEventListener("change", e => { this.illKey = (e.target as HTMLSelectElement).value; this.update(true); });
    } else if (this.mode === "benchmark") {
      c.innerHTML = `<label class="field"><span>Record (${this.recs.length} with two or more gold branches)</span>
        <select class="sel" id="rzRec">${this.recs.map(r =>
          `<option value="${r.id}" ${r.id === this.recId ? "selected" : ""}>${esc(r.id)} — ${esc(r.q.length > 40 ? r.q.slice(0, 40) + "…" : r.q)}</option>`).join("")}</select></label>
        <div class="rz-prov" id="rzRecProv"></div>`;
      $("#rzRec", c)!.addEventListener("change", e => { this.recId = (e.target as HTMLSelectElement).value; this.update(true); });
    } else {
      const o = this.own;
      const fld = (id: string, label: string, val: string, ph = "", area = false) => `<label class="field"><span>${label}</span>${area
        ? `<textarea class="inp" id="${id}" rows="2" placeholder="${esc(ph)}">${esc(val)}</textarea>`
        : `<input class="inp" id="${id}" value="${esc(val)}" placeholder="${esc(ph)}">`}</label>`;
      c.innerHTML = `<div class="rz-byo">
        ${fld("oQ", "Question", o.q)}
        <p class="note" style="margin:0">Branch A is the general rule if you leave its attribute blank.</p>
        ${fld("oAOut", "Branch A · outcome", o.aOut, "what happens", true)}
        ${fld("oADesc", "Branch A · applies to", o.aDesc, "who or what")}
        <div class="pair">${fld("oAAttr", "attribute", o.aAttr, "optional")}${fld("oAVal", "value", o.aVal)}</div>
        ${fld("oBOut", "Branch B · outcome", o.bOut, "what happens", true)}
        ${fld("oBDesc", "Branch B · applies to", o.bDesc, "who or what")}
        <div class="pair">${fld("oBAttr", "attribute", o.bAttr, "e.g. card_tier")}${fld("oBVal", "value", o.bVal)}</div>
        <div class="rz-prov"><span class="chip warn">your input</span><span class="chip ok">real engine</span></div>
      </div>`;
      const map: Record<string, keyof typeof OWN_DEFAULT> = {
        oQ: "q", oAOut: "aOut", oADesc: "aDesc", oAAttr: "aAttr", oAVal: "aVal",
        oBOut: "bOut", oBDesc: "bDesc", oBAttr: "bAttr", oBVal: "bVal",
      };
      let t = 0;
      for (const [id, key] of Object.entries(map)) {
        $("#" + id, c)!.addEventListener("input", e => {
          this.own[key] = (e.target as HTMLInputElement).value;
          clearTimeout(t); t = window.setTimeout(() => this.update(false, true), 160);
        });
      }
    }
  }

  private input(): Input | { error: string } {
    if (this.mode === "illustrative") {
      const e = ILLUSTRATIVE.find(x => x.key === this.illKey) || FEE;
      return { kind: "illustrative", q: e.q, p: e.p, b: e.b, ty: e.ty, provenance: ["Illustrative input"] };
    }
    if (this.mode === "benchmark") {
      const r = this.recs.find(x => x.id === this.recId)!;
      return {
        kind: "benchmark", q: r.q, p: r.p, b: r.b || [], ty: r.ty, oa: r.oa, rec: r,
        provenance: [r.id, r.con === "split" ? "Tier 2 · constructed" : "Tier 1 · natural", human(r.sep), human(r.dom), "split: " + r.sp, r.prov],
      };
    }
    const o = this.own;
    if (!o.aOut.trim() || !o.bOut.trim()) return { error: "Give both branches an outcome." };
    if ((o.aAttr.trim() && !o.aVal.trim()) || (o.bAttr.trim() && !o.bVal.trim()))
      return { error: "An attribute needs a value (or leave both blank to make that branch the general rule)." };
    if (!o.aAttr.trim() && !o.bAttr.trim())
      return { error: "Give at least one branch an attribute and value, so the scope algebra has something to compare." };
    const mk = (id: string, sup: string, out: string, desc: string, attr: string, val: string): Branch => {
      const def = !attr.trim() || !val.trim();
      const b: Branch = { id, out: out.trim(), desc: desc.trim() || "all cases", def, sup };
      if (!def) {
        b.cond = b.desc;
        b.at = [{ n: attr.trim().toLowerCase().replace(/\s+/g, "_"), k: "categorical", v: [val.trim().toLowerCase().replace(/\s+/g, "_")] }];
      }
      return b;
    };
    return {
      kind: "own", q: o.q || "(no question)", ty: "conditional",
      p: [{ id: "p0", t: o.aOut, st: "official_policy" }, { id: "p1", t: o.bOut, st: "official_policy" }],
      b: [mk("b0", "p0", o.aOut, o.aDesc, o.aAttr, o.aVal), mk("b1", "p1", o.bOut, o.bDesc, o.bAttr, o.bVal)],
      provenance: ["Your input"],
    };
  }

  /** @param fresh re-build the reader controls; @param quiet skip the step animation */
  update(fresh: boolean, quiet = false) {
    const inp = this.input();
    if ("error" in inp) {
      $("#rzTree", this.host)!.hidden = true;
      this.steps.innerHTML = `<li class="rz-step"><h4>Waiting for a valid input</h4><p class="state err" role="alert">${esc(inp.error)}</p></li>`;
      this.readerHost.innerHTML = "";
      return;
    }
    $("#rzTree", this.host)!.hidden = false;
    if (fresh) this.buildReader(readerDims(inp.b));
    if (this.mode === "benchmark") {
      const pv = $("#rzRecProv", this.host);
      if (pv) pv.innerHTML = inp.provenance.map((p, i) => `<span class="chip ${i === 1 ? (inp.rec?.con === "natural" ? "ok" : "warn") : ""}">${esc(p)}</span>`).join("");
    }
    this.render(inp, quiet);
  }

  private buildReader(dims: Dim[]) {
    this.reader = {};
    if (!dims.length) {
      this.readerHost.innerHTML = `<p class="note" style="margin:0">These branches carry no typed attributes, so membership cannot be evaluated exactly. Pick another input to set a reader.</p>`;
      return;
    }
    this.readerHost.innerHTML = dims.map((d, i) => {
      const id = "rd" + i;
      if (d.kind === "numeric_range") {
        const start = d.lo !== null && Number.isFinite(d.lo) ? d.lo : 0;
        this.reader[d.name] = start;
        return `<label class="field"><span>${esc(human(d.name))}</span><input class="inp" type="number" id="${id}" data-dim="${esc(d.name)}" value="${start}"></label>`;
      }
      this.reader[d.name] = "__other__";
      return `<label class="field"><span>${esc(human(d.name))}</span><select class="sel" id="${id}" data-dim="${esc(d.name)}">
        <option value="__other__">— none of these —</option>
        ${d.values.map(v => `<option value="${esc(v)}">${esc(human(v))}</option>`).join("")}</select></label>`;
    }).join("");
    $$<HTMLInputElement | HTMLSelectElement>("[data-dim]", this.readerHost).forEach(el => {
      el.addEventListener(el.tagName === "SELECT" ? "change" : "input", () => {
        const d = el.dataset.dim!;
        this.reader[d] = el.tagName === "SELECT" ? el.value : (el.value === "" ? null : Number(el.value));
        this.update(false, true);
      });
    });
  }

  private render(inp: Input, quiet: boolean) {
    const chain: ChainResult = runChain({ p: inp.p, ty: inp.ty, oa: inp.oa }, inp.b);
    const sel = selectionOf({ p: inp.p }, inp.b);
    const selAudit = audit(sel.answer, inp.b);
    const compAudit = audit(chain.answer, inp.b);
    const inn = inside(inp.b, this.reader);
    const readerSet = Object.values(this.reader).some(v => v !== "__other__" && v !== null && v !== undefined);
    const detector = inp.kind === "own" ? null : screen({ p: inp.p });

    const steps: string[] = [];
    const step = (title: string, tag: string, body: string) =>
      steps.push(`<li class="rz-step" style="--i:${steps.length}"><h4>${title}<span class="mono">${tag}</span></h4>${body}</li>`);

    // 1 passages
    step("Retrieved passages", inp.kind === "own" ? "your branches stand in as passages" : "retrieval output",
      `<p class="note" style="margin:0 0 10px">“${esc(inp.q)}”</p>
       <div class="rz-psg">${inp.p.map(p => `<div class="psg" data-psg="${esc(p.id)}">
         <span class="pid">${esc(p.id)}${p.st ? " · " + esc(human(p.st)) : ""}${p.dt ? " · " + esc(p.dt) : ""}</span>
         <p>${esc(p.t)}</p></div>`).join("")}</div>`);

    // 2 detector
    if (detector) {
      const s = detector.score;
      const feats = Object.entries(detector.features).filter(([k, v]) => v > 0 && /cues|clash|asymmetry/.test(k)).slice(0, 6);
      step("Stage 1: is this a conflict?", "detection/stage1.py → detector.js",
        `<div class="meter" role="img" aria-label="stage 1 score ${fx(s)} against floor 0.35 and threshold 0.50">
           <div class="fill" style="width:${Math.max(1, s * 100)}%"></div>
           <div class="mark floor" style="left:35%"><span>0.35 escalate</span></div>
           <div class="mark" style="left:50%"><span>0.50 conflict</span></div>
         </div>
         <dl class="kv">
           <dt>score</dt><dd>${fx(s, 4)} → <b>${detector.isConflict ? "conflict" : "no conflict"}</b></dd>
           <dt>escalates?</dt><dd>${detector.needsEscalation ? "yes — would go to the LLM stage (not run in this browser)" : "no"}</dd>
           <dt>NLI (offline heuristic)</dt><dd>entail ${fx(detector.nli.entailment, 2)} · neutral ${fx(detector.nli.neutral, 2)} · contradict ${fx(detector.nli.contradiction, 2)}</dd>
         </dl>
         ${feats.length ? `<div class="feat">${feats.map(([k, v]) => `<span class="chip">${esc(k.replace(/_/g, " "))} ${fx(v, 2)}</span>`).join("")}</div>` : ""}
         ${!detector.isConflict && inp.ty === "conditional" ? `<p class="verdict bad"><b>Missed.</b> The untrained stage-1 scorer does not see this conditional conflict — the same gap the bench measures across the corpus. The steps below continue from the ${inp.kind === "benchmark" ? "gold" : "given"} label, as the bench’s oracle arm does.</p>` : ""}`);
    } else {
      step("Stage 1: is this a conflict?", "not run on typed text",
        `<p class="state">Stage 1 is not run on typed branches: on the benchmark it misses most real conditional conflicts, so pointing it at an improvised example would demonstrate the one part that does not work yet. This starts from your branches, where the system is strong.</p>`);
    }

    // 3 type
    step("Conflict type", "five classes · contract/models.py",
      `<p><span class="chip ${inp.ty === "conditional" ? "warn" : ""}">${esc(inp.ty)}</span>
       <span class="note">${inp.kind === "benchmark" ? "gold label — the trained typing head is not available offline" : inp.kind === "illustrative" ? "given with the illustrative input" : "assumed for typed branches"}</span></p>
       ${inp.ty !== "conditional" ? `<p class="note">Not a conditional conflict, so it routes to ${inp.ty === "no_conflict" ? "pass-through" : "the published strategy for that class"} and scope analysis does not apply.</p>` : ""}`);

    // 4 relation
    step("Scope relation", "scope/relation.py → engines.js",
      `<dl class="kv">
         <dt>applicability</dt><dd>${esc(chain.setRel)}</dd>
         <dt>outcomes</dt><dd>${chain.agree ? "agree" : "disagree"} <span class="note">(${esc(chain.agreeSource)})</span></dd>
         <dt>relation</dt><dd><b>${esc(chain.relation)}</b>${inp.rec?.rel ? ` <span class="note">gold: ${esc(inp.rec.rel)}${inp.rec.rel !== chain.relation ? " — the analyser disagrees" : ""}</span>` : ""}</dd>
       </dl>
       <p class="note">${esc(chain.rationale)}.</p>`);

    // 5 applicability + reader
    step("Branches and their conditions", "who each branch is true for",
      `<div class="branches">${inp.b.map((b, i) => `
        <div class="branch ${b.def ? "def" : "exc"}">
          <span class="bcond">${b.def ? "Rule — applies generally" : "If · " + esc(b.cond || b.desc)}</span>
          <span class="bout">${esc(b.out)}</span>
          <div class="bmeta">
            ${(b.at || []).map(a => `<span class="chip">${esc(human(a.n))} = ${esc((a.v || [a.lo + "–" + a.hi]).join(", "))}</span>`).join("")}
            <span>from ${esc(b.sup || "—")}</span>
            ${readerSet ? (inn[i] ? `<span class="chip ok">your reader is inside</span>` : `<span class="chip">reader outside</span>`) : ""}
          </div></div>`).join("")}</div>`);

    // 6 route
    const flagged = chain.composed.flags.nested || chain.composed.flags.crossed;
    step("Routing decision", "contract/routing.py",
      `<p><span class="chip solid">${esc(chain.action)}</span> <span class="note">${esc(chain.routeReason)}</span></p>
       ${flagged ? `<p class="verdict bad"><b>Flagged ${chain.composed.flags.nested ? "nested" : "crossed"}.</b> Second-order structure is outside first-order scope, so the operator fell back to selection — it is not presented as a composition.</p>` : ""}
       ${chain.action === "select" && !flagged ? `<p class="verdict">A genuine contradiction: selection is the honest answer here, and it is reported as one.</p>` : ""}`);

    // 7 answers
    const lost = inp.b.map((_b, i) => i).filter(i => !sel.composed.branches.some(k => k.id === inp.b[i].id));
    const wrongFor = lost.filter(i => inn[i]);
    let verdict = "";
    if (readerSet) {
      verdict = wrongFor.length
        ? `<p class="verdict bad"><b>Selection is wrong for this reader.</b> They fall inside ${wrongFor.map(i => "<i>" + esc(inp.b[i].desc) + "</i>").join(" and ")}, which selection deleted — and nothing in its answer signals that a branch was dropped.</p>`
        : `<p class="verdict"><b>Selection happens to be right for this reader</b> — they fall outside the deleted branch. Change the reader and it stops being right.</p>`;
    } else {
      verdict = `<p class="note">Set a reader on the left to see who each answer is wrong for.</p>`;
    }
    step("Two answers to the same question", "renderTemplate · judgeTextual",
      `<div class="answers">
        <div class="answer-box sel"><span class="k">Selection · keeps ${esc(sel.winner || "—")}</span>
          <p>${markAnswer(sel.answer, sel.composed.branches)}</p>
          <span class="chip ${selAudit.rates.SR > 0 ? "bad" : "ok"}">SR ${fx(selAudit.rates.SR, 2)}</span></div>
        <div class="answer-box"><span class="k">ScopeCompose · ${esc(chain.composed.resolution)}</span>
          <p>${markAnswer(chain.answer, chain.composed.branches)}</p>
          <span class="chip ${compAudit.rates.SR > 0 ? "bad" : "ok"}">SR ${fx(compAudit.rates.SR, 2)}</span></div>
      </div>${verdict}`);

    // 8 provenance + faithfulness
    const f = chain.faith;
    step("Provenance and faithfulness", "checkFaithfulness",
      `<ul class="trace">${chain.composed.branches.map(b => `<li tabindex="0" data-cite="${esc(b.sup || "")}"><span class="chip ${b.def ? "ok" : "warn"}">${esc(b.sup || "—")}</span>
         <span>${b.def ? "Rule" : "For " + esc(b.desc)}: “${esc(b.out)}”</span></li>`).join("")}</ul>
       <p class="verdict ${f.ok ? "good" : "bad"}"><b>Faithfulness ${f.ok ? "passes" : "fails"}.</b>
         ${f.ok ? "Every kept branch appears in its own clause, no figure is unsupported, and nothing adjudicates between sources." :
           [f.missingBranches.length ? "branches missing: " + f.missingBranches.join(", ") : "",
            f.unsupportedNumbers.length ? "figures not in any branch: " + f.unsupportedNumbers.join(", ") : "",
            f.adjudicated ? "the answer argues one source over another" : ""].filter(Boolean).map(esc).join("; ")}
         ${f.missingAttribution.length ? " Attribution missing for " + esc(f.missingAttribution.join(", ")) + " (reported, not fatal)." : ""}</p>
       <p class="rz-honest">Not run in this browser: stage-2 LLM escalation and B1 contrastive extraction, which need a model provider. The branches above are ${inp.kind === "benchmark" ? "gold annotations" : inp.kind === "illustrative" ? "authored with the example" : "the ones you typed"}, so this is the operator given its inputs — an upper bound, not an end-to-end measurement.</p>`);

    // the tree: who each answer kept, and where the reader stands
    const exc = inp.b.findIndex((b, i) => !b.def && inn[i]);
    const at = readerSet ? (exc >= 0 ? exc : inp.b.findIndex((b, i) => b.def && inn[i])) : -1;
    const keptBy = (c: { branches: Branch[] }) => c.branches.map(k => k.id);
    const selKept = keptBy(sel.composed), compKept = keptBy(chain.composed);
    const atId = at >= 0 ? inp.b[at].id : null;
    this.tree.update({
      branches: inp.b.map(b => ({ id: b.id, sup: b.sup || "—", def: !!b.def, cond: b.cond || b.desc, out: b.out })),
      selKept, compKept,
      selLabel: "keeps " + (selKept.length ? selKept.length + " of " + inp.b.length : "none"),
      compLabel: chain.action === "select" ? "selects (a real contradiction)" : "keeps " + compKept.length + " of " + inp.b.length,
      readerAt: atId,
      selRight: atId ? selKept.includes(atId) : null,
      compRight: atId ? compKept.includes(atId) : null,
    });

    this.steps.classList.toggle("quiet", quiet);
    this.steps.innerHTML = steps.join("");
    if (quiet) $$(".rz-step", this.steps).forEach(s => (s.style.animation = "none", s.style.opacity = "1", s.style.transform = "none"));
    this.wireCites();
  }

  /** Hover or focus a citation and its passage lights up in step 1. */
  private wireCites() {
    const hl = (id: string, on: boolean) => {
      $$(`[data-psg]`, this.steps).forEach(p => p.classList.toggle(id === "p0" ? "hl" : "hl-b", on && p.dataset.psg === id));
    };
    $$<HTMLElement>("[data-cite], .cite", this.steps).forEach(el => {
      const id = el.dataset.cite || (el.textContent || "").trim();
      el.addEventListener("mouseenter", () => hl(id, true));
      el.addEventListener("mouseleave", () => hl(id, false));
      el.addEventListener("focus", () => hl(id, true));
      el.addEventListener("blur", () => hl(id, false));
    });
  }
}

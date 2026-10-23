/*  Bring your own branches: the composition operator on whatever you type.
 *  Same chain as the original byo.js -- compare, outcomes_match, combine,
 *  routeFor, composeBranches, renderTemplate, checkFaithfulness -- now
 *  through the shared adapter, so it is literally the code the story runs.
 */
import { BYO_PRESETS } from "../data/examples";
import { runChain } from "../engine/pipeline";
import type { Branch } from "../engine/types";
import { $, esc } from "../ui/dom";

type Key = keyof typeof BYO_PRESETS;

export function renderByo(host: HTMLElement) {
  const col = (side: "d" | "e", title: string, hint: string) => `
    <div class="byo-col">
      <span class="mono">${title}</span><p class="note" style="margin:0">${hint}</p>
      <label class="field"><span>outcome</span><textarea class="inp" id="${side}Out" rows="2"></textarea></label>
      <label class="field"><span>applicability</span><input class="inp" id="${side}Desc"></label>
      <div class="byo-attr" style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <label class="field"><span>attribute</span><input class="inp" id="${side}Attr" placeholder="optional"></label>
        <label class="field"><span>value</span><input class="inp" id="${side}Val"></label></div>
    </div>`;
  host.innerHTML = `
  <div class="wrap">
    <a class="back" href="#act-06">← back to the story</a>
    <div class="eyebrow">Bring your own — the composition operator, live</div>
    <h2 class="display">Type two branches.<br><em>Watch it decide.</em></h2>
    <p class="lede">The scope algebra, the four-way relation, the routing table, the operator and the renderer all run here on whatever you type — the same code the bench scores.</p>
    <p class="note" style="max-width:76ch"><b>Why there is no box for raw passages.</b> Detection is the part of this pipeline that does not work yet: stage 1 misses most real conditional conflicts in the benchmark. A free-text detector would reliably get your example wrong and teach you nothing, so this starts one stage later, from branches.</p>
    <section class="panel" style="margin-top:28px">
      <div class="panel-h"><h3>Input</h3>
        <label class="field" style="min-width:260px"><span>start from</span><select class="sel" id="byoPreset">${Object.entries(BYO_PRESETS).map(([k, v]) =>
          `<option value="${k}">${esc(v.label)}</option>`).join("")}</select></label></div>
      <div class="panel-b">
        <label class="field" style="margin-bottom:18px"><span>question</span><input class="inp" id="byoQ"></label>
        <div class="byo-grid">
          ${col("d", "Branch A", "Leave the attribute blank to make this the default — the general rule.")}
          ${col("e", "Branch B", "Give it an attribute and value: a narrower scope the algebra can reason over exactly.")}
        </div>
      </div>
    </section>
    <section class="panel" style="margin-top:20px">
      <div class="panel-h"><h3>What the operator did</h3><span class="chip" id="byoAction"></span></div>
      <div class="panel-b" id="byoOut" aria-live="polite"></div>
    </section>
  </div>`;

  const v = (id: string) => ($<HTMLInputElement>("#" + id, host)!.value || "").trim();
  const set = (id: string, val: string) => { $<HTMLInputElement>("#" + id, host)!.value = val; };
  const branch = (side: "d" | "e", id: string): Branch => {
    const attr = v(side + "Attr"), val = v(side + "Val"), desc = v(side + "Desc") || "all cases";
    const def = !attr || !val;
    const b: Branch = { id, out: v(side + "Out"), desc, def, sup: side === "d" ? "p0" : "p1" };
    if (!def) { b.cond = desc; b.at = [{ n: attr.toLowerCase().replace(/\s+/g, "_"), k: "categorical", v: [val.toLowerCase().replace(/\s+/g, "_")] }]; }
    return b;
  };
  const run = () => {
    const a = branch("d", "b0"), b = branch("e", "b1");
    const out = $("#byoOut", host)!, chip = $("#byoAction", host)!;
    if (!a.out || !b.out) {
      chip.textContent = "incomplete"; chip.className = "chip warn";
      out.innerHTML = `<p class="state err" role="alert">Give both branches an outcome.</p>`;
      return;
    }
    const record = { p: [{ id: "p0", t: a.out, st: "official_policy" }, { id: "p1", t: b.out, st: "official_policy" }] };
    const r = runChain(record, [a, b]);
    chip.textContent = r.action;
    chip.className = "chip " + (r.action === "compose" ? "ok" : r.action === "select" ? "bad" : "warn");
    const kept = new Set(r.composed.branches.map(x => x.id));
    out.innerHTML = `
      <dl class="kv"><dt>set relation</dt><dd>${esc(r.setRel)}</dd>
        <dt>outcomes</dt><dd>${r.agree ? "agree" : "disagree"} <span class="note">(${esc(r.agreeSource)})</span></dd>
        <dt>scope relation</dt><dd>${esc(r.relation)}</dd><dt>routes to</dt><dd>${esc(r.action)}</dd><dt>resolution</dt><dd>${esc(r.composed.resolution)}</dd></dl>
      <p class="note">${esc(r.rationale)}.</p>
      <div class="branches" style="margin-top:14px">${[a, b].map(x => `<div class="branch ${x.def ? "def" : "exc"} ${kept.has(x.id) ? "" : "gone"}">
        <span class="bcond">${x.def ? "Default — applies generally" : "If · " + esc(x.cond || x.desc)}</span><span class="bout">${esc(x.out)}</span>
        <div class="bmeta">${kept.has(x.id) ? "kept" : "dropped by the operator"}</div></div>`).join("")}</div>
      <p class="verdict ${r.action === "compose" ? "good" : ""}"><b>Answer:</b> ${esc(r.answer)}</p>
      <p class="note"><b>Faithfulness:</b> ${r.faith.ok ? "passes — every branch appears in a clause of its own, no figure is unsupported, and nothing adjudicates between sources."
        : "fails — " + esc([r.faith.missingBranches.length ? "branches missing: " + r.faith.missingBranches.join(", ") : "",
            r.faith.unsupportedNumbers.length ? "figures not in any branch: " + r.faith.unsupportedNumbers.join(", ") : "",
            r.faith.adjudicated ? "the answer argues one source is better" : ""].filter(Boolean).join("; "))}</p>`;
  };
  const load = (k: Key) => {
    const p = BYO_PRESETS[k];
    set("byoQ", p.q);
    for (const [side, src] of [["d", p.d], ["e", p.e]] as const) {
      set(side + "Out", src.out); set(side + "Desc", src.desc); set(side + "Attr", src.attr); set(side + "Val", src.val);
    }
    run();
  };
  for (const id of ["byoQ", "dOut", "dDesc", "dAttr", "dVal", "eOut", "eDesc", "eAttr", "eVal"]) $("#" + id, host)!.addEventListener("input", run);
  $<HTMLSelectElement>("#byoPreset", host)!.addEventListener("change", e => load((e.target as HTMLSelectElement).value as Key));
  load("fee");
}

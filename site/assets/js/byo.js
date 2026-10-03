/*  Bring your own branches.
 *
 *  Deliberately NOT a free-text detector. Stage 1 returns "no conflict" at
 *  roughly 0.9 confidence on 42 of 42 genuine conflicts in the benchmark, so
 *  pointing it at someone's improvised example would demonstrate the one
 *  part of the pipeline that does not work, on the one occasion it matters.
 *
 *  Instead this exercises the part that does: type two branches and the
 *  composition operator runs for real -- compare, combine, route, compose,
 *  render, then a faithfulness check on its own output. Same code the bench
 *  scores at 0.976.
 */
"use strict";

(function () {
  const host = document.getElementById("view-byo");
  if (!host) return;

  const esc = s => String(s).replace(/[&<>"]/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const PRESETS = {
    fee: {
      label: "Fee with a product carve-out",
      q: "Is there a monthly fee on my current account?",
      d: { out: "a monthly account fee of 12.00 is payable", desc: "all cases",
           attr: "", val: "" },
      e: { out: "no monthly charge applies", desc: "Graduate Scheme accounts",
           attr: "account_product", val: "graduate" },
    },
    visa: {
      label: "Work ban with a sponsorship exception",
      q: "Can I work while studying?",
      d: { out: "employment is not permitted during the validity of the leave",
           desc: "all cases", attr: "", val: "" },
      e: { out: "work is allowed up to 20 hours per week during term time",
           desc: "students sponsored by a provider with track record status",
           attr: "sponsor_status", val: "track_record" },
    },
    disjoint: {
      label: "Two rules that never meet",
      q: "What notice period applies?",
      d: { out: "one month of notice is required", desc: "fixed-term contracts",
           attr: "contract_type", val: "fixed_term" },
      e: { out: "three months of notice is required", desc: "permanent contracts",
           attr: "contract_type", val: "permanent" },
    },
  };

  function fieldset(side, title, hint){
    return `
      <div class="byo-col">
        <div class="lab">${title}</div>
        <p class="note" style="margin:6px 0 10px">${hint}</p>
        <label class="lab" for="${side}Out">outcome</label>
        <textarea id="${side}Out" rows="2" placeholder="what happens"></textarea>
        <label class="lab" for="${side}Desc">applicability</label>
        <input id="${side}Desc" type="text" placeholder="who or what it applies to">
        <div class="byo-attr">
          <input id="${side}Attr" type="text" placeholder="attribute (optional)">
          <input id="${side}Val" type="text" placeholder="value">
        </div>
      </div>`;
  }

  host.innerHTML = `
  <section>
    <div class="wrap">
      <div class="eyebrow">Bring your own &mdash; the composition operator, live</div>
      <h2 class="serif">Type two branches. Watch it decide.</h2>
      <p class="lede">The scope algebra, the four-way relation, the routing table, the operator
        and the renderer all run here on whatever you type. This is the same code the bench
        scores, not a demonstration of it.</p>

      <p class="note" style="max-width:76ch"><b>Why there is no box for raw passages.</b>
        Detection is the part of this pipeline that does not work yet &mdash; stage 1 returns
        <i>no conflict</i> on 42 of 42 genuine conflicts in the benchmark. A free-text detector
        would reliably get your example wrong and teach you nothing. So this starts one stage
        later, from branches, where the system is actually good.</p>

      <div class="panel" style="margin-top:30px">
        <div class="phead"><span class="ptitle">Input</span>
          <div class="ctl" style="margin-left:auto">
            <label class="lab" for="byoPreset">Start from</label>
            <select id="byoPreset">${Object.entries(PRESETS).map(([k, v]) =>
              `<option value="${k}">${esc(v.label)}</option>`).join("")}</select>
          </div></div>
        <div class="pbody">
          <label class="lab" for="byoQ">question</label>
          <input id="byoQ" type="text" style="width:100%; margin-bottom:18px">
          <div class="byo-grid">
            ${fieldset("d", "Branch A", "Leave the attribute blank to mark this the "
              + "default &mdash; the general rule, applying to every case.")}
            ${fieldset("e", "Branch B", "Give it an attribute and value to make it a narrower "
              + "scope the algebra can reason over exactly.")}
          </div>
        </div>
      </div>

      <div class="panel" style="margin-top:22px">
        <div class="phead"><span class="ptitle">What the operator did</span>
          <span class="pill" id="byoAction" style="margin-left:auto"></span></div>
        <div class="pbody" id="byoOut"></div>
      </div>
    </div>
  </section>`;

  const $$id = id => document.getElementById(id);
  const ids = ["byoQ", "dOut", "dDesc", "dAttr", "dVal", "eOut", "eDesc", "eAttr", "eVal"];

  function load(key){
    const p = PRESETS[key];
    $$id("byoQ").value = p.q;
    for (const [side, src] of [["d", p.d], ["e", p.e]]){
      $$id(side + "Out").value = src.out;
      $$id(side + "Desc").value = src.desc;
      $$id(side + "Attr").value = src.attr;
      $$id(side + "Val").value = src.val;
    }
    run();
  }

  function branchFrom(side, id){
    const attr = $$id(side + "Attr").value.trim();
    const val = $$id(side + "Val").value.trim();
    const desc = $$id(side + "Desc").value.trim() || "all cases";
    const isDefault = !attr || !val;
    const b = { id, out: $$id(side + "Out").value.trim(), desc,
                def: isDefault, sup: side === "d" ? "p0" : "p1" };
    if (!isDefault){
      b.cond = desc;
      b.at = [{ n: attr.toLowerCase().replace(/\s+/g, "_"), k: "categorical",
                v: [val.toLowerCase().replace(/\s+/g, "_")] }];
    }
    return b;
  }

  function run(){
    const a = branchFrom("d", "b0"), b = branchFrom("e", "b1");
    const out = $$id("byoOut"), chip = $$id("byoAction");
    if (!a.out || !b.out){
      chip.textContent = "incomplete";
      chip.className = "pill p-un";
      out.innerHTML = `<p class="note" style="margin:0">Give both branches an outcome.</p>`;
      return;
    }

    // The real chain, in order.
    const setRel = compare({ def: a.def, at: a.at }, { def: b.def, at: b.at });
    const agree = outcomes_match(a.out, b.out);
    const [relation, why] = combine(setRel, agree);
    const action = routeFor("conditional", relation);
    const record = { p: [{ id: "p0", t: a.out, st: "official_policy" },
                         { id: "p1", t: b.out, st: "official_policy" }] };
    const composed = composeBranches(record, [a, b], action);
    const answer = renderTemplate(composed);
    const faith = checkFaithfulness(answer, composed.branches);

    chip.textContent = action;
    chip.className = "pill " + (action === "compose" ? "p-ok"
                              : action === "select" ? "p-no" : "p-un");

    const kept = new Set(composed.branches.map(x => x.id));
    out.innerHTML = `
      <dl class="kv">
        <dt>set relation</dt><dd>${esc(setRel)}</dd>
        <dt>outcomes</dt><dd>${agree ? "agree" : "disagree"}   (lexical &mdash; the pipeline uses NLI here)</dd>
        <dt>scope relation</dt><dd>${esc(relation)}</dd>
        <dt>routes to</dt><dd>${esc(action)}</dd>
        <dt>resolution</dt><dd>${esc(composed.resolution)}</dd>
      </dl>
      <p class="note">${esc(why)}.</p>
      <div class="branches" style="margin-top:16px">
        ${[a, b].map(x => `
          <div class="branch ${kept.has(x.id) ? "live" : "gone"}">
            <span class="bcond">${x.def ? "DEFAULT &mdash; applies generally"
              : "IF &middot; " + esc(x.cond || x.desc)}</span>
            <span class="bout">${esc(x.out)}</span>
            <div class="bmeta">scope: <span class="ic">${esc(x.desc)}</span>
              &middot; ${kept.has(x.id) ? "kept" : "dropped by the operator"}</div>
          </div>`).join("")}
      </div>
      <div class="verdict ${action === "compose" ? "good" : ""}" style="margin-top:18px">
        <b>Answer:</b> ${esc(answer)}
      </div>
      <p class="note"><b>Faithfulness:</b> ${faith.ok
        ? "passes &mdash; every branch appears in a clause of its own, no figure is unsupported, "
          + "and nothing adjudicates between sources."
        : "fails &mdash; " + [
            faith.missingBranches.length ? `branches missing from the answer: ${faith.missingBranches.join(", ")}` : "",
            faith.unsupportedNumbers.length ? `figures not in any branch: ${faith.unsupportedNumbers.join(", ")}` : "",
            faith.adjudicated ? "the answer argues one source is better than another" : "",
          ].filter(Boolean).join("; ")}
        ${faith.missingAttribution.length
          ? ` Attribution missing for ${faith.missingAttribution.join(", ")} (reported, not fatal).`
          : ""}</p>`;
  }

  ids.forEach(id => $$id(id).addEventListener("input", run));
  $$id("byoPreset").addEventListener("change", e => load(e.target.value));
  load("fee");
})();

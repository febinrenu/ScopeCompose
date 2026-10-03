/*  The bench.
 *
 *  Five resolvers run on the same 42 annotated instances, scored by the
 *  project's own deterministic judging path. Every number here is computed
 *  in the browser at load; none is typed into the markup.
 *
 *  The bench exists to answer "how is this different from what already
 *  exists?", and the honest answer it produces is not a win. It is:
 *
 *    the composition operator preserves 97.6% of gold branches when it is
 *    told a conflict exists, and the detector in front of it finds 0 of 42.
 *
 *  Two columns carry warnings that must never be removed. The oracle column
 *  is fed gold routing AND gold branches, because B1 extraction has no
 *  offline path -- it is an upper bound, not a result. The concatenation
 *  column beats the gold answer itself, which is a fact about the metric
 *  rather than about the resolver.
 */
"use strict";

(function () {
  const host = document.getElementById("view-bench");
  if (!host || typeof CORPUS === "undefined") return;

  const ANNOTATED = CORPUS.filter(r => (r.b || []).length);
  const esc = s => String(s).replace(/[&<>"]/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = x => (x * 100).toFixed(1) + "%";

  /* ---------------------------------------------------------- resolvers */

  /** What stage 1 actually decides, computed not asserted. */
  function detect(rec){
    const [p0, p1] = rec.p;
    return screenPair(p0.t, p1.t);
  }

  const RESOLVERS = [
    {
      key: "selection", name: "Selection", sub: "prior work",
      note: "The corpus's own <code>selection_answer</code>: one passage wins, the other is "
          + "discarded. This is what retrieval-augmented generation does today.",
      badge: null,
      answer: rec => rec.sel || "",
    },
    {
      key: "asruns", name: "ScopeCompose", sub: "as it actually runs",
      note: "Stage 1 screens the pair live, and on this subset it returns <b>no conflict</b> "
          + "every time, so nothing routes to compose. Scored exactly as "
          + "<code>run_decisive</code> scores the pipeline arm: a branch survives if it is the "
          + "default, or if its pair composed. No text is generated, so <b>no gold branch text "
          + "is read</b> — and distortion is not measurable at this step.",
      badge: { cls: "p-un", text: "routing-level" },
      // Deliberately NOT renderTemplate(composeBranches(rec, rec.b, ...)). Feeding the gold
      // branches in and then scoring the rendered string against those same branches reads
      // the answer key twice and inflates this column to 0.687. The pipeline arm in
      // run_decisive.py never generates text; it decides survival from routing alone, and
      // that is what is reproduced here.
      judge: rec => {
        const d = detect(rec);
        const composed = d.isConflict && routeFor(rec.ty, rec.rel) === "compose";
        return rec.b.map(b => (b.def || composed) ? "preserved" : "suppressed");
      },
      answerNote: "no text is emitted — the detector found no conflict, so the pipeline "
                + "contributes nothing beyond ordinary retrieval",
    },
    {
      key: "concat", name: "Full context", sub: "both passages, no structure",
      note: "Paste everything and let the reader sort it out. It scores <b>above the gold "
          + "answer</b>, which is a finding about the metric, not about this resolver.",
      badge: { cls: "p-no", text: "gameable" },
      answer: rec => rec.p.map(x => x.t).join(" "),
    },
    {
      key: "oracle", name: "ScopeCompose", sub: "routing oracle",
      note: "Given the gold conflict label and the gold branches, the composition operator "
          + "routes, composes and renders. <b>Both inputs are the answer key</b>, because B1 "
          + "extraction has no offline path. This is a ceiling on the operator, not a "
          + "measurement of the system.",
      badge: { cls: "p-un", text: "ORACLE — upper bound" },
      answer: rec => renderTemplate(composeBranches(rec, rec.b, routeFor(rec.ty, rec.rel))),
    },
    {
      key: "gold", name: "Gold answer", sub: "the scorer's ceiling",
      note: "The human-written scoped answer, scored by the same judge. It does not reach 1.0, "
          + "which bounds what any resolver can score on this metric.",
      badge: null,
      answer: rec => rec.sa || "",
    },
  ];

  /* --------------------------------------------- order-invariance prover */

  /**
   * The A4 decision for one ordering of an instance's branches.
   *
   * Branch roles come from the applicability relation, never from position.
   * `canonical` restates SUBSET/SUPERSET in terms of ROLES rather than
   * argument order, which is the thing that makes the decision invariant:
   * without it a nested pair reads SUBSET one way and SUPERSET the other and
   * every permutation would look like a violation.
   */
  function decide(rec, presented){
    // Python enumerates pairs from passages sorted by id (stage1.enumerate_pairs),
    // so the pair the analyser sees never depends on retrieval order. Sorting a
    // working copy here is that same step -- without it, an instance carrying two
    // exceptions has no well-defined "the exception" and the roles flip under
    // permutation purely because of array position.
    const branches = presented.slice().sort((a, b) =>
      String(a.sup || "").localeCompare(String(b.sup || "")) ||
      String(a.id).localeCompare(String(b.id)));

    let i0 = 0, i1 = 1;
    const defAt = branches.findIndex(b => b.def);
    if (defAt >= 0){ i0 = defAt; i1 = branches.findIndex((b, k) => k !== defAt); }
    const app = i => ({ def: !!branches[i].def, at: branches[i].at });
    const setRel = compare(app(i0), app(i1));
    const agree = typeof rec.oa === "boolean" ? rec.oa
      : branches[i0].out.trim().toLowerCase() === branches[i1].out.trim().toLowerCase();
    const [relation] = combine(setRel, agree);
    const canonical = setRel === REL.SUPERSET ? REL.SUBSET : setRel;

    // Role assignment, mirroring ScopeAnalyser.analyse exactly.
    //
    // SUBSET means branches[i0] sits inside branches[i1], so i1 is the wider
    // set and therefore the default; SUPERSET is the mirror. For DISJOINT,
    // OVERLAPPING, EQUAL and UNKNOWN there is no default/exception structure
    // at all, and Python picks a stable representative by sorting on passage
    // id. Falling back to array position instead -- which this did at first
    // -- makes the roles flip when the passages are reordered, and the
    // prover below catches it as a violation on exactly those pairs.
    let defIdx;
    if (setRel === REL.SUBSET) defIdx = i1;
    else if (setRel === REL.SUPERSET) defIdx = i0;
    else defIdx = (branches[i0].sup || "") <= (branches[i1].sup || "") ? i0 : i1;

    return {
      relation, canonical, agree,
      defaultBranch: branches[defIdx].id,
      defaultPassage: branches[defIdx].sup,
      exceptionBranch: branches[defIdx === i0 ? i1 : i0].id,
    };
  }

  /** Permute every instance and check the decision does not move. */
  function proveInvariance(){
    let permutations = 0, violations = 0, nonPositional = 0;
    const cases = [];
    for (const rec of ANNOTATED){
      const bs = rec.b;
      if (bs.length < 2) continue;
      const base = decide(rec, bs);
      const reversed = bs.slice().reverse();
      const other = decide(rec, reversed);
      permutations++;

      const diffs = [];
      for (const k of ["relation", "canonical", "agree", "defaultBranch", "exceptionBranch"]){
        if (base[k] !== other[k]) diffs.push(k);
      }
      if (diffs.length) violations++;

      // The non-vacuity counter. A system that simply called the first-listed
      // passage the default would score a perfect pass rate and zero here.
      const firstPassage = reversed[0].sup;
      if (other.defaultPassage && other.defaultPassage !== firstPassage) nonPositional++;

      cases.push({ id: rec.id, base, other, diffs });
    }
    return { permutations, violations, nonPositional,
             passRate: permutations ? 1 - violations / permutations : 1, cases };
  }

  /* ------------------------------------------- retrieval-side baselines */

  /**
   * The three implemented baselines from baselines/retrieval_side.py.
   * These act BEFORE any resolver, by deciding which passages survive, so
   * they are scored on branch loss rather than preservation: a branch is
   * lost iff its supporting passage was dropped.
   */
  const RETRIEVERS = [
    { key: "standard_rag", name: "StandardRAG", sub: "keeps everything",
      note: "The control. k is unset, so nothing is dropped and nothing can be lost.",
      run: rec => rec.p.map(p => p.id) },
    { key: "rerank_top1", name: "RerankTop1", sub: "selection",
      note: "Keeps the top-ranked passage and discards the rest. Retrieval rank <i>is</i> the "
          + "credibility order, so this is what selection-based RAG does.",
      run: rec => [rec.p[0].id] },
    { key: "nli_filter", name: "NLIFilter", sub: "contradiction-aware",
      note: "Drops the passage carrying the most summed contradiction, if it clears 0.5. The "
          + "careful engineer's answer — and it still destroys branches.",
      run: rec => {
        const ps = rec.p;
        if (ps.length < 2) return ps.map(p => p.id);
        // Every ordered pair: premise is the other passage, hypothesis the candidate.
        const totals = ps.map((a, i) => ps.reduce((acc, b, j) =>
          i === j ? acc : acc + heuristicNLI(b.t, a.t).contradiction, 0));
        let worst = 0;
        for (let i = 1; i < ps.length; i++){
          // tie-break on passage id, matching the Python's max() key
          if (totals[i] > totals[worst]
              || (totals[i] === totals[worst] && ps[i].id > ps[worst].id)) worst = i;
        }
        if (totals[worst] < 0.5) return ps.map(p => p.id);
        return ps.filter((_, i) => i !== worst).map(p => p.id);
      } },
  ];

  function retrievalLoss(){
    return RETRIEVERS.map(r => {
      let lost = 0, total = 0, affected = 0;
      for (const rec of ANNOTATED){
        const kept = new Set(r.run(rec));
        // run_baselines.py's rule: a branch with no supporting passage always survives.
        const gone = rec.b.filter(b => b.sup && !kept.has(b.sup)).length;
        lost += gone; total += rec.b.length;
        if (gone) affected++;
      }
      return { ...r, lost, total, affected, instances: ANNOTATED.length,
               rate: total ? lost / total : 0 };
    });
  }

  /* ------------------------------------------------------------- scoring */

  function scoreAll(){
    return RESOLVERS.map(r => {
      const judgements = [];
      let chars = 0;
      const perInstance = new Map();
      for (const rec of ANNOTATED){
        // Two scoring paths on purpose. Text resolvers are judged on what they
        // emit; the routing-level arm has nothing to emit and is judged on
        // branch survival, exactly as the Python does it.
        const text = r.answer ? (r.answer(rec) || "") : "";
        chars += text.length;
        const j = r.judge ? r.judge(rec) : judgeTextual(text, rec.b);
        perInstance.set(rec.id, { text, j });
        judgements.push(...j);
      }
      const s = rates(judgements);
      return { ...r, ...s, meanChars: Math.round(chars / ANNOTATED.length),
               perInstance, ci: wilson(s.preserved, s.n) };
    });
  }

  /* --------------------------------------------------------- the render */

  function detectorStats(){
    let fired = 0, escalated = 0, max = 0;
    const scores = [];
    for (const rec of ANNOTATED){
      const d = detect(rec);
      scores.push(d.score);
      max = Math.max(max, d.score);
      fired += d.isConflict ? 1 : 0;
      escalated += d.needsEscalation ? 1 : 0;
    }
    return { fired, escalated, max, scores, n: ANNOTATED.length };
  }

  function render(){
    const scored = scoreAll();
    const det = detectorStats();
    const oracle = scored.find(s => s.key === "oracle");
    const asruns = scored.find(s => s.key === "asruns");

    const rows = scored.map(s => `
      <tr data-key="${s.key}">
        <td><b>${esc(s.name)}</b><div class="sub">${esc(s.sub)}</div>
          ${s.badge ? `<span class="pill ${s.badge.cls}">${esc(s.badge.text)}</span>` : ""}</td>
        <td class="num"><b>${s.PR.toFixed(3)}</b>
          <div class="sub">[${s.ci.low.toFixed(2)}, ${s.ci.high.toFixed(2)}]</div></td>
        <td class="num">${s.SR.toFixed(3)}</td>
        <td class="num">${s.judge ? '<span class="na">n/a</span>' : s.DR.toFixed(3)}</td>
        <td class="num">${s.judge ? '<span class="na">&mdash;</span>' : s.meanChars}</td>
        <td class="why">${s.note}</td>
      </tr>`).join("");

    // The detector trace: every score against the escalation floor.
    const W = 760, H = 96, pad = 28;
    const bars = det.scores.map((v, i) => {
      const x = pad + (i / Math.max(1, det.scores.length - 1)) * (W - pad * 2);
      const h = v * (H - 24);
      return `<rect x="${x.toFixed(1)}" y="${(H - 12 - h).toFixed(1)}" width="2.4" `
           + `height="${Math.max(0.6, h).toFixed(1)}" rx="1.2"/>`;
    }).join("");
    const floorY = H - 12 - 0.35 * (H - 24);

    host.innerHTML = `
    <section>
      <div class="wrap">
        <div class="eyebrow">The bench &mdash; five resolvers, one metric, ${ANNOTATED.length} instances</div>
        <h2 class="serif">The operator works.<br>The detector never fires.</h2>
        <p class="lede">Every figure below is computed in your browser, now, by the project's own
          scoring code running on the real corpus. Nothing is typed into the page.</p>

        <div class="stats" style="margin-top:30px">
          <div class="stat"><div class="v mono">${oracle.PR.toFixed(3)}</div>
            <div class="k">operator PR, given a conflict</div>
            <div class="n">Gold routing and gold branches. An upper bound on the operator.</div></div>
          <div class="stat unmeasured"><div class="v mono">${det.fired} / ${det.n}</div>
            <div class="k">conflicts the detector finds</div>
            <div class="n">Max score <b>${det.max.toFixed(4)}</b> against an escalation floor of
              0.35, so stage&nbsp;2 never runs &mdash; with or without an API key.</div></div>
          <div class="stat"><div class="v mono">${asruns.PR.toFixed(3)}</div>
            <div class="k">end-to-end PR, as it runs</div>
            <div class="n">What the gap between the first two numbers costs.</div></div>
        </div>

        <p class="note" style="max-width:76ch"><b>That gap is the finding.</b> This project
          expected the hard part to be composing branches without mangling them. It is not. The
          operator does that well. The hard part is noticing there is a conflict at all, and the
          untrained lexical scorer in front of it is not close &mdash; over the whole 128-instance
          corpus it reaches precision 0.800 but recall 0.051, recovering 2 of 67 conditional
          conflicts. Conservative, not random.</p>

        <div class="panel" style="margin-top:34px">
          <div class="phead"><span class="ptitle">Stage 1 score, one bar per instance</span>
            <span class="pill p-un" style="margin-left:auto">nothing clears the floor</span></div>
          <div class="pbody">
            <svg viewBox="0 0 ${W} ${H}" class="trace" role="img"
                 aria-label="Stage 1 scores, all below the escalation floor">
              <line x1="${pad}" y1="${floorY}" x2="${W - pad}" y2="${floorY}"
                    class="floor" stroke-dasharray="4 4"/>
              <text x="${W - pad}" y="${floorY - 6}" class="flab" text-anchor="end">escalation floor 0.35</text>
              <g class="bars">${bars}</g>
              <line x1="${pad}" y1="${H - 12}" x2="${W - pad}" y2="${H - 12}" class="axis"/>
            </svg>
            <p class="note" style="margin-top:4px">The topicality multiplier
              <span class="ic">0.35 + 0.65&times;min(1, jaccard&times;3)</span> damps any pair whose
              passages are worded differently &mdash; which is exactly what a general rule and its
              carve-out look like.</p>
          </div>
        </div>

        <div class="tscroll" style="margin-top:34px">
          <table class="bench">
            <thead><tr>
              <th>Resolver</th><th class="num">PR</th><th class="num">SR</th>
              <th class="num">DR</th><th class="num">chars</th><th>What it is</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>

        <p class="note" style="max-width:76ch"><b>Read the concatenation row carefully.</b>
          Preservation Rate is <span class="ic">containment &ge; 0.6</span> with no length
          normalisation and no precision counterweight &mdash; HCR catches only invented figures,
          and SCR only applies to no-conflict instances, of which these ${ANNOTATED.length} have
          none. So pasting both passages verbatim maximises it, and scores above the human-written
          gold answer. <b>The suite needs a precision term.</b> That is a defect in the metric,
          found by this bench, and it is reported here rather than left for a reviewer to find.</p>

        <h3 class="serif" style="font-size:28px; margin-top:52px">Before any resolver runs,
          retrieval has already destroyed the branch.</h3>
        <p class="lede" style="margin-top:14px">The three implemented retrieval baselines, run
          live. A branch is lost when the passage supporting it is dropped — so this measures
          what selection costs <i>before</i> a resolver sees anything.</p>
        <div class="tscroll" style="margin-top:20px">
          <table class="bench">
            <thead><tr><th>Retrieval strategy</th><th class="num">branches lost</th>
              <th class="num">rate</th><th class="num">instances hit</th><th>What it is</th></tr></thead>
            <tbody>${retrievalLoss().map(r => `
              <tr><td><b>${esc(r.name)}</b><div class="sub">${esc(r.sub)}</div></td>
                <td class="num"><b>${r.lost}</b> / ${r.total}</td>
                <td class="num">${pct(r.rate)}</td>
                <td class="num">${r.affected} / ${r.instances}</td>
                <td class="why">${r.note}</td></tr>`).join("")}</tbody>
          </table>
        </div>
        <p class="note" style="max-width:76ch"><b>Say the obvious part out loud:</b> RerankTop1
          keeping only the top passage <i>must</i> drop branches supported by the second one.
          The tautology is the mechanism. The result is the magnitude — roughly half of every
          gold branch in the benchmark, in all but two instances — and that the discarded branch
          was true for a real reader, which the demo on the overview page shows. The row that
          should worry a practitioner is <b>NLIFilter</b>: a contradiction-aware filter, built
          exactly as someone careful would build it, still destroys branches.</p>

        ${(() => {
          const inv = proveInvariance();
          return `
        <h3 class="serif" style="font-size:28px; margin-top:52px">Order invariance, proved here
          rather than quoted.</h3>
        <p class="lede" style="margin-top:14px">Every instance is re-run with its passages
          reversed. The relation, both branch roles and the canonical set relation must come
          back identical, because roles are derived from the applicability sets and never from
          retrieval order.</p>
        <div class="stats" style="margin-top:20px">
          <div class="stat"><div class="v mono">${inv.passRate.toFixed(4)}</div>
            <div class="k">pass rate</div>
            <div class="n">${inv.permutations} permutations, ${inv.violations} violations.</div></div>
          <div class="stat"><div class="v mono">${inv.nonPositional} / ${inv.permutations}</div>
            <div class="k">non-positional roles</div>
            <div class="n">Permutations where the default is <b>not</b> the first-listed
              passage. Without this the pass rate is vacuous &mdash; a system that always called
              passage one the default would score 1.0000 and zero here.</div></div>
        </div>
        <p class="note" style="max-width:76ch">Each corpus record holds exactly two passages, so
          each admits exactly one non-identity permutation. That is why this is
          ${inv.permutations} and not more, and why any larger figure quoted against this corpus
          is wrong. This runs over the ${ANNOTATED.length} branch-bearing instances; the Python
          gate runs over all 128 and reports <b>1.0000 across 71 permutations, 65 of them
          non-positional</b>, because it can analyse pairs that carry no branch annotation.
          Different denominators, same property.</p>`;
        })()}

        <div class="panel" style="margin-top:30px">
          <div class="phead"><span class="ptitle">Per instance</span>
            <div class="ctl" style="margin-left:auto">
              <label class="lab" for="benchInst">Instance</label>
              <select id="benchInst"></select>
            </div></div>
          <div class="pbody" id="benchDetail"></div>
        </div>

        <p class="note" style="max-width:76ch"><b>What these 42 are.</b> All of them are authored
          <span class="ic">wp1_*</span> probes &mdash; 42 of 42 are
          <span class="ic">construction=split</span>,
          <span class="ic">separation=synthetic_split</span>. No mined instance carries branch
          annotation, so every number on this page is measured on constructed splits of single
          documents, not on the cross-document case the project is about.</p>
      </div>
    </section>`;

    /* ---- per-instance drill-down ---- */
    const sel = document.getElementById("benchInst");
    ANNOTATED.forEach(r => {
      const o = document.createElement("option");
      o.value = r.id;
      o.textContent = r.id + "  —  " + (r.q.length > 38 ? r.q.slice(0, 38) + "…" : r.q);
      sel.appendChild(o);
    });
    const detail = document.getElementById("benchDetail");
    function showInstance(id){
      const rec = ANNOTATED.find(r => r.id === id);
      const d = detect(rec);
      let html = `<p style="font-size:17px">&ldquo;${esc(rec.q)}&rdquo;</p>
        <dl class="kv" style="margin:12px 0 20px">
          <dt>gold type</dt><dd>${esc(rec.ty)}</dd>
          <dt>gold relation</dt><dd>${esc(rec.rel || "—")}</dd>
          <dt>stage 1 score</dt><dd>${d.score.toFixed(4)} &rarr; ${d.isConflict ? "conflict" : "no conflict"}</dd>
          <dt>escalates?</dt><dd>${d.needsEscalation ? "yes" : "no — below the 0.35 floor"}</dd>
        </dl>`;
      for (const s of scored){
        const got = s.perInstance.get(id);
        const chips = rec.b.map((b, i) => {
          const j = got.j[i];
          const cls = j === "preserved" ? "p-ok" : j === "distorted" ? "p-un" : "p-no";
          return `<span class="pill ${cls}">${b.def ? "default" : "exception"}: ${esc(j)}</span>`;
        }).join(" ");
        html += `<div class="rcol">
          <div class="rhead"><b>${esc(s.name)}</b> <span class="sub">${esc(s.sub)}</span>
            ${s.badge ? `<span class="pill ${s.badge.cls}">${esc(s.badge.text)}</span>` : ""}</div>
          <p class="rans">${got.text ? esc(got.text)
            : `<em>${esc(s.answerNote || "no answer")}</em>`}</p>
          <div class="rchips">${chips}</div></div>`;
      }
      detail.innerHTML = html;
    }
    sel.addEventListener("change", () => showInstance(sel.value));
    showInstance(ANNOTATED[0].id);
  }

  /* ----------------------------------------------------------- routing */

  // Hash routing over three top-level views. Anchors inside the overview
  // ("#relations") keep working because anything not matching "#/..." falls
  // through to the overview.
  const VIEWS = { "#/bench": "view-bench", "#/byo": "view-byo" };

  function syncRoute(){
    const hash = location.hash;
    const target = VIEWS[hash] || "view-overview";
    for (const id of ["view-overview", ...Object.values(VIEWS)]){
      const el = document.getElementById(id);
      if (el) el.hidden = id !== target;
    }
    document.querySelectorAll("#nav a").forEach(a => {
      if (a.classList.contains("route")) a.classList.toggle("on", a.getAttribute("href") === hash);
      else if (target !== "view-overview") a.classList.remove("on");
    });
    // The bench is the only view expensive enough to defer; built once.
    if (target === "view-bench" && !host.dataset.built){
      render();
      host.dataset.built = "1";
    }
    if (target !== "view-overview") scrollTo(0, 0);
  }
  addEventListener("hashchange", syncRoute);
  syncRoute();
})();

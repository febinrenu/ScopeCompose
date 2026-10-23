/*  The bench: five resolvers, one metric, the 42 annotated instances.
 *  Every figure is computed at load by the project's own scoring code.
 *
 *  Three framing rules travel with the numbers and must never be dropped:
 *  the oracle column carries its upper-bound badge; the as-it-runs column is
 *  scored at the routing step and never reads gold branch text; and the
 *  concatenation column's score above gold is a defect in the metric, with
 *  its answer length shown beside it.
 */
import { annotated } from "../data/corpus";
import { detectorAnnotatedStats, detectorCorpusStats, proveInvariance, retrievalLoss, scoreResolvers, screen } from "../engine/pipeline";
import { ORDER_GATE } from "../data/facts";
import { $, esc, fx, pct } from "../ui/dom";

export function renderBench(host: HTMLElement) {
  const rows = annotated();
  const scored = scoreResolvers();
  const det = detectorAnnotatedStats();
  const corpusDet = detectorCorpusStats();
  const oracle = scored.find(s => s.key === "oracle")!;
  const asruns = scored.find(s => s.key === "asruns")!;
  const inv = proveInvariance();
  const W = 760, H = 96, pad = 28;
  const bars = det.scores.map((v, i) => {
    const x = pad + (i / Math.max(1, det.scores.length - 1)) * (W - pad * 2);
    const h = v * (H - 24);
    return `<rect x="${x.toFixed(1)}" y="${(H - 12 - h).toFixed(1)}" width="2.4" height="${Math.max(0.6, h).toFixed(1)}" rx="1.2"/>`;
  }).join("");
  const floorY = H - 12 - 0.35 * (H - 24);
  const badge = (b: { tone: string; text: string } | null) => b ? `<span class="chip ${b.tone}">${esc(b.text)}</span>` : "";

  host.innerHTML = `
  <div class="wrap">
    <a class="back" href="#act-06">← back to the story</a>
    <div class="eyebrow">The bench — five resolvers, one metric, ${rows.length} instances</div>
    <h2 class="display">The operator works.<br><em>The detector rarely fires.</em></h2>
    <p class="lede">Every figure below is computed in your browser, now, by the project’s own scoring code running on the real corpus. Nothing is typed into the page.</p>

    <div class="stats" style="margin-top:34px">
      <div class="stat"><div class="v">${fx(oracle.PR)}</div><div class="k">operator PR, given a conflict</div><div class="n">Gold routing and gold branches. An upper bound on the operator.</div></div>
      <div class="stat hatch"><div class="v">${det.fired} / ${det.n}</div><div class="k">conflicts stage 1 finds here</div><div class="n">Max score <b>${fx(det.max, 4)}</b> on this subset, against an escalation floor of 0.35.</div></div>
      <div class="stat"><div class="v">${fx(asruns.PR)}</div><div class="k">end-to-end PR, as it runs</div><div class="n">What the gap between the first two numbers costs.</div></div>
    </div>
    <p class="note" style="max-width:78ch"><b>That gap is the finding.</b> Over the whole ${corpusDet.n}-instance corpus the untrained stage-1 scorer reaches precision <b>${fx(corpusDet.precision)}</b> and recall <b>${fx(corpusDet.recall)}</b> — it fires on ${corpusDet.fired} pairs and finds ${corpusDet.tp} of ${corpusDet.positives} conflicts. Conservative, not random: when it fires it is usually right, and it almost never fires.</p>

    <section class="panel" style="margin-top:34px">
      <div class="panel-h"><h3>Stage-1 score, one bar per annotated instance</h3><span class="chip warn">${det.escalated} reach the escalation band</span></div>
      <div class="panel-b">
        <svg viewBox="0 0 ${W} ${H}" class="trace-svg" role="img" aria-label="Stage 1 scores for ${det.n} instances against the 0.35 escalation floor">
          <line x1="${pad}" y1="${floorY}" x2="${W - pad}" y2="${floorY}" class="floor" stroke-dasharray="4 4"/>
          <text x="${W - pad}" y="${floorY - 6}" class="flab" text-anchor="end">escalation floor 0.35</text>
          <g class="bars">${bars}</g>
          <line x1="${pad}" y1="${H - 12}" x2="${W - pad}" y2="${H - 12}" class="axis"/>
        </svg>
        <p class="note">The topicality multiplier <span class="ic">0.35 + 0.65×min(1, jaccard×3)</span> damps any pair whose passages are worded differently — which is exactly what a general rule and its carve-out look like.</p>
      </div>
    </section>

    <div class="tscroll" style="margin-top:34px">
      <table class="data"><thead><tr><th>Resolver</th><th class="num">PR</th><th class="num">SR</th><th class="num">DR</th><th class="num">chars</th><th>What it is</th></tr></thead>
      <tbody>${scored.map(s => `<tr data-key="${s.key}">
        <td><b>${esc(s.name)}</b><div class="sub">${esc(s.sub)}</div>${badge(s.badge)}</td>
        <td class="num"><b>${fx(s.PR)}</b><div class="sub">[${fx(s.ci.low, 2)}, ${fx(s.ci.high, 2)}]</div></td>
        <td class="num">${fx(s.SR)}</td>
        <td class="num">${s.judge ? '<span class="na">n/a</span>' : fx(s.DR)}</td>
        <td class="num">${s.judge ? '<span class="na">—</span>' : s.meanChars}</td>
        <td class="why">${esc(s.note)}</td></tr>`).join("")}</tbody></table>
    </div>
    <p class="note" style="max-width:78ch"><b>Read the concatenation row carefully.</b> Preservation Rate is <span class="ic">containment ≥ 0.6</span> with no length normalisation and no precision counterweight. HCR counts instances whose predicted branches cite a passage that does not exist, and SCR applies only to no-conflict instances, of which these ${rows.length} have none — so neither penalises verbosity. Pasting both passages verbatim therefore maximises PR and scores above the human-written gold answer. <b>The suite needs a precision term.</b> That is a defect in the metric, found by this bench.</p>

    <h3 class="h3" style="margin-top:56px">Before any resolver runs, retrieval has already destroyed the branch.</h3>
    <p class="lede" style="margin-top:0">The three implemented retrieval baselines, from <span class="ic">public/engine/baselines.js</span> — the same file the verifier checks. A branch is lost when the passage supporting it is dropped.</p>
    <div class="tscroll" style="margin-top:18px"><table class="data"><thead><tr><th>Retrieval strategy</th><th class="num">branches lost</th><th class="num">rate</th><th class="num">instances hit</th><th>What it is</th></tr></thead>
      <tbody>${retrievalLoss().map(r => `<tr><td><b>${esc(r.name)}</b><div class="sub">${esc(r.sub)}</div></td>
        <td class="num"><b>${r.lost}</b> / ${r.total}</td><td class="num">${pct(r.rate)}</td><td class="num">${r.affected} / ${r.instances}</td><td class="why">${esc(r.note)}</td></tr>`).join("")}</tbody></table></div>

    <h3 class="h3" style="margin-top:56px">Order invariance, proved here rather than quoted.</h3>
    <div class="stats">
      <div class="stat"><div class="v">${fx(inv.passRate, 4)}</div><div class="k">pass rate</div><div class="n">${inv.permutations} permutations, ${inv.violations} violations.</div></div>
      <div class="stat"><div class="v">${inv.nonPositional} / ${inv.permutations}</div><div class="k">non-positional roles</div><div class="n">Permutations where the default is <b>not</b> the first-listed passage. Without this the pass rate is vacuous.</div></div>
      <div class="stat"><div class="v">${esc(ORDER_GATE.value)}</div><div class="k">Python gate</div><div class="n">${esc(ORDER_GATE.detail)}. <span class="mono" style="text-transform:none">${esc(ORDER_GATE.source)}</span></div></div>
    </div>
    <p class="note" style="max-width:78ch">Each record holds two passages, so each admits exactly one non-identity permutation here. The Python gate analyses pairs without branch annotation too, and samples its own permutations — different denominators, same property.</p>

    <section class="panel" style="margin-top:34px">
      <div class="panel-h"><h3>Per instance</h3>
        <label class="field" style="min-width:min(100%,420px)"><span class="sr-only">Instance</span><select class="sel" id="benchInst">${rows.map(r =>
          `<option value="${esc(r.id)}">${esc(r.id)} — ${esc(r.q.length > 40 ? r.q.slice(0, 40) + "…" : r.q)}</option>`).join("")}</select></label></div>
      <div class="panel-b" id="benchDetail"></div>
    </section>
    <p class="note" style="max-width:78ch"><b>What these ${rows.length} are.</b> All are authored <span class="ic">wp1_*</span> probes — construction split, separation synthetic split. No mined instance carries branch annotation yet, so every number here is measured on constructed splits of single documents, not on the cross-document case the project is about.</p>
  </div>`;

  const sel = $<HTMLSelectElement>("#benchInst", host)!;
  const detail = $("#benchDetail", host)!;
  const show = (id: string) => {
    const rec = rows.find(r => r.id === id)!;
    const d = screen(rec);
    detail.innerHTML = `<p style="font-size:17px;margin:0">“${esc(rec.q)}”</p>
      <dl class="kv" style="margin:14px 0 18px"><dt>gold type</dt><dd>${esc(rec.ty)}</dd><dt>gold relation</dt><dd>${esc(rec.rel || "—")}</dd>
        <dt>stage-1 score</dt><dd>${fx(d.score, 4)} → ${d.isConflict ? "conflict" : "no conflict"}</dd>
        <dt>escalates?</dt><dd>${d.needsEscalation ? "yes" : "no"}</dd></dl>
      ${scored.map(s => {
        const got = s.perInstance.get(id)!;
        return `<div class="rcol"><div class="rhead"><b>${esc(s.name)}</b><span class="muted">${esc(s.sub)}</span>${badge(s.badge)}</div>
          <p class="rans">${got.text ? esc(got.text) : `<em>${esc(s.answerNote || "no answer")}</em>`}</p>
          <div class="rchips">${(rec.b || []).map((b, i) => {
            const j = got.j[i];
            return `<span class="chip ${j === "preserved" ? "ok" : j === "distorted" ? "warn" : "bad"}">${b.def ? "default" : "exception"}: ${esc(j)}</span>`;
          }).join("")}</div></div>`;
      }).join("")}`;
  };
  sel.addEventListener("change", () => show(sel.value));
  show(rows[0].id);
}

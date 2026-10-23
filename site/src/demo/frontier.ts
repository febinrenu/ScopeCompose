/*  Act 09: the honest frontier.
 *
 *  The decisive numbers are read from benchmark/data/decisive.json at build
 *  time; the detector gap and oracle ceiling are computed live. The point of
 *  this section is that a result against the hypothesis is shown as clearly
 *  as one for it.
 */
import { DECISIVE, JUDGE, LONG_CONTEXT, PROBE, BENCHMARK_TARGET, ORDER_GATE } from "../data/facts";
import { detectorCorpusStats, power, proveInvariance, scoreResolvers } from "../engine/pipeline";
import { annotated, goldBranchCount } from "../data/corpus";
import { $, esc, fx, pct } from "../ui/dom";

const sgn = (x: number) => (x > 0 ? "+" : x < 0 ? "−" : "") + pct(Math.abs(x), 1);

/** Confidence intervals on one axis: the two PRs on 0..1, and the difference around zero. */
function ciSvg(D: typeof DECISIVE): string {
  const W = 760, x = (v: number) => 20 + v * (W - 40);
  const dx = (v: number) => W / 2 + v * (W - 40);   // difference axis: -0.5..+0.5 around the centre
  return `<svg class="ci" viewBox="0 0 ${W} 150" role="img" aria-label="Pipeline PR ${fx(D.pipelinePR)} with 95% interval ${fx(D.pipelineCI[0])} to ${fx(D.pipelineCI[1])}; baseline ${fx(D.baselinePR)} with ${fx(D.baselineCI[0])} to ${fx(D.baselineCI[1])}; difference ${fx(D.diff)} with interval spanning zero.">
    <line class="axis" x1="20" y1="18" x2="${W - 20}" y2="18"/><text x="20" y="12">0</text><text x="${W - 20}" y="12" text-anchor="end">1.0 · preservation rate</text>
    <line class="bar-p draw" x1="${x(D.pipelineCI[0])}" y1="40" x2="${x(D.pipelineCI[1])}" y2="40"/><text x="${x(D.pipelineCI[1]) + 10}" y="44">pipeline</text>
    <line class="bar-b draw" x1="${x(D.baselineCI[0])}" y1="66" x2="${x(D.baselineCI[1])}" y2="66"/><text x="${x(D.baselineCI[1]) + 10}" y="70">baseline</text>
    <line class="axis" x1="20" y1="104" x2="${W - 20}" y2="104"/>
    <line class="zero" x1="${W / 2}" y1="92" x2="${W / 2}" y2="146"/><text x="${W / 2 + 6}" y="146">0 · no difference</text>
    <line class="bar-d draw" x1="${dx(D.diffCI[0])}" y1="124" x2="${dx(D.diffCI[1])}" y2="124"/><text x="${dx(D.diffCI[0])}" y="116">difference ${sgn(D.diff)}</text>
  </svg>`;
}

export function renderFrontier() {
  const host = $("#frontier");
  if (!host) return;
  const det = detectorCorpusStats();
  const scored = scoreResolvers();
  const oracle = scored.find(s => s.key === "oracle")!;
  const asruns = scored.find(s => s.key === "asruns")!;
  const inv = proveInvariance();
  const D = DECISIVE;
  const splits = Object.entries(D.splits).sort().map(([k, v]) => `${v} ${k}`).join(", ");
  const nAnn = annotated().length, nB = goldBranchCount();

  host.innerHTML = `
    <article class="panel">
      <div class="panel-h"><h3>The decisive experiment · n = ${D.instances} instances, ${D.branches} gold branches</h3>
        <span class="chip warn">underpowered — concludes nothing</span></div>
      <div class="panel-b">
        <p class="note" style="margin-top:0">The pipeline against <b>${esc(D.baselineName)}</b> — ${esc(LONG_CONTEXT.model)} given every passage and asked for structured branches. If the baseline matches the pipeline, the architectural claim is falsified.</p>
        <div class="versus">
          <div class="vs-side"><div class="k">pipeline · preservation rate</div><div class="v" data-scramble>${fx(D.pipelinePR, 3)}</div></div>
          <div class="vs-mid">vs</div>
          <div class="vs-side base"><div class="k">long-context baseline</div><div class="v" data-scramble>${fx(D.baselinePR, 3)}</div></div>
        </div>
        ${ciSvg(D)}
        <div><span class="under" id="underStamp">Underpowered</span></div>
        <div class="stats" style="margin-top:26px">
          <div class="stat"><div class="v" style="font-size:22px">Wilson 95%</div><div class="k">intervals</div>
            <div class="n">pipeline [${fx(D.pipelineCI[0], 3)}, ${fx(D.pipelineCI[1], 3)}] · baseline [${fx(D.baselineCI[0], 3)}, ${fx(D.baselineCI[1], 3)}]</div></div>
          <div class="stat hatch"><div class="v">${sgn(D.diff)}</div><div class="k">difference</div>
            <div class="n">[${sgn(D.diffCI[0])}, ${sgn(D.diffCI[1])}] · McNemar p = ${fx(D.mcnemarP, 3)} (${esc(D.mcnemarMethod)}, ${D.discordant} discordant). The interval spans zero: no conclusion either way.</div></div>
          <div class="stat hatch"><div class="v">n/a</div><div class="k">pipeline distortion · HCR · SCR</div>
            <div class="n">Not measurable at the routing step, where the pipeline arm is scored. The baseline's ${fx(D.baselineDistortion, 3)} distortion comes from a generated answer; the two are not comparable, so they are not compared.</div></div>
        </div>
        <p class="note"><b>This run read data that is now held out.</b> Its instances were picked by file order before splits existed and now straddle ${esc(splits)}. It is recorded rather than quietly re-run. Seed ${D.seed}, commit ${esc(D.gitSha)}.</p>
        <p class="note mono" style="text-transform:none;letter-spacing:0">${esc(D.source)}</p>
      </div>
    </article>

    <article class="panel">
      <div class="panel-h"><h3>The gap that matters most</h3><span class="chip bad">computed now, in this browser</span></div>
      <div class="panel-b">
        <div class="stats">
          <div class="stat"><div class="v">${fx(oracle.PR, 3)}</div><div class="k">operator PR, given a conflict</div>
            <div class="n">Gold routing and gold branches — an upper bound on composition (${oracle.preserved}/${oracle.n}).</div></div>
          <div class="stat hatch"><div class="v">${fx(det.recall, 3)}</div><div class="k">stage-1 recall, whole corpus</div>
            <div class="n">Precision ${fx(det.precision, 3)}: it fires on ${det.fired} of ${det.n} pairs and finds ${det.tp} of ${det.positives} conflicts (${det.conditionalFound} of ${det.conditional} conditional).</div></div>
          <div class="stat"><div class="v">${fx(asruns.PR, 3)}</div><div class="k">end-to-end PR, as it runs</div>
            <div class="n">What that gap costs on the ${nAnn} branch-bearing instances.</div></div>
        </div>
        <p class="note">The hard part turned out not to be composing branches without mangling them — the operator does that well. It is noticing there is a conflict at all, and the untrained stage-1 scorer is conservative to the point of near-silence. The bars behind this page are its real scores, one per instance, against both thresholds.</p>
      </div>
    </article>

    <article>
      <h3 class="h3">Open risks, with their status</h3>
      <ul class="risks">
        ${[
          ["Natural multi-source cases at scale", "Tier 1 yields ≈0.08 plausible pairs per curated pairing; zero mined instances carry branch annotation yet. The benchmark target is " + BENCHMARK_TARGET.total + " (" + BENCHMARK_TARGET.split + ").", "needs data", "bad"],
          ["Does contrastive probing beat direct extraction?", `Probe set of ${PROBE.cases} cases human-verified ${PROBE.verified}. Decision rule fixed in code before the run: ${PROBE.rule}. Live run pending.`, "pending", "warn"],
          ["Does the long-context baseline match the pipeline?", `One run at n = ${D.instances}, pointing toward the baseline, interval spanning zero. Needs a powered re-run on a defined split.`, "pending", "warn"],
          ["Does the metric judge agree with humans?", `LLM judge ${JUDGE.status}; the bar is agreement ≥ ${JUDGE.bar}. Until then judged numbers are not quoted.`, "implemented, unvalidated", "warn"],
          ["Can refinement vs opposed be annotated consistently?", "κ = 1.000 at pilot 9 after a written decision procedure — but redundant and opposed have two instances each.", "tested, thin", "warn"],
          ["Order invariance", `${ORDER_GATE.value} in the Python gate (${ORDER_GATE.detail}); ${fx(inv.passRate, 4)} here over ${inv.permutations} permutations, ${inv.nonPositional} non-positional.`, "tested", "ok"],
        ].map(([t, d, s, tone]) => `<li><div><b>${esc(t)}</b><p>${esc(d)}</p></div><span class="chip ${tone}">${esc(s)}</span></li>`).join("")}
      </ul>
    </article>

    <article class="panel">
      <div class="panel-h"><h3>How much data would be enough</h3><span class="chip">metrics/power.py → engines.js</span></div>
      <div class="panel-b cols2">
        <div>
          <div class="slider"><div class="sl"><label for="pEff">Difference to detect</label><span id="vEff"></span></div><input type="range" id="pEff" min="2" max="30" value="10" step="1"></div>
          <div class="slider"><div class="sl"><label for="pDis">Discordant rate</label><span id="vDis"></span></div><input type="range" id="pDis" min="10" max="70" value="35" step="1"></div>
          <p class="note">Observed in the decisive run: ${D.discordant} of ${D.branches} branches discordant (${pct(D.discordant / D.branches, 1)}). Only discordant branches inform McNemar.</p>
        </div>
        <div>
          <div class="stats"><div class="stat"><div class="v" id="pwB">—</div><div class="k">gold branches needed</div></div>
            <div class="stat"><div class="v" id="pwI">—</div><div class="k">instances needed</div></div></div>
          <p class="note" id="pwNote" aria-live="polite"></p>
        </div>
      </div>
    </article>

    <div class="claims">
      <div><h4>Can claim today</h4><ul>
        <li>A formal four-way scope relation with an executable routing table, and a benchmark schema that enforces it.</li>
        <li>Order invariance as a structural property, with a non-vacuity count beside the pass rate.</li>
        <li>Tier-1 scarcity with its mechanism — a result about where cross-document conditional conflicts live.</li>
        <li>A metric suite (PR / SR / HCR / SCR) with provenance recorded on every judged number.</li>
      </ul></div>
      <div><h4>Cannot claim yet</h4><ul>
        <li>That composition beats selection end to end: one run, n = ${D.instances}, interval spanning zero, pointing the other way.</li>
        <li>Anything about redundant or opposed at scale — two instances each.</li>
        <li>Any LLM-judged metric, until the judge is validated against human labels.</li>
        <li>Prevalence of conditional conflict in real retrieval. The corpus cannot support it, and the code refuses to compute it.</li>
      </ul></div>
    </div>
    <p class="note">Branch-scored figures use the ${nAnn} instances (${nB} gold branches) that carry branch annotation — all constructed (Tier 2) splits.</p>`;

  // the stamp lands, and the interval bars draw, when the panel is in view
  const stamp = $("#underStamp", host);
  const bars = Array.from(host.querySelectorAll<SVGLineElement>(".ci .draw"));
  // pathLength="1" lets the stroke draw on without measuring each line (a forced layout at boot)
  bars.forEach(b => { b.setAttribute("pathLength", "1"); b.style.strokeDasharray = "1"; b.style.strokeDashoffset = "1"; b.style.transition = "stroke-dashoffset 1.2s cubic-bezier(0.2,0.8,0.2,1)"; });
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    io.disconnect();
    bars.forEach((b, i) => setTimeout(() => (b.style.strokeDashoffset = "0"), i * 220));
    setTimeout(() => stamp?.classList.add("in"), 900);
  }), { threshold: 0.4 });
  const ci = host.querySelector(".ci"); if (ci) io.observe(ci);
  if (document.documentElement.classList.contains("rm")) { bars.forEach(b => (b.style.strokeDashoffset = "0")); stamp?.classList.add("in"); }

  const eff = $<HTMLInputElement>("#pEff", host)!, dis = $<HTMLInputElement>("#pDis", host)!;
  const upd = () => {
    const e = Number(eff.value) / 100, d = Number(dis.value) / 100;
    $("#vEff", host)!.textContent = eff.value + "%";
    $("#vDis", host)!.textContent = dis.value + "%";
    const r = power(e, d);
    const note = $("#pwNote", host)!;
    if (!r.branches) {
      $("#pwB", host)!.textContent = "n/a"; $("#pwI", host)!.textContent = "n/a";
      note.innerHTML = "<b>Not a sample-size problem.</b> An effect larger than the discordant rate cannot be expressed.";
    } else {
      $("#pwB", host)!.textContent = String(r.branches); $("#pwI", host)!.textContent = String(r.instances);
      note.innerHTML = `At α = 0.05 and 80% power: <b>${r.discordant}</b> discordant branches. The corpus has <b>${r.have}</b> annotated instances today${r.instances <= r.have ? " — enough for this effect." : `, so this needs <b>${r.instances - r.have}</b> more.`} A lower discordant rate needs fewer branches: the effect is then concentrated where the systems differ.`;
    }
  };
  eff.addEventListener("input", upd); dis.addEventListener("input", upd);
  upd();
}

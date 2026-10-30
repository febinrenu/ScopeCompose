/*  The technical architecture, without the cinema: one accessible SVG and
 *  the module map, for a reviewer who wants the system at a glance.
 */
import { STAGES } from "../demo/stations";
import { CONTRACT_VERSION } from "../data/facts";
import { esc } from "../ui/dom";
import { drawDiagram, introduce } from "./motion";

export function renderArchitecture(host: HTMLElement) {
  const W = 1180, nodeW = 150, nodeH = 112, y = 70;
  const xs = STAGES.map((_s, i) => 20 + i * 158 + (i >= 4 ? 30 : 0));
  const node = (s: (typeof STAGES)[number], x: number) => `
    <g class="node ${s.side.toLowerCase()}" transform="translate(${x},${y})">
      <rect width="${nodeW}" height="${nodeH}" rx="2"/>
      <text class="s" x="12" y="22">${s.id}</text>
      ${wrap(s.name, 17).map((l, i) => `<text class="t" x="12" y="${44 + i * 17}">${esc(l)}</text>`).join("")}
      ${wrap(s.code.replace(/ · /g, " "), 21).slice(0, 2).map((l, i) => `<text class="s" x="12" y="${86 + i * 13}">${esc(l)}</text>`).join("")}
    </g>`;
  const cx = (xs[3] + nodeW + xs[4]) / 2;
  host.innerHTML = `
  <div class="wrap arch">
    <a class="back" href="#act-05">← back to the story</a>
    <div class="eyebrow">Technical architecture</div>
    <h2 class="display">Two halves,<br><em>one frozen contract.</em></h2>
    <p class="lede">Member A decides what kind of disagreement it is and how the scopes relate. Member B extracts the conditions, composes and generates. They meet at one JSON record, version ${CONTRACT_VERSION}, and nowhere else.</p>
    <div class="tscroll" style="margin-top:36px">
      <svg viewBox="0 0 ${W} 300" role="img" aria-labelledby="archT archD" style="min-width:900px">
        <title id="archT">ScopeCompose pipeline</title>
        <desc id="archD">${esc(STAGES.map(s => `${s.id} ${s.name}`).join(", then "))}. The contract sits between A4 and B1.</desc>
        <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="rgba(241,239,232,0.4)"/></marker></defs>
        ${STAGES.map((s, i) => node(s, xs[i])).join("")}
        ${STAGES.slice(0, -1).map((_s, i) => i === 3 ? "" : `<path class="edge" d="M${xs[i] + nodeW},${y + nodeH / 2} L${xs[i + 1] - 4},${y + nodeH / 2}"/>`).join("")}
        <path class="edge" d="M${xs[3] + nodeW},${y + nodeH / 2} L${cx - 26},${y + nodeH / 2}"/>
        <path class="edge" d="M${cx + 26},${y + nodeH / 2} L${xs[4] - 4},${y + nodeH / 2}"/>
        <g class="contract"><rect x="${cx - 24}" y="${y - 22}" width="48" height="${nodeH + 44}" rx="2"/>
          <text x="${cx}" y="${y + nodeH + 44}" text-anchor="middle">contract v${CONTRACT_VERSION}</text></g>
        <text x="20" y="${y - 30}" class="s" style="fill:var(--cyan);font-family:var(--f-mono);font-size:11px">MEMBER A · FRONT END</text>
        <text x="${xs[4]}" y="${y - 30}" class="s" style="fill:var(--amber);font-family:var(--f-mono);font-size:11px">MEMBER B · BACK END</text>
        <g transform="translate(20,${y + nodeH + 74})" style="font-family:var(--f-mono);font-size:11px;fill:var(--muted)">
          <text>routing · conditional: refinement / disjoint → compose · redundant → merge · opposed → select · nested / crossed flags → select</text>
          <text y="18">routing · other types: factual / temporal / opinion → prior work · no_conflict → pass through</text>
        </g>
      </svg>
    </div>
    <div class="arch-legend">
      ${STAGES.map(s => `<div><h4>${s.id} · ${esc(s.name)}</h4><p>${esc(s.body)}</p><code>${esc(s.code)}</code></div>`).join("")}
      <div><h4>The contract</h4><p>One JSON record per query: passages, conflict pairs with type and scope relation, the construction tier, and the multi-exception flags B sets. Any schema change bumps the version and the mock generator in the same commit.</p><code>contract/models.py · contract/schema/ · contract/mock.py</code></div>
      <div><h4>What runs in this browser</h4><p>The scope algebra, four-way relation, routing, stage-1 detector (with its offline heuristic NLI), composition operator, renderer, faithfulness check, textual judge, statistics and the retrieval baselines — JavaScript ports verified against the Python on every build.</p><code>site/public/engine/*.js · site/tools/verify_engines.js</code></div>
    </div>
  </div>`;
  introduce(host);
  const svg = host.querySelector<SVGSVGElement>("svg");
  if (svg) drawDiagram(svg);
}

function wrap(s: string, n: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const w of s.split(/\s+/)) {
    if ((line + " " + w).trim().length > n && line) { out.push(line); line = w; }
    else line = (line + " " + w).trim();
  }
  if (line) out.push(line);
  return out;
}

/*  The story's numbers, computed.
 *
 *  Acts 02, 03 and 07 tell the story with the illustrative fee example. The
 *  example is illustrative; the decisions made on it are not -- which passage
 *  selection keeps, how each branch is judged, and the composed answer all
 *  come from the verified engines, run here on that input.
 */
import { FEE } from "../data/examples";
import { audit, runChain, selectionOf } from "../engine/pipeline";
import { byId, PROTAGONIST_ID } from "../data/corpus";
import { $, esc, fx, human } from "../ui/dom";
import type { Branch } from "../engine/types";

/** "(per p0)" -> a citation chip; exception scopes -> amber. */
export function markAnswer(answer: string, branches: Branch[]): string {
  let html = esc(answer);
  for (const b of branches) {
    if (b.def || !b.desc) continue;
    const d = esc(b.desc);
    html = html.replace(d, `<span class="exc">${d}</span>`);
  }
  return html.replace(/\(per (p\d+)\)/g, '<span class="cite" title="supported by passage $1">$1</span>');
}

export interface StoryFacts {
  winner: string | null;
  winnerType: string;
  loserType: string;
  selectionAnswer: string;
  composedAnswer: string;
  srSelection: number;
  srComposed: number;
}

export function renderStory(): StoryFacts {
  const sel = selectionOf(FEE, FEE.b);
  const chain = runChain(FEE, FEE.b);
  const selAudit = audit(sel.answer, FEE.b);
  const compAudit = audit(chain.answer, FEE.b);
  const winner = FEE.p.find(p => p.id === sel.winner);
  const loser = FEE.p.find(p => p.id !== sel.winner);

  const note = $("#selWinnerNote");
  if (note) note.textContent =
    `mostCredible() keeps ${sel.winner} (${human(winner?.st)}) over ${loser?.id} (${human(loser?.st)}) — the project's own selection fallback, run on the illustrative passages.`;

  const ans = $("#fvAnswer");
  if (ans) ans.innerHTML = "“" + markAnswer(sel.answer, FEE.b) + "”";
  const list = $("#fvAudit");
  if (list) list.innerHTML = FEE.b.map((b, i) => {
    const j = selAudit.judgements[i];
    const tone = j === "preserved" ? "ok" : j === "distorted" ? "warn" : "bad";
    return `<li class="${j}"><span class="chip ${tone}">${esc(j)}</span>
      <span class="out">${b.def ? "Default" : "If " + esc(b.desc)} — ${esc(b.out)}</span></li>`;
  }).join("");
  const sr = $("#fvSR");
  if (sr) sr.textContent = fx(selAudit.rates.SR, 2);

  const comp = $("#composedFee");
  if (comp) comp.innerHTML = markAnswer(chain.answer, chain.composed.branches);
  const trace = $("#composedTrace");
  if (trace) trace.innerHTML = chain.composed.branches.map(b => {
    const p = FEE.p.find(x => x.id === b.sup);
    return `<li><span class="chip ${b.def ? "ok" : "warn"}">${esc(b.sup || "—")}</span>
      <span>${b.def ? "Rule" : "Exception (" + esc(b.desc) + ")"} — from “${esc(p?.t || "")}”</span></li>`;
  }).join("") + `<li><span class="chip">SR ${fx(compAudit.rates.SR, 2)}</span><span>Suppression Rate of this answer, same judge. Faithfulness check: ${chain.faith.ok ? "passes" : "fails"}.</span></li>`;

  // The same shape in the benchmark: a real constructed-tier record.
  const rec = byId(PROTAGONIST_ID);
  if (rec && rec.b) {
    const real = runChain(rec, rec.b);
    const el = $("#composedReal");
    if (el) el.innerHTML = markAnswer(real.answer, real.composed.branches);
    const prov = $("#composedRealProv");
    if (prov) prov.textContent = `${rec.id} · ${rec.con === "split" ? "Tier 2, constructed" : "Tier 1, natural"} · ${human(rec.sep)} · gold relation ${rec.rel} · routed: ${real.action}`;
  }

  return {
    winner: sel.winner,
    winnerType: human(winner?.st),
    loserType: human(loser?.st),
    selectionAnswer: sel.answer,
    composedAnswer: chain.answer,
    srSelection: selAudit.rates.SR,
    srComposed: compAudit.rates.SR,
  };
}

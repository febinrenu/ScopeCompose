/*  Act 05: the pipeline, with its real module names. */
import { $, esc, fx } from "../ui/dom";
import { screen } from "../engine/pipeline";
import { FEE } from "../data/examples";

export const STAGES = [
  { id: "A1", side: "A", name: "Hybrid retrieval",
    body: "BM25 and dense Contriever retrieval fused by reciprocal-rank fusion, returning the top K = 5 passages.",
    code: "retrieval/sparse.py · dense.py · fusion.py" },
  { id: "A2", side: "A", name: "Pairwise conflict detection",
    body: "Stage 1 scores every pair locally: a DeBERTa-v3-base NLI cross-encoder plus learned lexical features. Pairs in the uncertain band 0.35–0.75 escalate to the configured LLM provider.",
    code: "detection/stage1.py · stage2.py · pipeline.py" },
  { id: "A3", side: "A", name: "Five-class typing",
    body: "no_conflict, factual, temporal, opinion, conditional. Three detector architectures are compared: lexical + NLI, fine-tuned DeBERTa-v3, and structured entailment.",
    code: "contract/models.py · detection/variants/" },
  { id: "A4", side: "A", name: "Four-way scope relation",
    body: "Applicability overlap and outcome agreement are decided separately, then combined. Roles come from the relation, never from retrieval order — gated by the order-invariance check.",
    code: "scope/attributes.py · relation.py · order_invariance.py" },
  { id: "B1", side: "B", name: "Contrastive scope probing",
    body: "Asks what changes between two passages, to surface implicit conditions a direct ask misses. The grounding gate keeps a condition only if its passage entails it.",
    code: "extraction/probing.py" },
  { id: "B2", side: "B", name: "Composition operator",
    body: "Refinement and disjoint compose, redundant merges, opposed falls back to selection. Nested or crossed exceptions are flagged and routed to selection, never silently composed.",
    code: "composition/operator.py · contract/routing.py" },
  { id: "B3", side: "B", name: "Conflict-aware generation",
    body: "A template renderer by default — it cannot invent — citing the passage behind every branch, then a clause-level faithfulness check on its own output.",
    code: "generation/scoped_answer.py" },
] as const;

export function renderStations(): HTMLElement[] {
  const host = $("#stations");
  if (!host) return [];
  const live = screen(FEE);
  host.innerHTML = STAGES.map((s, i) => `
    <li class="tcard" data-side="${s.side}">
      <span class="tnum" aria-hidden="true">${s.id}</span>
      <span class="mono tk">${s.side === "A" ? "Member A · front end" : "Member B · back end"} · ${i + 1} of 7</span>
      <h3>${esc(s.name)}</h3>
      <p>${esc(s.body)}${s.id === "A2"
        ? ` <span class="live">In this browser: <b data-scramble>${fx(live.score, 3)}</b> on the fee pair, with the offline heuristic NLI.</span>` : ""}
        ${s.id === "A4" ? ` <span class="live">Next: the frozen JSON contract, v1.2.0 — the only thing both halves agree on.</span>` : ""}</p>
      <code>${esc(s.code)}</code>
    </li>`).join("");
  return Array.from(host.children) as HTMLElement[];
}

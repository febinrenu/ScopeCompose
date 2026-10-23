/*  The retrieval-side baselines.
 *
 *  Port of baselines/retrieval_side.py (StandardRAG, RerankTop1, NLIFilter).
 *  These act BEFORE any resolver, by deciding which passages survive, so they
 *  are scored on branch loss: a branch is lost iff its supporting passage was
 *  dropped.
 *
 *  This used to exist twice -- once inline in tools/verify_engines.js and once
 *  in the bench -- so the copy the site ran was not the copy the verifier
 *  checked. It now lives here only, and both load this file.
 *
 *  Leans on heuristicNLI from detector.js, so it loads after it.
 */
"use strict";

const RETRIEVAL_BASELINES = {
  standard_rag: rec => rec.p.map(p => p.id),
  rerank_top1: rec => [rec.p[0].id],
  nli_filter: rec => {
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
  },
};

/** run_baselines.py's rule: a branch with no supporting passage always survives. */
function branchesLost(rec, kept){
  const keep = new Set(kept);
  return (rec.b || []).filter(b => b.sup && !keep.has(b.sup)).map(b => b.id).sort();
}

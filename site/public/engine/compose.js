/*  The composition operator and scoped-answer renderer.
 *
 *  Port of composition/operator.py and generation/scoped_answer.py. Both are
 *  entirely deterministic in the Python -- zero model calls -- so this is a
 *  faithful port rather than an approximation.
 *
 *  IMPORTANT, and the site must say it wherever this runs: the branches fed
 *  in here come from the GOLD annotation, because B1 (extraction/probing.py)
 *  has no offline path. Everything this produces is therefore an upper bound
 *  on what the operator could do given perfect extraction and perfect
 *  routing. It is not a measurement of the system end to end.
 */
"use strict";

const RESOLUTION = {
  COMPOSED: "composed", MERGED: "merged", SELECTED: "selected",
  PRIOR_WORK: "prior_work", PASS_THROUGH: "pass_through",
};

/**
 * Do two exception branches interact in a way outside first-order scope?
 *
 * Order matters: DISJOINT is checked before outcomes, because when no case
 * falls under both, comparing outcomes is vacuous. UNKNOWN is never flagged
 * -- flagging on ignorance would inflate the reported nested/crossed rates.
 */
/** Corpus branches carry `def`/`at` directly; compare() wants that shape. */
const _app = b => ({ def: !!b.def, at: b.at });

function classifyExceptionPair(a, b){
  const rel = compare(_app(a), _app(b));
  if (rel === REL.DISJOINT)
    return [null, "exception scopes do not intersect; both stand"];
  if (rel === REL.SUBSET || rel === REL.SUPERSET){
    if (outcomes_match(a.out, b.out))
      return [null, "nested scopes but the same outcome; a restatement"];
    return ["nested", "one exception is itself carved back by another -- second-order structure"];
  }
  if (rel === REL.EQUAL){
    if (outcomes_match(a.out, b.out))
      return [null, "same scope, same outcome; a restatement"];
    return ["crossed", "identical scope with disagreeing outcomes"];
  }
  if (rel === REL.OVERLAPPING){
    if (outcomes_match(a.out, b.out))
      return [null, "overlapping scopes agreeing on the overlap; compatible"];
    return ["crossed", "two exceptions can co-occur and disagree, neither taking precedence by scope"];
  }
  return [null, "scope relation undecidable from typed attributes; not flagged"];
}

const _CREDIBILITY = ["official_policy", "government_guidance", "product_terms",
                      "news", "third_party", "user_generated"];

/** Deterministic selection fallback: credibility rank, then earliest date. */
function mostCredible(passages){
  let best = null, bestKey = null;
  for (const p of passages){
    const idx = _CREDIBILITY.indexOf(p.st || "unknown");
    const key = [idx < 0 ? _CREDIBILITY.length : idx, p.dt || ""];
    if (!bestKey || key[0] < bestKey[0] || (key[0] === bestKey[0] && key[1] < bestKey[1])){
      best = p; bestKey = key;
    }
  }
  return best ? best.id : null;
}

/**
 * Compose gold branches under a routing decision.
 *
 * `action` comes from routeFor(type, relation) -- in the bench that is the
 * GOLD type and relation, which is what makes this the oracle arm.
 */
function composeBranches(record, branches, action){
  const defaults = branches.filter(b => b.def);
  const exceptions = branches.filter(b => !b.def);
  const flags = { nested: false, crossed: false };
  let flagReason = "";

  if (action === "pass_through"){
    // Deliberately the default alone, not every branch: keeping all of them
    // here would inflate Preservation Rate on instances the system decided
    // had no conflict at all.
    const kept = defaults.length ? [defaults[0]] : branches.slice(0, 1);
    return { resolution: RESOLUTION.PASS_THROUGH, branches: kept, flags,
             reason: "no conflict detected" };
  }
  if (action === "select"){
    const winner = mostCredible(record.p);
    const survived = branches.filter(b => b.sup === winner);
    return { resolution: RESOLUTION.SELECTED,
             branches: survived.length ? survived : branches.slice(0, 1), flags,
             reason: "a real contradiction; selection is the honest answer",
             selectedPassage: winner };
  }
  if (action === "prior_work"){
    return { resolution: RESOLUTION.PRIOR_WORK, branches, flags,
             reason: "handled by the published strategy for that conflict class" };
  }

  let kept = branches;
  let merged = 0;
  if (action === "merge"){
    const def = defaults[0];
    if (def){
      kept = [def, ...branches.filter(b => !b.def && !outcomes_match(def.out, b.out))];
      merged = branches.length - kept.length;
    }
    return { resolution: RESOLUTION.MERGED, branches: kept, flags,
             reason: "the branches agree throughout their overlap -- a restatement, not a conflict",
             mergedPairs: merged };
  }

  // compose: check the surviving exceptions against each other
  const exc = kept.filter(b => !b.def);
  for (let i = 0; i < exc.length; i++){
    for (let j = i + 1; j < exc.length; j++){
      const [flag, why] = classifyExceptionPair(exc[i], exc[j]);
      if (flag === "nested" && !flags.nested){ flags.nested = true; flagReason = why; }
      else if (flag === "crossed" && !flags.crossed){ flags.crossed = true; flagReason = flagReason || why; }
    }
  }
  if (flags.nested || flags.crossed){
    const name = flags.nested ? "nested" : "crossed";
    const winner = mostCredible(record.p);
    const survived = kept.filter(b => b.sup === winner);
    return { resolution: RESOLUTION.SELECTED,
             branches: survived.length ? survived : kept.slice(0, 1), flags,
             reason: `${name}: ${flagReason}`, selectedPassage: winner };
  }
  return { resolution: RESOLUTION.COMPOSED, branches: kept, flags,
           reason: "a valid exception narrows the default; both branches are preserved" };
}

/* ------------------------------------------------------- rendering ------ */

const _clause = t => String(t || "").trim().replace(/\.$/, "");
const _sentence = t => String(t || "").charAt(0).toUpperCase() + String(t || "").slice(1);

/** Deterministic template. The Python default is use_llm=False: free, and
 *  incapable of inventing anything. */
function renderTemplate(composed){
  const branches = composed.branches || [];
  if (!branches.length) return "No answer could be resolved from the retrieved passages.";

  const cite = b => b.sup ? ` (per ${b.sup})` : "";

  if (composed.resolution === RESOLUTION.PASS_THROUGH){
    const b = branches[0];
    return `${_sentence(_clause(b.out))}${cite(b)}.`;
  }
  if (composed.resolution === RESOLUTION.SELECTED){
    const b = branches[0];
    let flag = "";
    if (composed.flags.nested || composed.flags.crossed){
      const which = composed.flags.nested ? "nested" : "crossed";
      flag = ` The exceptions here interact (${which}), which is outside what this system `
           + `resolves, so a single source was used.`;
    }
    return `${_sentence(_clause(b.out))}${cite(b)}.${flag}`;
  }

  const parts = [];
  const def = branches.find(b => b.def);
  if (def) parts.push(`By default, ${_clause(def.out)}${cite(def)}`);
  for (const b of branches.filter(x => !x.def)){
    const scope = b.desc || b.cond || "certain cases";
    parts.push(`for ${scope}, ${_clause(b.out)}${cite(b)}`);
  }
  if (!parts.length) return "No answer could be resolved from the retrieved passages.";
  if (parts.length === 1) return _sentence(parts[0]) + ".";
  return _sentence(parts[0]) + "; " + parts.slice(1).join("; ") + ".";
}

/* ---------------------------------------------------- faithfulness ------ */

// The digit lookarounds are load-bearing. A plain [;.] split tore
// "a 2.75% fee applies" into "a 2" / "75% fee applies", so every answer
// quoting a decimal was reported as having dropped its own branches.
const _CLAUSE_RE = /;\s*|(?<!\d)\.(?!\d)\s*/;
// And this lookbehind stops the 0 in "(per p0)" reading as an invented
// figure -- which flagged template answers, which cannot invent anything.
const _FAITH_NUM_RE = /(?<![A-Za-z0-9])\d+(?:[.,]\d+)?%?/g;

const _ADJUDICATION_CUES = ["more reliable", "more authoritative", "more credible",
  "less reliable", "should be trusted", "takes precedence", "outweighs",
  "more up to date", "the correct answer is", "we recommend relying"];

function checkFaithfulness(text, branches){
  const low = String(text || "").toLowerCase();
  const clauses = String(text || "").split(_CLAUSE_RE).filter(c => c.trim());
  const report = { missingBranches: [], missingAttribution: [], unsupportedNumbers: [] };

  for (const b of branches){
    if (!clauses.some(c => outcomes_match(b.out, c))) report.missingBranches.push(b.id);
    else if (b.sup && !low.includes(String(b.sup).toLowerCase())) report.missingAttribution.push(b.id);
  }

  const allowed = new Set();
  for (const b of branches){
    for (const field of [b.out, b.desc, b.cond]){
      for (const n of String(field || "").match(_FAITH_NUM_RE) || []) allowed.add(n);
    }
  }
  for (const n of String(text || "").match(_FAITH_NUM_RE) || []){
    if (!allowed.has(n)) report.unsupportedNumbers.push(n);
  }

  report.adjudicated = _ADJUDICATION_CUES.some(c => low.includes(c));
  // missingAttribution is reported but does NOT fail the check.
  report.ok = !(report.missingBranches.length || report.unsupportedNumbers.length
                || report.adjudicated);
  return report;
}

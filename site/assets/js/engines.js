/*  ScopeCompose - engine layer
 *
 *  Pure logic, no DOM. Every function here is a direct port of the Python
 *  implementation it names, and is checked against it by tools/verify_engines.js
 *  over the whole corpus. If a port drifts, that check fails.
 *
 *  Loaded as a classic script before app.js, so these declarations are
 *  visible to it.
 */
"use strict";

/* ======================================================================
   ENGINE 1 — set algebra over typed applicability attributes.
   Direct port of scope/attributes.py. An applicability set is a
   CONJUNCTION of per-attribute constraints; unmentioned dimensions are
   unconstrained, so more constraints means a SMALLER set and the default
   branch (no constraints) is the universe U.
   ====================================================================== */
const REL = {SUBSET:"subset", SUPERSET:"superset", EQUAL:"equal",
             DISJOINT:"disjoint", OVERLAPPING:"overlapping", UNKNOWN:"unknown"};

function constraintOf(a){
  const c = {kind:a.k, values:null, lo:null, hi:null};
  if (a.k === "categorical" || a.k === "ordered_tier"){
    c.values = new Set((a.v || []).map(v => String(v).trim().toLowerCase()));
  } else if (a.k === "boolean"){
    c.values = new Set([(a.b === undefined || a.b === null ? "none" : String(a.b)).toLowerCase()]);
  } else if (a.k === "numeric_range"){
    c.lo = (a.lo === undefined || a.lo === null) ? -Infinity : a.lo;
    c.hi = (a.hi === undefined || a.hi === null) ?  Infinity : a.hi;
  }
  return c;
}
const decidable = c => c.kind !== "free_text";

function conDisjoint(a, b){
  if (!decidable(a) || !decidable(b) || a.kind !== b.kind) return false;
  if (a.values && b.values){ for (const v of a.values) if (b.values.has(v)) return false; return true; }
  if (a.lo !== null && b.lo !== null) return a.hi < b.lo || b.hi < a.lo;
  return false;
}
function conSubset(a, b){
  if (!decidable(a) || !decidable(b) || a.kind !== b.kind) return false;
  if (a.values && b.values){ for (const v of a.values) if (!b.values.has(v)) return false; return true; }
  if (a.lo !== null && b.lo !== null) return b.lo <= a.lo && a.hi <= b.hi;
  return false;
}
const constraintsOf = app => {
  const m = new Map();
  (app.at || []).forEach(a => m.set(String(a.n).trim().toLowerCase(), constraintOf(a)));
  return m;
};

/** compare(a, b) — how applicability set `a` relates to `b`. */
function compare(a, b){
  if (a.def && b.def) return REL.EQUAL;
  if (a.def) return REL.SUPERSET;
  if (b.def) return REL.SUBSET;

  const ca = constraintsOf(a), cb = constraintsOf(b);
  if (!ca.size || !cb.size) return REL.UNKNOWN;

  // Disjointness first: one incompatible dimension empties the whole conjunction.
  for (const [name, x] of ca){ const y = cb.get(name); if (y && conDisjoint(x, y)) return REL.DISJOINT; }

  for (const c of [...ca.values(), ...cb.values()]) if (!decidable(c)) return REL.UNKNOWN;

  let shared = 0; for (const name of ca.keys()) if (cb.has(name)) shared++;
  // Entirely different dimensions: they intersect, and neither contains the
  // other, because each leaves the other's dimension unconstrained.
  if (!shared) return REL.OVERLAPPING;

  let aInB = true; for (const [n, y] of cb){ const x = ca.get(n); if (!x || !conSubset(x, y)){ aInB = false; break; } }
  let bInA = true; for (const [n, x] of ca){ const y = cb.get(n); if (!y || !conSubset(y, x)){ bInA = false; break; } }

  if (aInB && bInA) return REL.EQUAL;
  if (aInB) return REL.SUBSET;
  if (bInA) return REL.SUPERSET;
  return REL.OVERLAPPING;
}

/* ---- the four-way relation: ScopeAnalyser._combine ---- */
function combine(setRel, outcomesAgree){
  if (setRel === REL.DISJOINT)
    return ["disjoint", "the applicability sets do not intersect, so no case falls under both and outcome comparison is vacuous"];
  if (setRel === REL.UNKNOWN)
    return ["opposed", "the scope relation could not be established from attributes; falling back to selection rather than composing on unestablished scope"];
  if (outcomesAgree)
    return ["redundant", "the scopes overlap or nest but the outcomes agree throughout the overlap — a restatement, not an exception"];
  if (setRel === REL.SUBSET || setRel === REL.SUPERSET)
    return ["refinement", "one applicability set is strictly inside the other and the outcomes differ there — a genuine exception that narrows the default without contesting it"];
  if (setRel === REL.EQUAL)
    return ["opposed", "the two branches cover exactly the same cases and disagree about the outcome — a direct contradiction"];
  return ["opposed", "the applicability sets overlap, neither contains the other, and the outcomes disagree on the overlap — a real contradiction"];
}

/* ---- routing: contract/routing.py::route ---- */
const ROUTE = {refinement:["compose","both branches are true and both are preserved"],
               disjoint:["compose","no case falls under both; compose trivially"],
               redundant:["merge","a restatement, not a conflict — composing as two branches would be wrong"],
               opposed:["select","a real contradiction; selection is the honest answer, and is reported as one"]};

/* ---- membership: does a reader fall inside this branch? ---- */
function satisfies(app, persona){
  if (app.def) return true;                       // the default branch is U
  for (const a of (app.at || [])){
    const dim = String(a.n).trim().toLowerCase();
    const got = persona[dim];
    if (got === undefined || got === null || got === "__other__") return false;
    const c = constraintOf(a);
    if (c.values){ if (!c.values.has(String(got).trim().toLowerCase())) return false; }
    else if (c.lo !== null){ const n = Number(got); if (!(n >= c.lo && n <= c.hi)) return false; }
    else return false;                            // free text: cannot verify
  }
  return true;
}

function invNorm(p){
  const a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,
           1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00];
  const b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,
           6.680131188771972e+01,-1.328068155288572e+01];
  const c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,
           -2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00];
  const d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00];
  const plow=0.02425, phigh=1-plow;
  let q, r;
  if (p < plow){ q = Math.sqrt(-2*Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
  if (p > phigh){ q = Math.sqrt(-2*Math.log(1-p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
  q = p-0.5; r = q*q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}
function pairedSize(effect, discordantRate, branchesPerInstance){
  const p = 0.5 + effect / (2 * discordantRate);
  if (p >= 1.0) return {branches:0, instances:0, discordant:0};
  const za = invNorm(0.975), zb = invNorm(0.80);
  const nd = Math.ceil(Math.pow(za*0.5 + zb*Math.sqrt(p*(1-p)), 2) / Math.pow(p-0.5, 2));
  const nb = Math.ceil(nd / discordantRate);
  return {branches:nb, instances:Math.ceil(nb / branchesPerInstance), discordant:nd};
}

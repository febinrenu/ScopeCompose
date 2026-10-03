/*  Check the browser engines against the Python they port.
 *
 *      python site/tools/make_reference.py    # record Python's answers
 *      node   site/tools/verify_engines.js    # replay them through the JS
 *
 *  Exits non-zero on any disagreement. The deploy workflow runs both, and
 *  diffs the regenerated reference, so neither a drifted port nor a stale
 *  reference can ship.
 *
 *  Four families, because one of them used to hide a hole:
 *
 *    compare         every ordered branch pair, both directions
 *    combine         the FULL truth table, including arms the corpus cannot
 *                    reach -- replaying only corpus cases left `redundant`
 *                    untested while reporting a confident pass
 *    route           every (type, relation) the table can be asked about
 *    outcomes_agree  NOT a port check. Python decides this with an NLI model
 *                    the browser cannot run. Recorded so the divergence
 *                    between it and the browser's lexical stand-in is a
 *                    reported number rather than a silent difference.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const SITE = path.resolve(__dirname, "..");
const load = p => fs.readFileSync(path.join(SITE, p), "utf8");

// engines.js is a classic script, not a module -- evaluate it and lift the
// symbols out rather than maintaining a parallel export list.
const { compare, combine, routeFor } = new Function(
  load("assets/js/engines.js") + "\nreturn { compare, combine, routeFor };"
)();

const { screenPair } = new Function(
  load("assets/js/detector.js") + "\nreturn { screenPair };"
)();

const { judgeTextual } = new Function(
  load("assets/js/match.js") + "\nreturn { judgeTextual };"
)();

const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

const refPath = path.join(__dirname, "compare_reference.json");
if (!fs.existsSync(refPath)) {
  console.error("No reference. Run: python site/tools/make_reference.py");
  process.exit(2);
}
const ref = JSON.parse(fs.readFileSync(refPath, "utf8"));
const corpus = JSON.parse(load("data/corpus.json"));
const byId = new Map(corpus.map(r => [r.id, r]));

let failures = 0;
const fail = msg => { failures++; if (failures <= 20) console.error("  " + msg); };

/* --- compare ------------------------------------------------------------ */
let ok = 0;
for (const c of ref.compare) {
  const rec = byId.get(c.inst);
  if (!rec) { fail(`MISSING instance ${c.inst}`); continue; }
  const a = rec.b.find(x => x.id === c.a), b = rec.b.find(x => x.id === c.b);
  const got = compare({ def: a.def, at: a.at }, { def: b.def, at: b.at });
  if (got === c.rel) ok++;
  else fail(`compare ${c.inst} ${c.a}->${c.b}: python=${c.rel} js=${got}`);
}
console.log(`compare         ${ok}/${ref.compare.length} match Python`);

/* --- combine: the full truth table -------------------------------------- */
let okc = 0;
for (const c of ref.combine) {
  const [got] = combine(c.set, c.agree);
  if (got === c.rel) okc++;
  else fail(`combine (${c.set}, agree=${c.agree}): python=${c.rel} js=${got}`);
}
const arms = [...new Set(ref.combine.map(c => c.rel))].sort().join(", ");
console.log(`combine         ${okc}/${ref.combine.length} truth-table rows  [${arms}]`);

/* --- route -------------------------------------------------------------- */
let okr = 0;
for (const c of ref.route) {
  let got;
  try { got = routeFor(c.type, c.rel); }
  catch (e) { got = "THREW: " + e.message; }
  if (got === c.action) okr++;
  else fail(`route (${c.type}, ${c.rel}): python=${c.action} js=${got}`);
}
console.log(`route           ${okr}/${ref.route.length} (type, relation) combinations`);

/* --- outcomes_agree: a disclosure, not a check -------------------------- */
const oa = ref.outcomes_agree || [];
let lexAgree = 0, pyAgree = 0, diverged = 0;
for (const c of oa) {
  const rec = byId.get(c.inst);
  if (!rec || (rec.b || []).length !== 2) continue;
  const [x, y] = rec.b;
  const lexical = x.out.trim().toLowerCase() === y.out.trim().toLowerCase();
  if (lexical !== c.lexical) fail(`lexical outcomes_agree ${c.inst}: ref=${c.lexical} js=${lexical}`);
  lexAgree += lexical; pyAgree += c.agree;
  if (lexical !== c.agree) diverged++;
}
console.log(`outcomes_agree  ${oa.length} instances: python agrees on ${pyAgree}, `
          + `browser's lexical test on ${lexAgree}`);
if (diverged) {
  console.log(`                ${diverged} divergence(s) -- EXPECTED. Python uses an NLI model`);
  console.log(`                the browser cannot run. The site must label its own`);
  console.log(`                outcome determination as a lexical approximation.`);
}

/* --- detector: stage 1 exactly as it runs offline ----------------------- */
let okd = 0, fired = 0, maxScore = 0;
for (const c of ref.detector || []) {
  const rec = byId.get(c.inst);
  if (!rec) { fail(`detector: missing instance ${c.inst}`); continue; }
  const pi = rec.p.find(x => x.id === c.i), pj = rec.p.find(x => x.id === c.j);
  const got = screenPair(pi.t, pj.t);
  maxScore = Math.max(maxScore, got.score);
  fired += got.isConflict ? 1 : 0;
  let bad = null;
  if (!near(got.score, c.score)) bad = `score python=${c.score} js=${got.score}`;
  else if (got.isConflict !== c.conflict) bad = `conflict python=${c.conflict} js=${got.isConflict}`;
  else if (got.needsEscalation !== c.escalate) bad = `escalate python=${c.escalate} js=${got.needsEscalation}`;
  else for (const [k, v] of Object.entries(c.f)) {
    if (!near(got.features[k], v)) { bad = `feature ${k} python=${v} js=${got.features[k]}`; break; }
  }
  if (bad) fail(`detector ${c.inst}: ${bad}`); else okd++;
}
console.log(`detector        ${okd}/${(ref.detector || []).length} pairs `
          + `(score, verdict, escalation, all 17 features)`);
console.log(`                fires on ${fired}, max score ${maxScore.toFixed(4)}`);

/* --- textual judging ---------------------------------------------------- */
let okt = 0;
for (const c of ref.textual || []) {
  const rec = byId.get(c.inst);
  if (!rec) { fail(`textual: missing instance ${c.inst}`); continue; }
  const text = c.sys === "selection" ? (rec.sel || "")
             : c.sys === "concat"    ? rec.p.map(x => x.t).join(" ")
             :                         (rec.sa || "");
  const got = judgeTextual(text, rec.b);
  if (got.length === c.j.length && got.every((v, i) => v === c.j[i])) okt++;
  else fail(`textual ${c.inst}/${c.sys}: python=[${c.j}] js=[${got}]`);
}
console.log(`textual         ${okt}/${(ref.textual || []).length} resolver judgements`);

if (failures) {
  console.error(`\nFAILED: ${failures} disagreement(s). A port has drifted from Python.`);
  process.exit(1);
}
console.log("\nOK: the browser engines agree with the Python implementation.");

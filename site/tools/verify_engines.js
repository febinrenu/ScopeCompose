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

const { screenPair, heuristicNLI } = new Function(
  load("assets/js/detector.js") + "\nreturn { screenPair, heuristicNLI };"
)();

// compose.js leans on compare/REL from engines.js and outcomes_match from
// match.js, so the three are evaluated in one scope, exactly as the browser
// loads them.
const { judgeTextual, composeBranches, renderTemplate, checkFaithfulness } = new Function(
  load("assets/js/engines.js") + "\n" +
  load("assets/js/match.js") + "\n" +
  load("assets/js/compose.js") +
  "\nreturn { judgeTextual, composeBranches, renderTemplate, checkFaithfulness };"
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

/* --- oracle chain: gold routing -> compose -> render -> judge ----------- */
let oko = 0, oraclePreserved = 0, oracleTotal = 0;
for (const c of ref.oracle || []) {
  const rec = byId.get(c.inst);
  if (!rec) { fail(`oracle: missing instance ${c.inst}`); continue; }
  const action = routeFor(rec.ty, rec.rel);
  const composed = composeBranches(rec, rec.b, action);
  const answer = renderTemplate(composed);
  const judged = judgeTextual(answer, rec.b);
  oracleTotal += judged.length;
  oraclePreserved += judged.filter(j => j === "preserved").length;

  let bad = null;
  if (composed.resolution !== c.resolution)
    bad = `resolution python=${c.resolution} js=${composed.resolution}`;
  else if (composed.branches.map(b => b.id).join(",") !== c.kept.join(","))
    bad = `kept python=[${c.kept}] js=[${composed.branches.map(b => b.id)}]`;
  else if (answer !== c.answer)
    bad = `answer\n      python: ${c.answer}\n      js:     ${answer}`;
  else if (judged.join(",") !== c.judge.join(","))
    bad = `judge python=[${c.judge}] js=[${judged}]`;
  else if (checkFaithfulness(answer, composed.branches).ok !== c.faithful)
    bad = `faithful python=${c.faithful}`;
  if (bad) fail(`oracle ${c.inst}: ${bad}`); else oko++;
}
console.log(`oracle          ${oko}/${(ref.oracle || []).length} instances `
          + `(resolution, kept branches, rendered answer, judgements, faithfulness)`);
if (oracleTotal) {
  console.log(`                PR ${oraclePreserved}/${oracleTotal} = `
            + `${(oraclePreserved / oracleTotal).toFixed(4)}  <- UPPER BOUND, gold routing and gold branches`);
}

/* --- retrieval-side baselines ------------------------------------------ */
const RETR = {
  standard_rag: rec => rec.p.map(p => p.id),
  rerank_top1: rec => [rec.p[0].id],
  nli_filter: rec => {
    const ps = rec.p;
    if (ps.length < 2) return ps.map(p => p.id);
    const totals = ps.map((a, i) => ps.reduce((acc, b, j) =>
      i === j ? acc : acc + heuristicNLI(b.t, a.t).contradiction, 0));
    let worst = 0;
    for (let i = 1; i < ps.length; i++){
      if (totals[i] > totals[worst]
          || (totals[i] === totals[worst] && ps[i].id > ps[worst].id)) worst = i;
    }
    if (totals[worst] < 0.5) return ps.map(p => p.id);
    return ps.filter((_, i) => i !== worst).map(p => p.id);
  },
};
let okret = 0;
const lostBy = {};
for (const c of ref.retrieval || []) {
  const rec = byId.get(c.inst);
  if (!rec) { fail(`retrieval: missing instance ${c.inst}`); continue; }
  const kept = new Set(RETR[c.sys](rec));
  const lost = rec.b.filter(b => b.sup && !kept.has(b.sup)).map(b => b.id).sort();
  lostBy[c.sys] = (lostBy[c.sys] || 0) + lost.length;
  if ([...kept].sort().join(",") !== c.kept.join(",")) fail(`retrieval ${c.inst}/${c.sys}: kept differs`);
  else if (lost.join(",") !== c.lost.join(",")) fail(`retrieval ${c.inst}/${c.sys}: lost python=[${c.lost}] js=[${lost}]`);
  else okret++;
}
console.log(`retrieval       ${okret}/${(ref.retrieval || []).length} baseline runs  `
          + Object.entries(lostBy).sort().map(([k, v]) => `${k} lost ${v}`).join(", "));

if (failures) {
  console.error(`\nFAILED: ${failures} disagreement(s). A port has drifted from Python.`);
  process.exit(1);
}
console.log("\nOK: the browser engines agree with the Python implementation.");

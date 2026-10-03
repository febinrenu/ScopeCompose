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

if (failures) {
  console.error(`\nFAILED: ${failures} disagreement(s). A port has drifted from Python.`);
  process.exit(1);
}
console.log("\nOK: the browser engines agree with the Python implementation.");

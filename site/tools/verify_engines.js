/*  Check the browser engine against the Python it ports.
 *
 *      python site/tools/make_reference.py    # record Python's answers
 *      node   site/tools/verify_engines.js    # replay them through the JS
 *
 *  Exits non-zero on any disagreement. Wired into the deploy workflow, so a
 *  drifted port stops the site shipping rather than being discovered by a
 *  reviewer who notices the page disagrees with the paper.
 *
 *  Two checks:
 *
 *    1. compare() against every ordered branch pair in the corpus, both
 *       directions, including the SUBSET/SUPERSET asymmetry that assigns
 *       branch roles.
 *    2. the four-way relation derived from the two determinations against the
 *       gold label on every two-branch instance. This one is the stronger
 *       statement: the page is not displaying the stored label, it is
 *       recomputing it and arriving at the same answer.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const SITE = path.resolve(__dirname, "..");
const load = p => fs.readFileSync(path.join(SITE, p), "utf8");

// engines.js is a classic script, not a module -- evaluate it and lift out
// the symbols rather than maintaining a parallel export list.
const engines = load("assets/js/engines.js");
const { compare, combine } = new Function(
  engines + "\nreturn { compare, combine };"
)();

const corpus = JSON.parse(load("data/corpus.json"));
const refPath = path.join(__dirname, "compare_reference.json");
if (!fs.existsSync(refPath)) {
  console.error("No reference file. Run: python site/tools/make_reference.py");
  process.exit(2);
}
const reference = JSON.parse(fs.readFileSync(refPath, "utf8"));
const byId = new Map(corpus.map(r => [r.id, r]));

let failures = 0;

/* --- 1. compare() ------------------------------------------------------- */
let matched = 0;
for (const c of reference) {
  const rec = byId.get(c.inst);
  if (!rec) { console.error(`  MISSING instance ${c.inst}`); failures++; continue; }
  const a = rec.b.find(x => x.id === c.a);
  const b = rec.b.find(x => x.id === c.b);
  const got = compare({ def: a.def, at: a.at }, { def: b.def, at: b.at });
  if (got === c.rel) matched++;
  else {
    failures++;
    if (failures <= 15) {
      console.error(`  MISMATCH ${c.inst} ${c.a}->${c.b}: python=${c.rel} js=${got}`);
    }
  }
}
console.log(`compare()          ${matched}/${reference.length} match Python`);

/* --- 2. the four-way relation ------------------------------------------- */
let agree = 0, differ = 0;
for (const r of corpus) {
  if (!r.b || r.b.length !== 2) continue;
  const [x, y] = r.b;
  const setRel = compare({ def: x.def, at: x.at }, { def: y.def, at: y.at });
  const outcomesAgree = x.out.trim().toLowerCase() === y.out.trim().toLowerCase();
  const [derived] = combine(setRel, outcomesAgree);
  if (derived === r.rel) agree++;
  else {
    differ++; failures++;
    if (differ <= 10) {
      console.error(`  RELATION ${r.id}: gold=${r.rel} derived=${derived} (set=${setRel})`);
    }
  }
}
console.log(`four-way relation  ${agree} derived == gold, ${differ} differ`);

if (failures) {
  console.error(`\nFAILED: ${failures} disagreement(s). The port has drifted from Python.`);
  process.exit(1);
}
console.log("\nOK: the browser engine agrees with the Python implementation.");

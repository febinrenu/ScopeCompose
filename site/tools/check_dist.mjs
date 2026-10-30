/*  The build must ship the verified engines untouched.
 *
 *      npm run build && node tools/check_dist.mjs
 *
 *  1. every file in public/engine and public/data is byte-identical in dist/
 *  2. dist/index.html loads them as classic scripts, in order, before the module
 *  3. no bundle in dist/assets contains a second copy of engine code
 *  4. nothing that is not the site (Python, sources, tools) is shipped
 *  Exits non-zero on any failure.
 */
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const site = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(site, "dist");
const fails = [];
const sha = p => createHash("sha256").update(readFileSync(p)).digest("hex");
const walk = d => readdirSync(d).flatMap(f => {
  const p = join(d, f);
  return statSync(p).isDirectory() ? walk(p) : [p];
});

if (!existsSync(dist)) { console.error("no dist/ — run `npm run build` first"); process.exit(2); }

// 1. byte identity
for (const sub of ["engine", "data"]) {
  // nested too: data/runs/ holds the studio's recorded runs, which ship as recorded
  for (const a of walk(join(site, "public", sub))) {
    const f = relative(join(site, "public", sub), a).replace(/\\/g, "/");
    const b = join(dist, sub, f);
    if (!existsSync(b)) fails.push(`missing from dist: ${sub}/${f}`);
    else if (sha(a) !== sha(b)) fails.push(`bytes differ: ${sub}/${f}`);
  }
}

// 2. order
const html = readFileSync(join(dist, "index.html"), "utf8");
const want = ["data/corpus.js", "engine/engines.js", "engine/match.js", "engine/stats.js",
  "engine/detector.js", "engine/compose.js", "engine/baselines.js"];
const pos = want.map(w => html.indexOf(w));
const mod = html.search(/<script type="module"[^>]*src="[^"]*assets\//);
if (pos.some(p => p < 0)) fails.push("an engine script tag is missing from dist/index.html");
for (let i = 1; i < pos.length; i++) if (pos[i] < pos[i - 1]) fails.push(`script order: ${want[i]} before ${want[i - 1]}`);
if (mod < 0 || pos.some(p => p > mod)) fails.push("the module bundle must load after every engine script");
for (const w of want) {
  const tag = html.slice(html.lastIndexOf("<script", html.indexOf(w)), html.indexOf("</script>", html.indexOf(w)));
  if (!/\bdefer\b/.test(tag)) fails.push(`${w} is not deferred`);
  if (/type="module"/.test(tag)) fails.push(`${w} must be a classic script`);
}

// 3. no second copy of the engines inside the bundles
const SENTINELS = ["function screenPair(", "function compare(a, b)", "function composeBranches(", "function judgeTextual(", "const RETRIEVAL_BASELINES"];
for (const f of walk(join(dist, "assets")).filter(f => f.endsWith(".js"))) {
  const s = readFileSync(f, "utf8");
  for (const t of SENTINELS) if (s.includes(t)) fails.push(`${relative(dist, f)} contains engine code (${t})`);
}

// 4. nothing else
for (const f of walk(dist)) {
  const r = relative(dist, f).replace(/\\/g, "/");
  if (/\.py$|^src\/|^tools\/|^tests\/|README|node_modules/.test(r)) fails.push(`should not ship: ${r}`);
}

if (fails.length) {
  console.error("check_dist FAILED:\n  " + fails.join("\n  "));
  process.exit(1);
}
console.log(`check_dist OK: ${want.length} engine/data scripts byte-identical and in order; no engine code re-bundled.`);

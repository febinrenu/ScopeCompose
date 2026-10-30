/*  Weight budgets, gzip, measured on dist/. Fails the build if exceeded.
 *
 *      node tools/size.mjs
 */
import { gzipSync } from "node:zlib";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const dist = resolve(dirname(fileURLToPath(import.meta.url)), "..", "dist");
if (!existsSync(dist)) { console.error("no dist/ — run `npm run build` first"); process.exit(2); }
const gz = p => gzipSync(readFileSync(p), { level: 9 }).length;
const kb = n => (n / 1024).toFixed(1) + " KB";

const html = readFileSync(join(dist, "index.html"), "utf8");
const entry = html.match(/<script type="module"[^>]*src="\.?\/?(assets\/[^"]+\.js)"/)?.[1];
const assets = readdirSync(join(dist, "assets"));
const js = assets.filter(f => f.endsWith(".js"));
const css = assets.filter(f => f.endsWith(".css"));
const scene = js.find(f => f.startsWith("boot-"));
const post = js.find(f => f.startsWith("post-"));

const rows = [];
const check = (label, bytes, budget) => rows.push({ label, bytes, budget, ok: bytes <= budget });

if (!entry) { console.error("could not find the entry bundle in dist/index.html"); process.exit(1); }
check("entry JS (DOM, scroll, story, demo)", gz(join(dist, entry)), 100 * 1024);
const classic = ["data/corpus.js", "engine/engines.js", "engine/match.js", "engine/stats.js",
  "engine/detector.js", "engine/compose.js", "engine/baselines.js"].reduce((s, f) => s + gz(join(dist, f)), 0);
check("engines + corpus (classic scripts)", classic, 40 * 1024);
check("initial total (entry + classic)", gz(join(dist, entry)) + classic, 150 * 1024);
if (!scene) { console.error("no boot-*.js chunk: the 3D world is no longer split out"); process.exit(1); }
check("3D chunk (Three.js + world, lazy)", gz(join(dist, "assets", scene)), 200 * 1024);
// post-processing loads in parallel with the world, tiers 2-3 only
if (post) check("post chunk (bloom, DoF, lens; tiers 2-3)", gz(join(dist, "assets", post)), 90 * 1024);
check("3D at tier 3 (world + post, lazy)", gz(join(dist, "assets", scene)) + (post ? gz(join(dist, "assets", post)) : 0), 270 * 1024);
// the studio (#/studio) is its own lazy chunk, with its own stylesheet
const studioJs = js.filter(f => /^(studio|eval|docs|runs|apidocs)-/.test(f));
const studioCss = css.filter(f => f.startsWith("studio-"));
if (!studioJs.length) { console.error("no studio-*.js chunk: the studio is no longer split out"); process.exit(1); }
check("studio JS (lazy, #/studio)", studioJs.reduce((s, f) => s + gz(join(dist, "assets", f)), 0), 60 * 1024);
check("CSS (site)", css.filter(f => !studioCss.includes(f)).reduce((s, f) => s + gz(join(dist, "assets", f)), 0), 15 * 1024);
check("CSS (studio, lazy)", studioCss.reduce((s, f) => s + gz(join(dist, "assets", f)), 0), 10 * 1024);

let failed = false;
for (const r of rows) {
  console.log(`${r.ok ? "ok  " : "OVER"}  ${r.label.padEnd(40)} ${kb(r.bytes).padStart(9)}  / ${kb(r.budget)}`);
  if (!r.ok) failed = true;
}
if (failed) process.exit(1);

/*  Load the browser engines exactly as the page does: classic scripts, in
 *  order, into one shared global scope. vm.runInThisContext gives them the
 *  same semantics as <script> tags -- top-level functions on the global
 *  object, top-level consts as global lexical bindings visible to modules.
 */
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { runInThisContext } from "node:vm";

const site = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
(globalThis as unknown as { window: unknown }).window = globalThis;

const ORDER = [
  "public/data/corpus.js",
  "public/engine/engines.js",
  "public/engine/match.js",
  "public/engine/stats.js",
  "public/engine/detector.js",
  "public/engine/compose.js",
  "public/engine/baselines.js",
];

const g = globalThis as unknown as { __SC_ENGINES__?: boolean };
if (!g.__SC_ENGINES__) {
  for (const f of ORDER) runInThisContext(readFileSync(resolve(site, f), "utf8"), { filename: f });
  g.__SC_ENGINES__ = true;
}

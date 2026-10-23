import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..");

/**
 * The engine scripts in public/ are served exactly as authored -- Vite never
 * bundles or rewrites them, because tools/verify_engines.js checks those bytes
 * against the Python. Their filenames are therefore not content-hashed, so this
 * appends ?v=<sha8> to each src attribute. The files themselves are untouched.
 */
function engineCacheBust() {
  return {
    name: "engine-cache-bust",
    transformIndexHtml(html: string) {
      return html.replace(/src="\.?\/?((?:engine|data)\/[\w.-]+\.js)"/g, (_m, p: string) => {
        const sha = createHash("sha256").update(readFileSync(resolve(here, "public", p))).digest("hex");
        return `src="./${p}?v=${sha.slice(0, 8)}"`;
      });
    },
  };
}

export default defineConfig({
  root: here,
  publicDir: "public",
  // Relative base: works at /ScopeCompose/ on Pages, under `vite preview`, and
  // from any static server. Routing is hash-only, so there are no deep paths.
  base: process.env.VITE_BASE ?? "./",
  plugins: [engineCacheBust()],
  server: {
    // facts.ts imports benchmark/data/decisive.json and the contract schema
    // directly, so the page quotes the artifact rather than a copy of it.
    fs: { allow: [repo] },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
    sourcemap: true,
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 900,
  },
});

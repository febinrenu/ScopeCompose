/*  The Studio, end to end, with the engine server running.
 *
 *      npm run build && node tools/studio_smoke.mjs
 *
 *  Starts `python -m service` (the repo's .venv, or $STUDIO_PYTHON) on its own
 *  port with a throwaway database, starts `vite preview`, and drives every
 *  Studio tab through a real browser: a console run streamed from the server,
 *  a collection built from pasted text and audited, a short evaluation, the
 *  history, and the API's "try it". Offline mode only: no keys, no network.
 */
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const site = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(site, "..");
const PORT = Number(process.env.STUDIO_SMOKE_PORT || 4182);
const API = Number(process.env.STUDIO_SMOKE_API || 8018);
const py = process.env.STUDIO_PYTHON
  || [join(root, ".venv", "Scripts", "python.exe"), join(root, ".venv", "bin", "python")].find(existsSync)
  || "python";

const fails = [];
const ok = m => console.log("  ok   " + m);
const bad = m => { fails.push(m); console.log("  FAIL " + m); };
const expect = (c, m) => (c ? ok(m) : bad(m));

const waitFor = (proc, needle, ms) => new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error(`timed out waiting for "${needle}"`)), ms);
  const on = d => { if (String(d).includes(needle)) { clearTimeout(t); res(); } };
  proc.stdout.on("data", on);
  proc.stderr.on("data", on);
  proc.on("exit", c => { clearTimeout(t); rej(new Error(`process exited (${c}) before "${needle}"`)); });
});

const db = join(mkdtempSync(join(tmpdir(), "studio-smoke-")), "studio.sqlite");
const engine = spawn(py, ["-m", "service", "--port", String(API)], {
  cwd: root, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, SCOPECOMPOSE_STUDIO_DB: db, PYTHONUNBUFFERED: "1" },
});
const preview = spawn(process.execPath, [resolve(site, "node_modules/vite/bin/vite.js"), "preview", "--port", String(PORT), "--strictPort"],
  { cwd: site, stdio: ["ignore", "pipe", "pipe"] });

let browser;
try {
  await Promise.all([waitFor(engine, "engine server on", 90000), waitFor(preview, String(PORT), 30000)]);
  browser = await chromium.launch({ channel: process.env.SMOKE_CHANNEL || "chrome", headless: true,
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(port => { try { localStorage.setItem("sc.server", `http://127.0.0.1:${port}`); } catch {} }, API);
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", m => { if (m.type() === "error" && !/Failed to load resource/.test(m.text())) errors.push(m.text()); });
  const done = sel => page.waitForFunction(s => /done/.test(document.querySelector(s)?.textContent || ""), sel, { timeout: 90000 });

  console.log("console");
  await page.goto(`http://localhost:${PORT}/?q=off#/studio`, { waitUntil: "networkidle" });
  await page.waitForSelector(".sc .srv[data-state='on'], .sc .srv[data-state='live']", { timeout: 20000 });
  ok("the studio finds the engine server");
  await page.click("[data-k=run]");
  await done(".sc-counters .st");
  const st = await page.evaluate(() => ({
    nodes: [...document.querySelectorAll(".pg-node")].map(n => n.getAttribute("data-s")),
    answer: document.querySelector(".res blockquote")?.textContent || "",
    hash: location.hash,
  }));
  expect(st.nodes.length === 14 && st.nodes.every(s => /done|skipped|warn/.test(s || "")), `every stage reported by the server finished (${st.nodes.join(",")})`);
  expect(/12\.00 monthly fee/.test(st.answer) && /Graduate Scheme/.test(st.answer), "the server's run composes both branches");
  expect(/^#\/studio\/run\/run_/.test(st.hash), "the run gets a shareable address");
  for (const [i, needle] of [[2, "17 pair features"], [5, "Refinement"], [8, "Composed"]]) {
    await page.locator(".pg-node").nth(i).click();
    await page.waitForTimeout(150);
    expect((await page.textContent(".sc-ins")).toLowerCase().includes(needle.toLowerCase()), `the inspector shows stage ${i + 1}'s own output (“${needle}”)`);
  }
  await page.click(".dock-bar [data-tab=contract]");
  expect((await page.locator(".snaps button").count()) >= 3, "the contract tab shows the QueryRecord at each stage");

  console.log("documents and audit");
  await page.goto(`http://localhost:${PORT}/?q=off#/studio/docs`, { waitUntil: "networkidle" });
  await page.fill("[data-k=name]", "Smoke fees");
  await page.click(".dv-new button");
  await page.waitForSelector("[data-k=dtext]");
  await page.fill("[data-k=dtext]", [
    "A non-sterling transaction fee of 2.99% applies to all card payments made in a foreign currency.",
    "Premium Plus cardholders pay no non-sterling transaction fee on card payments abroad.",
    "A monthly account fee of 12.00 is payable on all current accounts.",
    "Accounts opened under the Graduate Scheme carry no monthly charge for the first three years after registration.",
  ].join("\n\n"));
  await page.click(".dv-paste button");
  await page.waitForFunction(() => /4 passage/.test(document.querySelector(".dv-h")?.textContent || ""), null, { timeout: 20000 });
  ok("pasted text becomes four passages");
  await page.click("[data-k=audit]");
  await page.waitForFunction(() => /high-risk/.test(document.querySelector("[data-k=aprog]")?.textContent || ""), null, { timeout: 60000 });
  const risks = await page.$$eval(".finding2 header .chip", cs => cs.map(c => c.textContent));
  expect(risks.filter(r => /high/.test(r || "")).length >= 2, `the audit flags the two rule/exception pairs as high risk (${risks.join(", ")})`);
  await page.click("[data-k=ask]");
  await page.waitForTimeout(800);
  await page.fill("[data-k=q]", "Do I pay a fee on card payments abroad?");
  await page.click("[data-k=run]");
  await done(".sc-counters .st");
  expect((await page.textContent(".sc-feed")).includes("Retrieve") || (await page.textContent(".sc-ins")).length > 0,
    "a question over the collection runs end to end");

  console.log("evaluate, history, API");
  await page.goto(`http://localhost:${PORT}/?q=off#/studio/eval`, { waitUntil: "networkidle" });
  await page.fill("[data-k=limit]", "5");
  await page.click("[data-k=go]");
  await done("[data-k=prog]");
  expect(await page.locator("[data-k=rows] tr").count() === 5, "a five-instance evaluation streams five rows from the server");
  await page.goto(`http://localhost:${PORT}/?q=off#/studio/runs`, { waitUntil: "networkidle" });
  await page.waitForSelector("tr[data-id]", { timeout: 15000 });
  const kinds = await page.$$eval("tr[data-id] td:nth-child(3)", tds => tds.map(t => t.textContent));
  expect(["run", "audit", "eval"].every(k => kinds.includes(k)), `history lists every kind of run (${[...new Set(kinds)].join(", ")})`);
  await page.locator("tr[data-id]").first().click();
  await page.waitForSelector("[data-x=html]", { timeout: 10000 });
  ok("a run opens with its export buttons");
  await page.goto(`http://localhost:${PORT}/?q=off#/studio/api`, { waitUntil: "networkidle" });
  await page.click("[data-k=send]");
  await page.waitForFunction(() => /HTTP 200/.test(document.querySelector("[data-k=out]")?.textContent || ""), null, { timeout: 30000 });
  ok("the API page's “try it” gets HTTP 200 from /v1/resolve");

  expect(errors.length === 0, "no console errors" + (errors.length ? ": " + errors.slice(0, 3).join(" | ") : ""));
  await ctx.close();
} catch (e) {
  bad("studio smoke crashed: " + e.message);
} finally {
  await browser?.close();
  preview.kill();
  engine.kill();
}

if (fails.length) { console.error(`\nstudio smoke FAILED: ${fails.length} check(s)`); process.exit(1); }
console.log("\nstudio smoke OK");

/*  End-to-end smoke test in a real browser.
 *
 *      npm run build && node tools/smoke.mjs
 *
 *  Starts `vite preview`, drives the system Chrome through playwright-core
 *  (no browser download), and checks what a reviewer would notice first:
 *  engines present, no console errors, the story scrolls forward and back
 *  to the identical frame, WebGL context loss recovers, reduced motion
 *  turns smoothing off, the lab views compute their numbers, and no
 *  viewport scrolls sideways.
 */
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const site = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PORT = Number(process.env.SMOKE_PORT || 4179);
const BASE = `http://localhost:${PORT}/`;
const fails = [];
const ok = msg => console.log("  ok   " + msg);
const bad = msg => { fails.push(msg); console.log("  FAIL " + msg); };
const expect = (cond, msg) => (cond ? ok(msg) : bad(msg));

const server = spawn(process.execPath, [resolve(site, "node_modules/vite/bin/vite.js"), "preview", "--port", String(PORT), "--strictPort"],
  { cwd: site, stdio: ["ignore", "pipe", "pipe"] });
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite preview did not start")), 30000);
  server.stdout.on("data", d => { if (String(d).includes(String(PORT))) { clearTimeout(t); res(); } });
  server.on("exit", c => rej(new Error("vite preview exited " + c)));
});

const browser = await chromium.launch({
  channel: process.env.SMOKE_CHANNEL || "chrome", headless: true,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});

async function open(path, { width = 1440, height = 900, reducedMotion = "no-preference" } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height }, reducedMotion });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto(BASE + path, { waitUntil: "networkidle" });
  await page.waitForFunction(() => document.documentElement.classList.contains("gl")
    || document.documentElement.classList.contains("no-gl"), null, { timeout: 20000 });
  return { page, ctx, errors };
}

const scrollTo = async (page, y) => { await page.evaluate(v => window.scrollTo(0, v), y); await page.waitForTimeout(1600); };

/** Wait until the world has stopped drawing: the frame counter holds still. */
async function settle(page, quietMs = 900, maxMs = 20000) {
  const t0 = Date.now();
  let last = await page.evaluate(() => document.documentElement.dataset.frame), since = Date.now();
  while (Date.now() - t0 < maxMs) {
    await page.waitForTimeout(150);
    const f = await page.evaluate(() => document.documentElement.dataset.frame);
    if (f !== last) { last = f; since = Date.now(); } else if (Date.now() - since >= quietMs) return true;
  }
  return false;
}
const frameAt = async (page, y) => {
  await page.evaluate(v => window.scrollTo(0, v), y);
  await page.waitForTimeout(400);
  await settle(page);
  // the world alone: DOM copy is hidden, since text rasterisation (layers, AA) is not what this proves
  await page.addStyleTag({ content: "#story .frame > *, .bar { visibility: hidden !important; }" });
  await page.waitForTimeout(300);
  // a software renderer under load can still be finishing a frame: take the frame once
  // two captures a second apart agree
  let shot = await page.screenshot();
  for (let i = 0; i < 6; i++) {
    await page.waitForTimeout(1000);
    const again = await page.screenshot();
    if (again.equals(shot)) break;
    shot = again;
  }
  await page.evaluate(() => document.querySelectorAll("style").forEach(s => { if (s.textContent?.includes("visibility: hidden !important")) s.remove(); }));
  if (process.env.SMOKE_SHOTS) (await import("node:fs")).writeFileSync(`${process.env.SMOKE_SHOTS}/strict_${Date.now()}.png`, shot);
  return createHash("sha1").update(shot).digest("hex");
};

try {
  console.log("story (?q=low)");
  {
    const { page, ctx, errors } = await open("?q=low");
    expect(await page.evaluate(() => typeof compare === "function" && typeof composeBranches === "function"), "engines are loaded as page globals");
    expect(await page.evaluate(() => document.documentElement.classList.contains("gl-ready")), "the 3D world rendered its first frame");
    const H = await page.evaluate(() => document.documentElement.scrollHeight - innerHeight);
    const act3 = await page.evaluate(() => { const el = document.getElementById("act-03"); return el.getBoundingClientRect().top + scrollY + (el.offsetHeight - innerHeight) * 0.4; });

    const seenDown = [];
    for (let i = 0; i <= 20; i++) { await scrollTo(page, H * i / 20); seenDown.push(Number(await page.evaluate(() => document.documentElement.dataset.act))); }
    expect(seenDown.every((v, i) => i === 0 || v >= seenDown[i - 1]) && seenDown.at(-1) === 10, `acts advance in order going down (${[...new Set(seenDown)].join(",")})`);

    await scrollTo(page, act3);
    const down = createHash("sha1").update(await page.screenshot()).digest("hex");
    await scrollTo(page, H);
    await scrollTo(page, act3);
    const up = createHash("sha1").update(await page.screenshot()).digest("hex");
    expect(down === up, "the same scroll position gives the identical frame, after scrolling past and back");

    const seenUp = [];
    for (let i = 20; i >= 0; i--) { await scrollTo(page, H * i / 20); seenUp.push(Number(await page.evaluate(() => document.documentElement.dataset.act))); }
    expect(seenUp.every((v, i) => i === 0 || v <= seenUp[i - 1]) && seenUp.at(-1) === 0, "acts reverse in order going up");

    await page.evaluate(() => {
      const c = document.getElementById("world");
      window.__lose = c.getContext("webgl2").getExtension("WEBGL_lose_context");
      window.__lose.loseContext();
    });
    await page.waitForTimeout(400);
    expect(await page.evaluate(() => document.documentElement.classList.contains("gl-lost")), "context loss falls back to the poster");
    await page.evaluate(() => window.__lose.restoreContext());
    await page.waitForTimeout(800);
    expect(await page.evaluate(() => !document.documentElement.classList.contains("gl-lost")), "context restore brings the world back");
    expect(errors.length === 0, "no console errors" + (errors.length ? ": " + errors.slice(0, 3).join(" | ") : ""));
    await ctx.close();
  }

  console.log("post-processed world, strict (?q=mid&still)");
  {
    // bloom, transmission, the lens pass and the shatter: with time frozen the
    // frame must still be a pure function of the scroll position
    const { page, ctx, errors } = await open("?q=mid&still");
    expect(await page.evaluate(() => document.documentElement.dataset.tier === "2"), "tier 2 forced");
    const y = await page.evaluate(() => { const el = document.getElementById("act-02"); return el.getBoundingClientRect().top + scrollY + (el.offsetHeight - innerHeight) * 0.92; });
    const H = await page.evaluate(() => document.documentElement.scrollHeight - innerHeight);
    const a = await frameAt(page, y);
    await frameAt(page, H * 0.75);
    const b = await frameAt(page, y);
    expect(a === b, "mid-shatter frame is identical after scrolling far past and back");
    expect(errors.length === 0, "no console errors" + (errors.length ? ": " + errors.slice(0, 3).join(" | ") : ""));
    await ctx.close();
  }

  console.log("reduced motion");
  {
    const { page, ctx, errors } = await open("", { reducedMotion: "reduce" });
    const cls = await page.evaluate(() => document.documentElement.className);
    expect(/\brm\b/.test(cls) && !/\blenis\b/.test(cls), "reduced motion: no smooth scrolling, no pinned choreography");
    expect(await page.evaluate(() => [...document.querySelectorAll(".beat")].every(b => getComputedStyle(b).opacity === "1")), "every story beat is visible as ordinary text");
    expect(errors.length === 0, "no console errors");
    await ctx.close();
  }

  console.log("lab views");
  {
    const { page, ctx, errors } = await open("?q=off#/bench");
    await page.waitForSelector("#view-bench table.data");
    expect(await page.locator("#view-bench table.data").first().locator("tbody tr").count() === 5, "bench: five resolver rows");
    const benchText = await page.locator("#view-bench").innerText();
    expect(/oracle — upper bound/i.test(benchText) && /gameable/i.test(benchText) && /routing-level/i.test(benchText), "bench: the framing badges are present");
    expect(/0\.976/.test(benchText), "bench: oracle PR computed (0.976)");
    expect(await page.evaluate(() => getComputedStyle(document.documentElement).overflowY !== "hidden"
      && !document.documentElement.classList.contains("lenis-stopped")), "lab view scrolls (nothing locks the page)");
    await page.goto(BASE + "?q=off#/byo"); await page.waitForSelector("#byoOut .verdict");
    const byo = await page.locator("#byoOut").innerText();
    expect(/Answer:/.test(byo) && /passes/.test(byo), "bring-your-own: the preset composes and passes faithfulness");
    await page.goto(BASE + "?q=off#/architecture"); await page.waitForSelector("#view-architecture svg");
    expect(await page.locator("#view-architecture .arch-legend > div").count() >= 7, "architecture: all seven stages described");
    await page.goto(BASE + "?q=off#claims"); await page.waitForTimeout(1500);
    expect(!(await page.evaluate(() => document.documentElement.classList.contains("lab-on"))), "a legacy anchor (#claims) returns to the story");
    expect(errors.length === 0, "no console errors" + (errors.length ? ": " + errors.slice(0, 3).join(" | ") : ""));
    await ctx.close();
  }

  console.log("studio, with no engine server");
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    // point the studio at a port nothing listens on, so a server running on this machine cannot change the result
    await ctx.addInitScript(() => { try { localStorage.setItem("sc.server", "http://127.0.0.1:9"); } catch {} });
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    page.on("console", m => { if (m.type() === "error" && !/127\.0\.0\.1:9|ERR_CONNECTION_REFUSED|Failed to load resource/.test(m.text())) errors.push(m.text()); });
    await page.goto(BASE + "?q=off#/studio", { waitUntil: "networkidle" });
    await page.waitForSelector(".sc .srv[data-state='off']", { timeout: 15000 });
    expect(true, "studio: says plainly that no engine server is connected");
    await page.click("[data-k=run]");
    await page.waitForFunction(() => /done/.test(document.querySelector(".sc-counters .st")?.textContent || ""), null, { timeout: 30000 });
    const st = await page.evaluate(() => ({
      nodes: [...document.querySelectorAll(".pg-node")].map(n => n.getAttribute("data-s")),
      answer: document.querySelector(".res blockquote")?.textContent || "",
    }));
    expect(st.nodes.length === 14 && st.nodes.every(s => /done|skipped|warn/.test(s || "")), `studio: a browser run completes all 14 stages (${st.nodes.join(",")})`);
    expect(/12\.00 monthly fee/.test(st.answer) && /Graduate Scheme/.test(st.answer), "studio: the browser run composes both branches");
    await page.click(".sc-rec [data-rec]");
    await page.waitForFunction(() => /done/.test(document.querySelector(".sc-counters .st")?.textContent || ""), null, { timeout: 60000 });
    expect(await page.evaluate(() => document.querySelectorAll(".dock-bar [data-tab=llm] .n")[0]?.textContent !== undefined), "studio: a recorded engine-server run replays");
    await page.goto(BASE + "?q=off#/studio/eval", { waitUntil: "networkidle" });
    await page.fill("[data-k=limit]", "6");
    await page.click("[data-k=go]");
    await page.waitForFunction(() => /done/.test(document.querySelector("[data-k=prog]")?.textContent || ""), null, { timeout: 60000 });
    expect(await page.locator("[data-k=rows] tr").count() === 6, "studio: a browser evaluation fills six rows");
    expect(errors.length === 0, "studio: no console errors" + (errors.length ? ": " + errors.slice(0, 3).join(" | ") : ""));
    await ctx.close();
  }

  console.log("viewports");
  for (const [w, h] of [[1440, 900], [1024, 768], [390, 844]]) {
    const { page, ctx, errors } = await open("?q=low", { width: w, height: h });
    const over = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    expect(over <= 1, `${w}×${h}: no horizontal scroll (${over}px)`);
    expect(errors.length === 0, `${w}×${h}: no console errors`);
    await ctx.close();
  }
} catch (e) {
  bad("smoke run crashed: " + e.message);
} finally {
  await browser.close();
  server.kill();
}

if (fails.length) { console.error(`\nsmoke FAILED: ${fails.length} check(s)`); process.exit(1); }
console.log("\nsmoke OK");

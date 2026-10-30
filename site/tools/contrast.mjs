/*  Visibility audit: is every piece of text actually readable on the frame
 *  it sits on -- the 3D world, the scrims and the bloom included?
 *
 *      npm run build && node tools/contrast.mjs [--quick] [--out dir]
 *
 *  For each act, at several scroll positions and three viewports, it takes
 *  two screenshots of the same frame: one as rendered, one with every glyph
 *  made transparent. The second is exactly what each piece of text sits on.
 *  For every visible text run it samples that background inside the run's
 *  box (the 90th-percentile brightest pixels: the worst case for light type),
 *  composites the text colour over it, and computes the WCAG contrast ratio.
 *
 *  Thresholds: 4.5:1 for body text, 3:1 for large text (>= 24px, or >= 18.66px
 *  bold). Exits non-zero if any run falls below. `?still` freezes the world so
 *  both screenshots show the same frame.
 */
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const site = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const quick = args.includes("--quick");
const outDir = args.includes("--out") ? resolve(args[args.indexOf("--out") + 1]) : null;
const PORT = Number(process.env.CONTRAST_PORT || 4181);

const VIEWPORTS = quick ? [[1440, 900]] : [[1440, 900], [1024, 768], [390, 844]];
// act -> scroll progress points (the pinned acts' beats change with progress)
const POINTS = {
  "00": [0, 0.6], "01": [0.15, 0.55, 0.95], "02": [0.1, 0.3, 0.5, 0.7, 0.92], "03": [0.1, 0.7],
  "04": [0.05, 0.35, 0.6, 0.9], "05": [0.1, 0.5, 0.9], "06": [0.1, 0.5], "07": [0.2, 0.8],
  "08": [0.05, 0.4, 0.8], "09": [0.05, 0.4, 0.8], "10": [0.4, 0.95],
};

const server = spawn(process.execPath, [resolve(site, "node_modules/vite/bin/vite.js"), "preview", "--port", String(PORT), "--strictPort"],
  { cwd: site, stdio: ["ignore", "pipe", "pipe"] });
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite preview did not start")), 30000);
  server.stdout.on("data", d => { if (String(d).includes(String(PORT))) { clearTimeout(t); res(); } });
});

const browser = await chromium.launch({
  channel: process.env.CONTRAST_CHANNEL || "chrome", headless: true,
  args: ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"],
});

/** Every visible text run in the viewport: its box, colour, size, weight. */
const COLLECT = () => {
  const out = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const vw = innerWidth, vh = innerHeight;
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const text = n.textContent.replace(/\s+/g, " ").trim();
    if (text.length < 2) continue;
    const el = n.parentElement;
    if (!el || el.closest("script,style,noscript,svg,canvas,[aria-hidden=true] .sr-only,.sr-only,#poster,#labels,.cur-ring,.cur-dot,.skip,option,select,textarea,input")) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility !== "visible" || cs.display === "none") continue;
    // content of a closed <details>, content-visibility: hidden, and the like
    if (el.checkVisibility && !el.checkVisibility({ contentVisibilityAuto: true, opacityProperty: false, visibilityProperty: true })) continue;
    // effective opacity (beats fade in and out; anything mostly transparent is not "shown")
    let op = 1;
    for (let a = el; a; a = a.parentElement) op *= Number(getComputedStyle(a).opacity);
    if (op < 0.6) continue;
    // outlined type (transparent fill, stroked) is read by its stroke
    let color = cs.color;
    if (/rgba\([^)]*,\s*0\)$/.test(color) || color === "transparent") {
      if (parseFloat(cs.webkitTextStrokeWidth) > 0) color = cs.webkitTextStrokeColor; else continue;
    }
    const r = document.createRange();
    r.selectNodeContents(n);
    for (const b of r.getClientRects()) {
      if (b.width < 4 || b.height < 6) continue;
      const x0 = Math.max(0, b.left), y0 = Math.max(0, b.top), x1 = Math.min(vw, b.right), y1 = Math.min(vh, b.bottom);
      if (x1 - x0 < 4 || y1 - y0 < 6) continue;
      // actually on top at its centre? Skips text clipped by an overflow box, visually hidden,
      // or covered by an opaque panel -- none of which a reader can see
      const hit = document.elementFromPoint((x0 + x1) / 2, (y0 + y1) / 2);
      if (!hit || !(hit === el || el.contains(hit))) continue;
      // under the fixed header? that text is read against the header's own scrim, included in the shot
      out.push({ text: text.slice(0, 60), x: x0, y: y0, w: x1 - x0, h: y1 - y0, color, op,
                 size: parseFloat(cs.fontSize), weight: Number(cs.fontWeight) || 400,
                 sel: el.tagName.toLowerCase() + (el.className && typeof el.className === "string" ? "." + el.className.trim().split(/\s+/).slice(0, 2).join(".") : "") });
    }
  }
  return out;
};

/** In the page: background samples per box, from a PNG data URL. */
const SAMPLE = async ([png, boxes]) => {
  const img = new Image();
  img.src = png;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(img, 0, 0);
  const s = img.width / innerWidth;
  return boxes.map(b => {
    const d = g.getImageData(Math.floor(b.x * s), Math.floor(b.y * s), Math.max(1, Math.floor(b.w * s)), Math.max(1, Math.floor(b.h * s))).data;
    const px = [];
    for (let i = 0; i < d.length; i += 4) px.push([d[i], d[i + 1], d[i + 2]]);
    const lum = p => 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    px.sort((a, b) => lum(a) - lum(b));
    return { light: px[Math.floor(px.length * 0.9)] || [0, 0, 0], dark: px[Math.floor(px.length * 0.1)] || [0, 0, 0] };
  });
};

const lin = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
const L = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
const ratio = (a, b) => { const x = L(a), y = L(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
const parseColor = s => { const m = s.match(/[\d.]+/g).map(Number); return { rgb: m.slice(0, 3), a: m[3] ?? 1 }; };
const over = (fg, a, bg) => fg.map((c, i) => c * a + bg[i] * (1 - a));

const failures = [];
let checked = 0;
try {
  for (const [w, h] of VIEWPORTS) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    await page.goto(`http://localhost:${PORT}/?still`, { waitUntil: "networkidle" });
    await page.waitForTimeout(3500);
    for (const [act, points] of Object.entries(POINTS)) {
      for (const p of points) {
        await page.evaluate(([a, p]) => {
          const el = document.getElementById(`act-${a}`);
          if (!el) return;
          const top = el.getBoundingClientRect().top + scrollY;
          scrollTo(0, top + Math.max(0, el.offsetHeight - innerHeight) * p);
        }, [act, p]);
        await page.waitForTimeout(1900);
        const pe = await page.addStyleTag({ content: "#story *, #lab *, #colophon *, .bar * { pointer-events: auto !important; }" });
        const runs = await page.evaluate(COLLECT);
        await pe.evaluate(n => n.remove());
        if (!runs.length) continue;
        const shot = await page.screenshot();
        const tag = await page.addStyleTag({ content: "*{color:transparent!important;-webkit-text-stroke:0!important;text-shadow:none!important;caret-color:transparent!important}" });
        await page.waitForTimeout(250);
        const bare = await page.screenshot();
        await tag.evaluate(n => n.remove());
        const bgs = await page.evaluate(SAMPLE, ["data:image/png;base64," + bare.toString("base64"), runs]);
        let bad = 0;
        runs.forEach((r, i) => {
          const { rgb, a } = parseColor(r.color);
          const alpha = a * r.op;
          const large = r.size >= 24 || (r.size >= 18.66 && r.weight >= 700);
          const need = large ? 3 : 4.5;
          // light text is hurt by the brightest background behind it, dark text by the darkest
          const textLum = L(rgb);
          const bg = textLum > 0.4 ? bgs[i].light : bgs[i].dark;
          const cr = ratio(over(rgb, alpha, bg), bg);
          checked++;
          if (cr < need) {
            bad++;
            failures.push({ viewport: `${w}x${h}`, act, p, ratio: +cr.toFixed(2), need, text: r.text, sel: r.sel,
                            box: [Math.round(r.x), Math.round(r.y), Math.round(r.w), Math.round(r.h)] });
          }
        });
        if (bad && outDir) {
          mkdirSync(outDir, { recursive: true });
          writeFileSync(`${outDir}/fail_${w}_${act}_${p}.png`, shot);
        }
        process.stdout.write(`${w}x${h} act ${act} @${p}: ${runs.length} runs${bad ? `, ${bad} below threshold` : ""}\n`);
      }
    }
    await ctx.close();
  }
} finally {
  await browser.close();
  server.kill();
}

if (failures.length) {
  // one line per distinct (text, selector) at its worst ratio
  const worst = new Map();
  for (const f of failures) {
    const k = f.sel + "|" + f.text;
    if (!worst.has(k) || worst.get(k).ratio > f.ratio) worst.set(k, f);
  }
  console.log(`\n${failures.length} text run(s) below threshold (${worst.size} distinct):`);
  for (const f of [...worst.values()].sort((a, b) => a.ratio - b.ratio)) {
    console.log(`  ${String(f.ratio).padStart(5)} < ${f.need}  ${f.viewport} act ${f.act} @${f.p}  ${f.sel}  “${f.text}”`);
  }
  if (outDir) writeFileSync(`${outDir}/contrast.json`, JSON.stringify(failures, null, 1));
  process.exit(1);
}
console.log(`\ncontrast OK: ${checked} text runs, every one at or above 4.5:1 (body) / 3:1 (large)`);

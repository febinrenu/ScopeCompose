/*  Text into textures -- large, oversampled, colour-managed.
 *
 *  v2's archive text was tiny, mirrored and unmanaged, which is why it read
 *  as soup. Here every text surface is drawn at about three times its
 *  on-screen size, on a dark frosted backing, tagged sRGB so three decodes it
 *  correctly, and shown on the front face only.
 */
import { CanvasTexture, LinearFilter, LinearMipMapLinearFilter, SRGBColorSpace } from "three";

export function wrap(ctx: CanvasRenderingContext2D, text: string, maxW: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const w of text.split(/\s+/)) {
    const t = line ? line + " " + w : w;
    if (ctx.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t;
  }
  if (line) out.push(line);
  return out;
}

function finish(c: HTMLCanvasElement): CanvasTexture {
  const tex = new CanvasTexture(c);
  tex.colorSpace = SRGBColorSpace;
  tex.minFilter = LinearMipMapLinearFilter;
  tex.magFilter = LinearFilter;
  tex.anisotropy = 8;
  tex.generateMipmaps = true;
  return tex;
}

// the page's own type system (styles/type.css), drawn into textures
const SERIF = "'Newsreader Variable', Georgia, serif";
const SANS = "'Mona Sans Variable', Arial, sans-serif";
const DISPLAY = "'Hubot Sans Variable', Arial, sans-serif";
const MONO = "'Martian Mono Variable', ui-monospace, monospace";

export interface PanelText { tag: string; text: string; meta: string; accent: string }

/** A passage's face: tag, the passage in large serif, a meta line. Returns the caret position (uv). */
export function passageTexture(p: PanelText, aspect = 3.2 / 1.25): { tex: CanvasTexture; caret: [number, number] } {
  const W = 2048, H = Math.round(W / aspect);
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d")!;
  // frosted backing: dark enough that ivory type is always above 10:1
  const grd = g.createLinearGradient(0, 0, W, H);
  grd.addColorStop(0, "rgba(10,14,22,0.92)"); grd.addColorStop(1, "rgba(5,7,12,0.96)");
  g.fillStyle = grd; g.fillRect(0, 0, W, H);
  const pad = 110;
  g.textBaseline = "top";
  g.fillStyle = p.accent;
  g.font = `700 46px ${MONO}`;
  g.fillText(p.tag.toUpperCase(), pad, pad - 14);
  // The serif's hairlines average to grey once the texture is minified (about 3x),
  // so the passage is drawn with a stroke of its own colour to thicken them.
  g.fillStyle = g.strokeStyle = "#f4f2ec";       // ivory stays below the bloom threshold
  g.lineWidth = 5; g.lineJoin = "round";
  g.font = `italic 400 132px ${SERIF}`;
  const lines = wrap(g, p.text, W - pad * 2).slice(0, 3);
  let endX = pad, endY = pad;
  lines.forEach((l, k) => { const y = pad + 92 + k * 146; g.strokeText(l, pad, y); g.fillText(l, pad, y); endX = pad + g.measureText(l).width; endY = y; });
  g.fillStyle = "#d5d9e0";
  g.font = `600 42px ${MONO}`;
  g.fillText(p.meta.toUpperCase(), pad, H - pad - 12);
  return { tex: finish(c), caret: [(endX + 26) / W, 1 - (endY + 70) / H] };
}

/** A retrieved card: id and passage text in clean sans. */
export function cardTexture(id: string, text: string, rank: number): CanvasTexture {
  const W = 1536, H = 1024;
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d")!;
  g.fillStyle = "rgba(9,13,20,0.9)"; g.fillRect(0, 0, W, H);
  const pad = 90;
  g.textBaseline = "top";
  g.fillStyle = "#7ff0ff"; g.font = `600 44px ${MONO}`;
  g.fillText(`#${rank}  ·  ${id.toUpperCase()}`, pad, pad);
  g.fillStyle = "#f4f2ec"; g.font = `500 92px ${SANS}`;
  wrap(g, text, W - pad * 2).slice(0, 6).forEach((l, k) => g.fillText(l, pad, pad + 120 + k * 114));
  return finish(c);
}

/** A short label plate, for in-world signage (the contract gate, station names). */
export function labelTexture(text: string, sub: string, accent = "#f4f2ec"): CanvasTexture {
  const W = 1024, H = 256;
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d")!;
  g.textBaseline = "middle";
  g.fillStyle = accent; g.font = `760 92px ${DISPLAY}`;
  g.fillText(text, 24, 92);
  g.globalAlpha = 0.75; g.fillStyle = "#c9cdd6"; g.font = `500 44px ${MONO}`;
  g.fillText(sub.toUpperCase(), 28, 190);
  return finish(c);
}

/** Canvas the wordmark and return its outline as world-space points. */
export function wordPoints(text: string, worldWidth: number, count: number): Float32Array {
  const W = 2400, H = 520;
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d", { willReadFrequently: true })!;
  g.fillStyle = "#fff"; g.textBaseline = "middle"; g.textAlign = "center";
  let size = 400;
  g.font = `800 ${size}px ${DISPLAY}`;
  while (g.measureText(text).width > W * 0.96 && size > 40) { size -= 10; g.font = `800 ${size}px ${DISPLAY}`; }
  g.fillText(text, W / 2, H / 2);
  const data = g.getImageData(0, 0, W, H).data;
  const filled: number[] = [];
  for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) if (data[(y * W + x) * 4 + 3] > 128) filled.push(x, y);
  const out = new Float32Array(count * 2);
  const k = worldWidth / W;
  let s = 1;
  for (let i = 0; i < count; i++) {
    s = (s * 16807) % 2147483647;
    const j = (s % (filled.length / 2)) * 2;
    out[i * 2] = (filled[j] - W / 2) * k;
    out[i * 2 + 1] = -(filled[j + 1] - H / 2) * k;
  }
  return out;
}

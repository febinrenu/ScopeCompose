/*  ?debug — frame-time percentiles, draw calls, tier and DPR, and a button
 *  that deliberately loses the WebGL context so recovery can be checked.
 */
export function mountHud(get: () => { calls: number; triangles: number; points: number; px: number; stepped: boolean } | null,
  tier: number, cursor: () => number, lose: () => void) {
  const el = document.createElement("div");
  el.style.cssText = "position:fixed;left:10px;bottom:10px;z-index:99;font:11px/1.5 ui-monospace,monospace;background:rgba(0,0,0,.78);color:#cfe;padding:8px 10px;border:1px solid #345;white-space:pre";
  const btn = document.createElement("button");
  btn.textContent = "lose context";
  btn.style.cssText = "display:block;margin-top:6px;font:inherit;background:#222;color:#fcc;border:1px solid #633;cursor:pointer";
  btn.onclick = lose;
  const txt = document.createElement("div");
  el.append(txt, btn);
  document.body.appendChild(el);
  const frames: number[] = [];
  let last = performance.now();
  const loop = (t: number) => {
    frames.push(t - last); last = t;
    if (frames.length > 120) frames.shift();
    const s = frames.slice().sort((a, b) => a - b);
    const p = (q: number) => s[Math.floor(s.length * q)]?.toFixed(1) ?? "-";
    const st = get();
    txt.textContent = `tier ${tier}  cursor ${cursor().toFixed(3)}\nframe p50 ${p(0.5)}ms  p90 ${p(0.9)}ms\n`
      + (st ? `calls ${st.calls}  tris ${st.triangles}  pts ${st.points}\ndpr ${st.px}${st.stepped ? "  (stepped down)" : ""}` : "no world");
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

/*  ScopeCompose - motion layer
 *
 *  Loaded before app.js so the field is ready when the engine first reports
 *  a relation.
 *
 *  Everything here is GPU work or transform/opacity. No layout is read in a
 *  scroll handler, nothing is pinned, and native scrolling is never
 *  intercepted -- the page stays as smooth as the browser can make it, which
 *  on a high-refresh display is smoother than any scroll-smoothing library.
 *
 *  The field is not decoration. The two regions ARE the two applicability
 *  sets, and their geometry is driven by whatever scope/attributes.py::compare
 *  returned for the pair currently on screen.
 */
"use strict";

(function () {
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FINE = matchMedia("(pointer: fine)").matches;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  /* ====================================================================
     1. The scope field - WebGL.
     Two soft regions blended through fbm-distorted space. Their centres
     and radii are the live set relation; the distortion is what stops it
     looking like two CSS gradients.
     ==================================================================== */

  const VERT = `
    attribute vec2 p;
    void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;

  const FRAG = `
    precision highp float;
    uniform vec2  u_res;
    uniform float u_t;
    uniform vec3  u_a;      // x, y, radius
    uniform vec3  u_b;
    uniform vec3  u_ca;     // colour a
    uniform vec3  u_cb;
    uniform float u_amp;    // master opacity

    // value noise + fbm. Cheap, and enough to break up the circles.
    float hash(vec2 p){
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
    }
    float noise(vec2 p){
      vec2 i = floor(p), f = fract(p);
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
                 mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
    }
    float fbm(vec2 p){
      float v = 0.0, a = 0.5;
      for (int i = 0; i < 5; i++){ v += a * noise(p); p *= 2.02; a *= 0.5; }
      return v;
    }

    void main(){
      vec2 uv = gl_FragCoord.xy / u_res;
      float ar = u_res.x / u_res.y;
      vec2 p = vec2(uv.x * ar, uv.y);

      // Flow the sampling space rather than the shapes: the regions keep
      // their meaning while the field around them stays alive.
      float n = fbm(p * 1.7 + vec2(u_t * 0.035, u_t * -0.022));
      vec2 q = p + (n - 0.5) * 0.30;

      vec2 ca = vec2(u_a.x * ar, u_a.y);
      vec2 cb = vec2(u_b.x * ar, u_b.y);

      float da = length(q - ca) / max(u_a.z, 0.001);
      float db = length(q - cb) / max(u_b.z, 0.001);

      float fa = exp(-da * da * 1.9);
      float fb = exp(-db * db * 1.9);

      // Normalised rather than additive: summing blew the overlap out to
      // white, which lost the one thing the overlap is supposed to show.
      // Dividing keeps each region's hue and lets the overlap read as a
      // blend of the two.
      vec3 col = (u_ca * fa + u_cb * fb) / max(fa + fb, 1.0);

      // A soft contour where each field crosses its own threshold. This is
      // the applicability boundary -- the thing the whole project is about --
      // so it is drawn rather than implied.
      float ra = smoothstep(0.30, 0.355, fa) - smoothstep(0.355, 0.41, fa);
      float rb = smoothstep(0.30, 0.355, fb) - smoothstep(0.355, 0.41, fb);
      col += u_ca * ra * 0.85 + u_cb * rb * 0.85;

      float m = clamp(fa + fb + (ra + rb) * 0.7, 0.0, 1.0);

      float grain = (hash(gl_FragCoord.xy * 0.7 + u_t) - 0.5) * 0.028;
      // Ceiling well under 1: this sits behind the headline and must never
      // compete with it.
      gl_FragColor = vec4(col + grain, m * u_amp * 0.46);
    }`;

  function hexToRgb(h){
    h = (h || "").trim().replace("#", "");
    if (h.length === 3) h = h.split("").map(c => c + c).join("");
    const n = parseInt(h || "4ECBA5", 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }

  const Field = (function () {
    const cvs = $("#field");
    if (!cvs || REDUCED) return null;

    let gl = null;
    try {
      gl = cvs.getContext("webgl", { alpha: true, antialias: false,
                                     premultipliedAlpha: false, depth: false });
    } catch (e) { gl = null; }
    if (!gl) return null;

    function shader(type, src){
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) return null;
      return s;
    }
    const vs = shader(gl.VERTEX_SHADER, VERT), fs = shader(gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return null;

    const prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const U = n => gl.getUniformLocation(prog, n);
    const uRes = U("u_res"), uT = U("u_t"), uA = U("u_a"), uB = U("u_b"),
          uCa = U("u_ca"), uCb = U("u_cb"), uAmp = U("u_amp");

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // target / current, lerped - the relation changes are animated, not cut
    let T = { ax:.40, ay:.52, ar:.30, bx:.56, by:.52, br:.15 };
    let C = Object.assign({}, T);
    let amp = 0, ampT = 0, t0 = 0, raf = 0, visible = false, sized = 0;

    function resize(){
      const r = cvs.getBoundingClientRect();
      const dpr = Math.min(devicePixelRatio || 1, 1.5);   // capped: this is ambience
      const w = Math.max(1, Math.round(r.width * dpr));
      const h = Math.max(1, Math.round(r.height * dpr));
      if (w === cvs.width && h === cvs.height) return;
      cvs.width = w; cvs.height = h;
      gl.viewport(0, 0, w, h);
    }

    function frame(now){
      if (!visible) { raf = 0; return; }
      if (!t0) t0 = now;
      const t = (now - t0) / 1000;

      if (now - sized > 400) { resize(); sized = now; }

      for (const k in T) C[k] += (T[k] - C[k]) * 0.055;
      amp += (ampT - amp) * 0.04;

      const cs = getComputedStyle(document.documentElement);
      const ca = hexToRgb(cs.getPropertyValue("--field-a"));
      const cb = hexToRgb(cs.getPropertyValue("--field-b"));

      gl.uniform2f(uRes, cvs.width, cvs.height);
      gl.uniform1f(uT, t);
      gl.uniform3f(uA, C.ax, C.ay, C.ar);
      gl.uniform3f(uB, C.bx, C.by, C.br);
      gl.uniform3f(uCa, ca[0], ca[1], ca[2]);
      gl.uniform3f(uCb, cb[0], cb[1], cb[2]);
      gl.uniform1f(uAmp, amp);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      raf = requestAnimationFrame(frame);
    }

    resize();
    addEventListener("resize", () => { sized = 0; }, { passive: true });

    new IntersectionObserver(es => {
      visible = es[0].isIntersecting;
      if (visible && !raf) raf = requestAnimationFrame(frame);
    }, { threshold: 0 }).observe(cvs);

    return {
      /** Drive the geometry from a SetRelation. */
      setRelation(rel){
        if (rel === "disjoint")          T = { ax:.26, ay:.54, ar:.20, bx:.74, by:.50, br:.20 };
        else if (rel === "overlapping")  T = { ax:.40, ay:.54, ar:.25, bx:.60, by:.50, br:.25 };
        else if (rel === "equal")        T = { ax:.50, ay:.52, ar:.25, bx:.50, by:.52, br:.25 };
        else                             T = { ax:.46, ay:.52, ar:.34, bx:.57, by:.50, br:.14 };
      },
      reveal(){ ampT = 1; },
    };
  })();

  window.SCField = Field;
  if (Field) window.__SC_GL = true;        // app.js skips its 2D fallback
  // Observable from the DOM, so "is the shader actually running" is a fact
  // rather than an assumption when debugging on someone else's machine.
  document.documentElement.classList.add(Field ? "gl" : "no-gl");

  /* ====================================================================
     2. Line and word reveals.
     Headings are split into words inside overflow-hidden lines, then
     slid up on a stagger. Split once, at load; after that it is transform
     and opacity only.
     ==================================================================== */

  function splitInto(node, out){
    node.childNodes.forEach(child => {
      if (child.nodeType === 3){
        const words = child.textContent.split(/(\s+)/);
        words.forEach(w => {
          if (!w.trim()){ out.push(document.createTextNode(w)); return; }
          const mask = document.createElement("span");
          mask.className = "msk";
          const inner = document.createElement("span");
          inner.className = "msk-i";
          inner.textContent = w;
          mask.appendChild(inner);
          out.push(mask);
        });
      } else if (child.nodeName === "BR"){
        out.push(child.cloneNode());
      } else {
        // <em> and friends: keep the element, split what is inside it
        const clone = child.cloneNode(false);
        const inner = [];
        splitInto(child, inner);
        inner.forEach(n => clone.appendChild(n));
        out.push(clone);
      }
    });
  }

  function prepareSplits(){
    $$("[data-split]").forEach(el => {
      const out = [];
      splitInto(el, out);
      el.textContent = "";
      out.forEach(n => el.appendChild(n));
      const inners = $$(".msk-i", el);
      inners.forEach((n, i) => { n.style.transitionDelay = (i * 42) + "ms"; });
    });
  }

  function observeSplits(){
    const io = new IntersectionObserver((es, o) => {
      es.forEach(e => {
        if (!e.isIntersecting) return;
        e.target.classList.add("shown");
        o.unobserve(e.target);
      });
    }, { rootMargin: "0px 0px -60px 0px", threshold: 0.01 });
    $$("[data-split]").forEach(el => io.observe(el));
  }

  /* ====================================================================
     3. Cursor. Two lerped layers; the ring lags the dot, which is what
     makes it feel weighted rather than glued to the pointer.
     ==================================================================== */

  function cursor(){
    if (!FINE || REDUCED) return;
    const dot = document.createElement("div"); dot.className = "cur-dot";
    const ring = document.createElement("div"); ring.className = "cur-ring";
    const label = document.createElement("span"); label.className = "cur-lab";
    ring.appendChild(label);
    document.body.append(dot, ring);

    let mx = innerWidth / 2, my = innerHeight / 2;
    let dx = mx, dy = my, rx = mx, ry = my, on = false;

    addEventListener("pointermove", e => {
      mx = e.clientX; my = e.clientY;
      if (!on){ on = true; document.body.classList.add("has-cur"); }
    }, { passive: true });

    addEventListener("pointerdown", () => ring.classList.add("down"), { passive: true });
    addEventListener("pointerup", () => ring.classList.remove("down"), { passive: true });

    // Delegated, so it covers nodes rendered later by app.js
    document.addEventListener("pointerover", e => {
      const t = e.target.closest("[data-cur], a, button, select, input, .cell, .row, .stage");
      if (!t){ ring.classList.remove("big"); label.textContent = ""; return; }
      ring.classList.add("big");
      label.textContent = t.getAttribute("data-cur") || "";
    }, { passive: true });

    (function loop(){
      dx += (mx - dx) * 0.55; dy += (my - dy) * 0.55;
      rx += (mx - rx) * 0.16; ry += (my - ry) * 0.16;
      dot.style.transform  = `translate3d(${dx}px,${dy}px,0) translate(-50%,-50%)`;
      ring.style.transform = `translate3d(${rx}px,${ry}px,0) translate(-50%,-50%)`;
      requestAnimationFrame(loop);
    })();
  }

  /* ====================================================================
     4. Magnetic elements. Pointer-local, so no scroll work at all.
     ==================================================================== */

  function magnetic(){
    if (!FINE || REDUCED) return;
    $$("[data-mag]").forEach(el => {
      const strength = Number(el.getAttribute("data-mag")) || 0.3;
      let raf = 0, tx = 0, ty = 0;
      function apply(){ el.style.transform = `translate3d(${tx}px,${ty}px,0)`; raf = 0; }
      el.addEventListener("pointermove", e => {
        const r = el.getBoundingClientRect();
        tx = (e.clientX - (r.left + r.width / 2)) * strength;
        ty = (e.clientY - (r.top + r.height / 2)) * strength;
        if (!raf) raf = requestAnimationFrame(apply);
      }, { passive: true });
      el.addEventListener("pointerleave", () => {
        tx = ty = 0;
        if (!raf) raf = requestAnimationFrame(apply);
      }, { passive: true });
    });
  }

  /* ====================================================================
     5. Intro. One orchestrated sequence, then the page is just a page.
     ==================================================================== */

  function intro(){
    const veil = $("#veil");
    const run = () => {
      document.documentElement.classList.add("ready");
      if (Field) Field.reveal();
      if (veil) setTimeout(() => veil.remove(), 1400);
    };
    if (REDUCED){
      document.documentElement.classList.add("ready", "no-intro");
      if (veil) veil.remove();
      return;
    }
    // Wait for fonts so the headline does not reveal in a fallback face and
    // then reflow into the real one mid-animation.
    const fonts = document.fonts ? document.fonts.ready : Promise.resolve();
    Promise.race([fonts, new Promise(r => setTimeout(r, 1200))]).then(() => {
      requestAnimationFrame(() => setTimeout(run, 80));
    });
  }

  /* ====================================================================
     6. Scroll-velocity readout for the ambient layer only. Content is
     never transformed on scroll - that is what causes jank.
     ==================================================================== */

  function velocity(){
    if (REDUCED) return;
    let last = scrollY, v = 0, raf = 0;
    const root = document.documentElement;
    function tick(){
      const now = scrollY;
      v += ((now - last) - v) * 0.2;
      last = now;
      root.style.setProperty("--vel", Math.max(-1, Math.min(1, v / 55)).toFixed(3));
      if (Math.abs(v) > 0.1) raf = requestAnimationFrame(tick);
      else { raf = 0; root.style.setProperty("--vel", "0"); }
    }
    addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(tick); },
                     { passive: true });
  }

  /* ---- boot ---- */
  prepareSplits();
  observeSplits();
  cursor();
  magnetic();
  velocity();
  intro();
})();

/*  ScopeCompose - application layer
 *
 *  All DOM wiring. The computation it displays lives in engines.js.
 */
"use strict";
document.documentElement.classList.add("js");
// data/corpus.js assigns window.SC_CORPUS. Generated from the benchmark by
// site/build_data.py, so the page can never drift from benchmark/data/corpus.jsonl.
const CORPUS = window.SC_CORPUS;
if (!Array.isArray(CORPUS) || !CORPUS.length){
  document.body.insertAdjacentHTML("afterbegin",
    '<p style="padding:20px;font:14px system-ui;color:#c1441a">'
    + "Corpus data failed to load. Run <code>python site/build_data.py</code>, then reload.</p>");
}
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ======================================================================
   ENGINE 3 — selection vs composition, computed per reader.
   ====================================================================== */
const BRANCHY = CORPUS.filter(r => (r.b || []).length >= 2);
let heroRec = BRANCHY.find(r => r.id === "wp1_fin_imp_003") || BRANCHY[0];
let heroMode = "sel";          // sel | comp
let heroKeep = 0;              // which branch selection kept
let persona = {};

const heroInst = $("#heroInst");
BRANCHY.forEach(r => {
  const o = document.createElement("option");
  o.value = r.id;
  o.textContent = r.id + "  —  " + (r.q.length > 38 ? r.q.slice(0, 38) + "…" : r.q);
  heroInst.appendChild(o);
});
heroInst.value = heroRec.id;
heroInst.addEventListener("change", () => {
  heroRec = BRANCHY.find(r => r.id === heroInst.value);
  heroKeep = 0; buildPersona(); renderHero();
});
$("#mSel").addEventListener("click", () => setMode("sel"));
$("#mComp").addEventListener("click", () => setMode("comp"));
function setMode(m){
  heroMode = m;
  $("#mSel").setAttribute("aria-pressed", String(m === "sel"));
  $("#mComp").setAttribute("aria-pressed", String(m === "comp"));
  renderHero();
}

/** Build reader controls from the dimensions the branches actually constrain. */
function buildPersona(){
  const dims = new Map();
  (heroRec.b || []).forEach(b => (b.at || []).forEach(a => {
    const k = String(a.n).trim().toLowerCase();
    if (!dims.has(k)) dims.set(k, {kind:a.k, values:new Set(), lo:null, hi:null});
    const d = dims.get(k);
    (a.v || []).forEach(v => d.values.add(String(v).trim().toLowerCase()));
    if (a.b !== undefined && a.b !== null) d.values.add(String(a.b).toLowerCase());
    if (a.lo !== undefined && a.lo !== null) d.lo = (d.lo === null) ? a.lo : Math.min(d.lo, a.lo);
    if (a.hi !== undefined && a.hi !== null) d.hi = (d.hi === null) ? a.hi : Math.max(d.hi, a.hi);
  }));

  const host = $("#personaCtl");
  host.textContent = "";
  persona = {};
  if (!dims.size){
    host.innerHTML = '<span class="note" style="margin:0">This instance’s branches carry no typed '
      + 'attributes, so membership cannot be evaluated exactly — the relation falls back to '
      + 'entailment. Pick another instance to drive the reader controls.</span>';
    return;
  }
  dims.forEach((d, name) => {
    const wrap = document.createElement("span");
    wrap.style.cssText = "display:inline-flex;align-items:center;gap:6px";
    const lab = document.createElement("span");
    lab.className = "lab"; lab.textContent = name.replace(/_/g, " ");
    wrap.appendChild(lab);

    if (d.kind === "numeric_range"){
      const inp = document.createElement("input");
      inp.type = "number"; inp.style.width = "92px";
      const mid = (isFinite(d.lo) && d.lo !== null) ? d.lo : 0;
      inp.value = String(mid); persona[name] = mid;
      inp.addEventListener("input", () => { persona[name] = inp.value === "" ? null : Number(inp.value); renderHero(); });
      wrap.appendChild(inp);
    } else {
      const sel = document.createElement("select");
      const vals = Array.from(d.values);
      vals.forEach(v => { const o = document.createElement("option"); o.value = v;
        o.textContent = v.replace(/_/g, " "); sel.appendChild(o); });
      const other = document.createElement("option");
      other.value = "__other__"; other.textContent = "— neither —"; sel.appendChild(other);
      sel.value = "__other__"; persona[name] = "__other__";
      sel.addEventListener("change", () => { persona[name] = sel.value; renderHero(); });
      wrap.appendChild(sel);
    }
    host.appendChild(wrap);
  });
}

function renderHero(){
  const r = heroRec, bs = r.b || [];
  $("#heroQ").textContent = "“" + r.q + "”";
  $("#heroProv").innerHTML = '<span class="ic">' + r.id + '</span> &middot; '
    + r.con + " construction &middot; " + (r.sep || "—").replace(/_/g, " ")
    + " &middot; " + r.dom.replace(/_/g, " ") + " &middot; " + r.prov;

  // --- the two determinations, computed ---
  // Compare the default branch against the first exception. Roles come from
  // the applicability relation, never from the order the passages arrived in.
  let i0 = 0, i1 = 1;
  const defAt = bs.findIndex(b => b.def);
  if (defAt >= 0){ i0 = defAt; i1 = bs.findIndex((b, k) => k !== defAt); }
  const app = i => bs[i].def ? {def:true} : {def:false, at:bs[i].at};
  const setRel = compare(app(i0), app(i1));
  const agree = bs[i0].out.trim().toLowerCase() === bs[i1].out.trim().toLowerCase();
  const [relation, why] = combine(setRel, agree);
  let [action, areason] = ROUTE[relation] || ["—", ""];

  // Second-order structure is outside this project's first-order scope: more
  // than one surviving exception is FLAGGED and routed to selection rather
  // than silently mis-composed (contract/routing.py::route_with_flags).
  const nExc = bs.length - (defAt >= 0 ? 1 : 0);
  const multi = nExc > 1 && action === "compose";
  if (multi){
    action = "select";
    areason = "two independent exceptions on one rule — second-order structure, flagged "
      + "rather than mis-composed. How often real data contains this pattern is a result in "
      + "its own right";
  }

  // canonical: the exception sits inside the default, never argument-relative
  const canonical = (setRel === REL.SUPERSET) ? REL.SUBSET : setRel;
  const defIdx = (setRel === REL.SUBSET) ? i1 : i0;

  $("#heroCalc").innerHTML =
      row("set relation", canonical + (canonical === REL.SUBSET ? "  (exception ⊂ default)" : ""))
    + row("outcomes", agree ? "agree" : "disagree")
    + row("scope relation", relation)
    + row("routes to", action + (multi ? "  (multi-exception flag)" : ""))
    + row("evidence", (bs[i0].at || bs[i1].at) ? "attributes (exact)" : "default branch")
    + row("default branch", bs[defIdx].id)
    + row("branches", bs.length)
    + row("gold label", r.rel || "—");
  $("#heroRationale").textContent = why + ". Routing: " + areason + ".";

  // --- branches, and whether the reader falls inside each ---
  const inside = bs.map(b => satisfies({def:b.def, at:b.at}, persona));
  const host = $("#heroBranches");
  host.textContent = "";
  bs.forEach((b, i) => {
    const el = document.createElement("div");
    el.className = "branch";
    const kept = (heroMode === "comp") || (i === heroKeep);
    if (heroMode === "comp") el.classList.add("live");
    else if (i === heroKeep) el.classList.add("live");
    else el.classList.add("gone");

    el.innerHTML =
      '<span class="bcond">' + (b.def ? "DEFAULT — applies generally"
        : "IF · " + esc(b.cond || b.desc)) + "</span>"
      + '<span class="bout">' + esc(b.out) + "</span>"
      + '<div class="bmeta">scope: <span class="ic">' + esc(b.desc) + "</span>"
      + " &middot; from " + (b.sup || "—")
      + (inside[i] ? ' &middot; <span class="pill p-ok">your reader is inside this branch</span>' : "")
      + "</div>";

    const slot = document.createElement("div");
    slot.className = "bslot";
    slot.appendChild(el);
    if (heroMode === "sel" && i !== heroKeep){
      if (REDUCED) slot.classList.add("collapsed");
      // Two frames: the slot must be laid out at 1fr before transitioning to
      // 0fr, or the browser has nothing to animate from.
      else requestAnimationFrame(() => requestAnimationFrame(() => slot.classList.add("collapsed")));
    }
    host.appendChild(slot);
  });

  // --- the verdict, derived ---
  const v = $("#heroVerdict");
  const which = $("#selWhich");
  if (heroMode === "comp"){
    which.hidden = true;
    const n = inside.filter(Boolean).length;
    v.className = "verdict good";
    v.innerHTML = "<b>Composition keeps every branch.</b> The answer states the general rule and "
      + "its condition together, so it is correct for this reader and for every other reader "
      + "simultaneously. " + n + " of " + bs.length + " branches apply to the reader you set.";
  } else {
    which.hidden = false;
    which.textContent = "kept " + bs[heroKeep].id + " — click to switch";
    which.style.cursor = "pointer";
    which.onclick = () => { heroKeep = (heroKeep + 1) % bs.length; renderHero(); };
    const lost = bs.map((b, i) => i).filter(i => i !== heroKeep);
    const wrongFor = lost.filter(i => inside[i]);
    if (wrongFor.length){
      v.className = "verdict bad";
      v.innerHTML = "<b>Wrong for this reader.</b> They fall inside "
        + lost.map(i => "<i>" + esc(bs[i].desc) + "</i>").join(" and ")
        + ", but selection emitted the other branch. The deleted text was true — and nothing "
        + "in the answer signals that a branch was dropped. That is Silent Scope Omission.";
    } else {
      v.className = "verdict";
      v.innerHTML = "<b>Right for this reader, by luck.</b> They happen to fall outside the "
        + "discarded branch, so the emitted answer is correct for them. Switch the reader above "
        + "— or switch which branch selection kept — and it stops being correct. "
        + "<b>No single branch is right for everyone.</b>";
    }
  }
  paintField(setRel, inside);
}
const row = (k, v) => "<dt>" + k + "</dt><dd>" + esc(String(v)) + "</dd>";
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

/* ======================================================================
   The scope field — two regions whose geometry IS the current relation.
   One rAF loop, paused by IntersectionObserver when offscreen.
   ====================================================================== */
const cvs = $("#field"), ctx = cvs.getContext("2d");
let target = {ax:.38, ar:.23, bx:.52, br:.12}, cur = Object.assign({}, target);
let dots = [], running = false, dpr = 1;

function paintField(setRel, inside){
  // motion.js owns the field when WebGL is available; this 2D path is the
  // fallback for contexts that cannot give us a GL context.
  if (window.SCField) window.SCField.setRelation(setRel);
  if (setRel === REL.DISJOINT)         target = {ax:.30, ar:.17, bx:.70, br:.17};
  else if (setRel === REL.OVERLAPPING) target = {ax:.42, ar:.21, bx:.60, br:.21};
  else if (setRel === REL.EQUAL)       target = {ax:.50, ar:.21, bx:.50, br:.21};
  else                                 target = {ax:.47, ar:.27, bx:.55, br:.12};
  dots = (inside || []).map((v, i) => ({on:v, i}));
}
function sizeField(){
  const r = cvs.getBoundingClientRect();
  dpr = Math.min(devicePixelRatio || 1, 2);
  cvs.width = Math.max(1, r.width * dpr); cvs.height = Math.max(1, r.height * dpr);
}
function draw(){
  if (!running) return;
  const w = cvs.width, h = cvs.height;
  for (const k in target) cur[k] += (target[k] - cur[k]) * 0.07;
  ctx.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  const ca = cs.getPropertyValue("--field-a").trim() || "#4ECBA5";
  const cb = cs.getPropertyValue("--field-b").trim() || "#F2794B";
  const cy = h * 0.52, unit = Math.min(w, h * 1.7);
  blob(cur.ax * w, cy, cur.ar * unit, ca);
  blob(cur.bx * w, cy, cur.br * unit, cb);
  requestAnimationFrame(draw);
}
function blob(x, y, r, col){
  const g = ctx.createRadialGradient(x, y, r * 0.15, x, y, r);
  g.addColorStop(0, col + "38"); g.addColorStop(0.62, col + "18"); g.addColorStop(1, col + "00");
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
}
if (!REDUCED && !window.__SC_GL){
  sizeField();
  addEventListener("resize", sizeField, {passive:true});
  new IntersectionObserver(es => {
    const vis = es[0].isIntersecting;
    if (vis && !running){ running = true; draw(); } else if (!vis) running = false;
  }, {threshold:0}).observe(cvs);
}

/* ======================================================================
   ENGINE 2 — corpus explorer.
   ====================================================================== */
const FILTERS = [["fType","ty"],["fRel","rel"],["fSep","sep"],["fSplit","sp"],["fDom","dom"]];
FILTERS.forEach(([id, key]) => {
  const sel = $("#" + id);
  const vals = Array.from(new Set(CORPUS.map(r => r[key]).filter(Boolean))).sort();
  vals.forEach(v => { const o = document.createElement("option");
    o.value = v; o.textContent = v.replace(/_/g, " "); sel.appendChild(o); });
  sel.addEventListener("change", renderList);
});
$("#fq").addEventListener("input", debounce(renderList, 140));
function debounce(fn, ms){ let t; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; }

let selectedId = null;
function matching(){
  const q = $("#fq").value.trim().toLowerCase();
  return CORPUS.filter(r => {
    for (const [id, key] of FILTERS){ const v = $("#" + id).value; if (v && r[key] !== v) return false; }
    if (!q) return true;
    if (r.q.toLowerCase().includes(q) || r.id.toLowerCase().includes(q)) return true;
    return (r.p || []).some(p => p.t.toLowerCase().includes(q));
  });
}
function renderList(){
  const rows = matching();
  $("#expCount").textContent = rows.length + " of " + CORPUS.length;
  const host = $("#expList");
  host.textContent = "";
  const frag = document.createDocumentFragment();
  rows.slice(0, 400).forEach(r => {
    const b = document.createElement("button");
    b.className = "row"; b.type = "button";
    if (r.id === selectedId) b.setAttribute("aria-current", "true");
    b.innerHTML = '<span class="rid">' + esc(r.id) + "</span>"
      + '<div class="rq">' + esc(r.q) + "</div>"
      + '<div class="rtags"><span class="tag">' + esc(r.ty) + "</span>"
      + (r.rel ? '<span class="tag t-rel">' + esc(r.rel) + "</span>" : "")
      + '<span class="tag">' + esc(r.sp) + "</span>"
      + '<span class="tag">' + esc((r.sep || "").replace(/_/g, " ")) + "</span>"
      + ((r.b || []).length ? '<span class="tag">' + r.b.length + " branches</span>" : "")
      + "</div>";
    b.addEventListener("click", () => { selectedId = r.id; renderList(); renderDetail(r); });
    frag.appendChild(b);
  });
  host.appendChild(frag);
  if (rows.length > 400){
    const n = document.createElement("div");
    n.className = "count"; n.textContent = "showing first 400 of " + rows.length;
    host.appendChild(n);
  }
  if (!rows.length) host.innerHTML = '<div class="count">no instances match these filters</div>';
}
function renderDetail(r){
  const d = $("#expDet");
  if (!r){ d.innerHTML = '<p class="note" style="margin:0">Select an instance to see its passages, '
    + "gold labels, branch structure and provenance.</p>"; return; }
  let html = '<p style="font-size:17px; margin-bottom:4px">“' + esc(r.q) + "”</p>"
    + '<p class="note" style="margin-top:0"><span class="ic">' + esc(r.id) + "</span></p>"
    + '<dl class="kv" style="margin:16px 0 20px">'
    + row("type", r.ty) + row("scope relation", r.rel || "— (not a conditional conflict)")
    + row("routes to", r.rel ? (ROUTE[r.rel] || ["—"])[0] : "pass through")
    + row("construction", r.con) + row("separation", (r.sep || "—").replace(/_/g, " "))
    + row("split", r.sp) + row("domain", r.dom.replace(/_/g, " "))
    + row("annotation", r.prov) + "</dl>";

  html += '<span class="lab">Passages</span><div style="margin-top:10px">';
  (r.p || []).forEach(p => {
    html += '<div class="psg"><span class="pid">' + esc(p.id)
      + (p.d ? " · " + esc(p.d) : "") + (p.dt ? " · " + esc(p.dt) : "") + "</span>"
      + '<p class="ptx">' + esc(p.t) + "</p>"
      + (p.u ? '<p class="psrc">' + esc(p.u) + "</p>" : "") + "</div>";
  });
  html += "</div>";

  if ((r.b || []).length){
    html += '<span class="lab">Gold branches</span><div class="branches" style="margin-top:10px">';
    r.b.forEach(b => {
      html += '<div class="branch' + (b.def ? "" : " live") + '">'
        + '<span class="bcond">' + (b.def ? "DEFAULT" : "IF · " + esc(b.cond || b.desc)) + "</span>"
        + '<span class="bout">' + esc(b.out) + "</span>"
        + '<div class="bmeta">scope: <span class="ic">' + esc(b.desc) + "</span>"
        + (b.exp ? " &middot; " + esc(b.exp) : "") + " &middot; from " + esc(b.sup || "—")
        + ((b.at || []).length ? " &middot; " + b.at.length + " typed attribute"
            + (b.at.length > 1 ? "s" : "") : " &middot; no typed attributes") + "</div></div>";
    });
    html += "</div>";
  } else {
    html += '<p class="note"><b>No branch structure.</b> This instance carries detection labels '
      + "only. 86 of 128 instances are in this state, which is why the branch-scored metrics have "
      + "a 42-instance denominator.</p>";
  }
  d.innerHTML = html;
}

/* ======================================================================
   ENGINE 4 — power analysis. Port of metrics/power.py.
   ====================================================================== */
const BPI = 83 / 42;   // observed branches per annotated instance
function renderPower(){
  const eff = Number($("#sEff").value) / 100, dis = Number($("#sDis").value) / 100;
  $("#vEff").textContent = $("#sEff").value + "%";
  $("#vDis").textContent = $("#sDis").value + "%";
  const r = pairedSize(eff, dis, BPI);
  const note = $("#pwNote");
  if (!r.branches){
    $("#pwBranch").textContent = "n/a"; $("#pwInst").textContent = "n/a";
    note.innerHTML = "<b>Not a sample-size problem.</b> An effect larger than the discordant rate "
      + "cannot be expressed: every disagreement would have to favour one system and still fall short.";
  } else {
    $("#pwBranch").textContent = r.branches; $("#pwInst").textContent = r.instances;
    const have = r.instances <= 42;
    note.innerHTML = "At &alpha;&nbsp;=&nbsp;0.05 and 80% power, needing <b>" + r.discordant
      + "</b> discordant branches. The corpus has <b>42</b> annotated instances and <b>83</b> gold "
      + "branches today" + (have ? " — <b>which is already enough for this effect.</b>"
        : ", so this target needs <b>" + (r.instances - 42) + "</b> more annotated instances.");
  }
}
["sEff","sDis"].forEach(id => $("#" + id).addEventListener("input", renderPower));

/* ======================================================================
   Pipeline stage notes.
   ====================================================================== */
const STAGES = [
  ["A1 · Retrieval", "Dense retrieval over the document pool, returning the passage set a "
    + "normal RAG system would see. Nothing here is special — that is the point: the conflict "
    + "must be detectable from an ordinary retrieval, not a curated pair.", "retrieval/"],
  ["A2 · Detection", "Two-stage. A cheap lexical and embedding filter proposes candidate pairs, "
    + "then a trained head decides conflict or no conflict. Stage 1 alone sits at chance on this "
    + "data, which is why stage 2 exists and why both are reported separately.", "detection/stage1.py, detection/stage2.py"],
  ["A3 · Typing", "Five classes: conditional, factual, temporal, opinion, no_conflict. Only "
    + "conditional pairs continue; the rest route to the published strategy for that class and this "
    + "project makes no claim about them.", "detection/head.py"],
  ["A4 · Scope analysis", "The contribution. Two independent determinations — how the "
    + "applicability sets relate, and whether the outcomes agree — combined into the four-way "
    + "relation. Typed attributes give an exact answer; entailment is the fallback, not the default.",
    "scope/attributes.py, scope/relation.py"],
  ["B1 · Contrastive probing", "Extracts the condition and outcome of each branch, with a "
    + "grounding gate at τ = 0.5 that rejects any extracted span not supported by the passage. "
    + "An ungrounded branch is a hallucinated condition, and HCR measures exactly that.", "extraction/probing.py"],
  ["B2 · Composition operator", "Assembles the surviving branches into one structure, ordered "
    + "by the applicability relation rather than retrieval order. Second-order structure — an "
    + "exception carved back by a further exception — is flagged and routed to selection rather "
    + "than mis-composed.", "composition/operator.py"],
  ["B3 · Scoped generation", "Renders the composed structure into an answer that states each "
    + "condition explicitly, then checks faithfulness clause by clause against the source passages.",
    "generation/scoped_answer.py"]
];
function showStage(i){
  $$(".stage").forEach(g => g.classList.toggle("on", Number(g.dataset.s) === i));
  const [h, body, files] = STAGES[i];
  $("#stageNote").innerHTML = '<div class="sh">' + h + "</div>" + esc(body)
    + '<br><code style="display:inline-block;margin-top:9px">' + esc(files) + "</code>";
}
$$(".stage").forEach(g => {
  const i = Number(g.dataset.s);
  g.addEventListener("mouseenter", () => showStage(i));
  g.addEventListener("click", () => showStage(i));
  g.addEventListener("focus", () => showStage(i));
  g.addEventListener("keydown", e => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft"){
      e.preventDefault();
      const n = (i + (e.key === "ArrowRight" ? 1 : STAGES.length - 1)) % STAGES.length;
      const t = $('.stage[data-s="' + n + '"]'); if (t) t.focus();
    }
  });
});

/* ======================================================================
   Corpus counts: relation cells and composition table.
   ====================================================================== */
function countBy(key){
  const m = {};
  CORPUS.forEach(r => { const v = r[key]; if (v) m[v] = (m[v] || 0) + 1; });
  return m;
}
(function fillCounts(){
  const rel = countBy("rel");
  ["refinement","disjoint","redundant","opposed"].forEach(k => {
    const el = $("#c-" + k); if (el) el.textContent = rel[k] || 0;
  });
  const ty = countBy("ty"), sep = countBy("sep"), con = countBy("con"),
        sp = countBy("sp"), prov = countBy("prov");
  const branchy = CORPUS.filter(r => (r.b || []).length).length;
  const nBranch = CORPUS.reduce((s, r) => s + (r.b || []).length, 0);
  const rows = [
    ["instances, total", CORPUS.length, "every row below partitions this"],
    ["conditional conflicts", ty.conditional || 0, "the class this project addresses"],
    ["no conflict (distractors)", ty.no_conflict || 0, "guards against a detector that always fires"],
    ["factual / temporal / opinion", (ty.factual||0)+(ty.temporal||0)+(ty.opinion||0), "routed to prior work"],
    ["natural construction", con.natural || 0, "mined from real published documents"],
    ["synthetic split", con.split || 0, "authored to populate the thin relation cells"],
    ["cross-document separation", sep.cross_document || 0, "Tier 1 — the hard, scarce row"],
    ["same-guide separation", sep.same_guide || 0, "reported as its own row, never blended"],
    ["with gold branch structure", branchy, "the denominator for PR / SR / HCR / SCR"],
    ["gold branches", nBranch, "what the branch-scored metrics actually count"],
    ["two independent annotators", prov["two-annotator"] || 0, "κ measured on this class only"],
    ["model-proposed + adjudicated", prov["model-proposed+adjudicated"] || 0, "never pooled with the above"]
  ];
  $("#compTable").innerHTML = rows.map(([k, v, n]) =>
    "<tr><td>" + k + '</td><td class="num">' + v + '</td><td style="color:var(--muted);font-size:12.5px">'
    + n + "</td></tr>").join("");
})();

/* ======================================================================
   Theme, reveals, nav highlighting.
   ====================================================================== */
(function theme(){
  const KEY = "sc-theme";
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) {}
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  $("#themeBtn").addEventListener("click", () => {
    const now = document.documentElement.getAttribute("data-theme");
    const sysDark = matchMedia("(prefers-color-scheme: dark)").matches;
    const next = now ? (now === "dark" ? "light" : "dark") : (sysDark ? "light" : "dark");
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem(KEY, next); } catch (e) {}
  });
})();

if (!REDUCED && "IntersectionObserver" in window){
  const io = new IntersectionObserver((es, o) => {
    es.forEach(e => { if (e.isIntersecting){ e.target.classList.add("in"); o.unobserve(e.target); } });
  // Fixed pixels, not a percentage: a percentage rootMargin scales with
  // viewport height, so on a tall window the tail of the page never
  // intersects and its content stays invisible.
  }, {rootMargin:"0px 0px -40px 0px", threshold:0.01});
  $$(".rv").forEach((el, i) => { el.style.transitionDelay = (Math.min(i % 4, 3) * 55) + "ms"; io.observe(el); });
} else {
  $$(".rv").forEach(el => el.classList.add("in"));
}

(function navspy(){
  const links = $$("#nav a");
  const secs = links.map(a => $(a.getAttribute("href"))).filter(Boolean);
  const io = new IntersectionObserver(es => {
    es.forEach(e => {
      if (!e.isIntersecting) return;
      links.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
    });
  }, {rootMargin:"-45% 0px -50% 0px"});
  secs.forEach(s => io.observe(s));
})();

/* ======================================================================
   Marquee. Figures only, read from the corpus -- it is a readout, not
   decoration, so it says nothing the rest of the page does not.
   ====================================================================== */
(function marquee(){
  const host = $("#marquee");
  if (!host) return;
  const rel = countBy("rel"), ty = countBy("ty"), sep = countBy("sep");
  const branchy = CORPUS.filter(r => (r.b || []).length).length;
  const nBranch = CORPUS.reduce((s, r) => s + (r.b || []).length, 0);
  const items = [
    [CORPUS.length, "instances"], [ty.conditional || 0, "conditional conflicts"],
    [rel.refinement || 0, "refinement"], [rel.disjoint || 0, "disjoint"],
    [rel.redundant || 0, "redundant"], [rel.opposed || 0, "opposed"],
    [sep.cross_document || 0, "cross-document"], [branchy, "with branch structure"],
    [nBranch, "gold branches"], ["1.0000", "order invariance, 47 permutations"],
    ["0.34", "candidates per pairing"], ["29→80%", "distinct after re-pairing"],
    ["518", "tests passing"], ["κ 1.000", "n = 39, p_e = 0.540"],
    ["−12.5%", "decisive difference, interval spans zero"],
  ];
  const html = items.map(([v, k]) => "<span><b>" + v + "</b>" + k + "</span>").join("");
  host.innerHTML = html + html;    // doubled: the -50% keyframe seams exactly
})();

/* ---- boot ---- */
buildPersona();
renderHero();
renderList();
selectedId = CORPUS[0].id; renderList(); renderDetail(CORPUS[0]);
renderPower();
showStage(3);
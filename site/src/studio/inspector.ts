/*  The stage inspector: what one stage computed, drawn from its own events. */
import { esc, fx, human } from "../ui/dom";
import { allData, latest, STAGE_LABEL, type RunState } from "./model";

const pct = (x: number | null | undefined) => (x === null || x === undefined || !Number.isFinite(x) ? "—" : `${Math.round(x * 100)}%`);
const chip = (text: string, cls = "") => `<span class="chip ${cls}">${esc(text)}</span>`;
const bar = (v: number, max = 1, cls = "") =>
  `<span class="ib ${cls}"><i style="width:${Math.max(0, Math.min(100, (v / (max || 1)) * 100)).toFixed(1)}%"></i></span>`;

function cite(text: string): string {
  return esc(text).replace(/\(per (p\d+)\)/g, '<span class="cite">$1</span>');
}

function empty(msg: string) {
  return `<p class="ins-empty">${esc(msg)}</p>`;
}

function skippedNote(s: RunState, id: string): string {
  const sk = latest(s, id, "skipped");
  return sk ? `<p class="ins-skip">${chip("skipped")} ${esc(sk.reason || "")}</p>` : "";
}

// --------------------------------------------------------------------------- //

function setup(s: RunState) {
  const d = latest(s, "setup", "components");
  if (!d) return empty("Waiting for the engine…");
  const rows = Object.entries(d.components || {}).map(([k, v]) => `<dt>${esc(human(k))}</dt><dd>${esc(v)}</dd>`).join("");
  const t = d.thresholds || {};
  return `<dl class="kv">${rows}</dl>
    <p class="ins-note">Stage-1 band [${fx(t.low, 2)}, ${fx(t.high, 2)}], decision at ${fx(t.decide ?? 0.5, 2)}${t.tuned ? "" : " (placeholders until tuned)"}${d.tau_ground !== undefined ? ` · grounding gate τ = ${fx(d.tau_ground, 2)}` : ""}.</p>`;
}

function ingest(s: RunState) {
  const r = latest(s, "ingest", "benchmark record");
  if (r) return `<dl class="kv">
      <dt>record</dt><dd><code>${esc(r.id)}</code></dd><dt>question</dt><dd>${esc(r.query)}</dd>
      <dt>domain</dt><dd>${esc(human(r.domain))}</dd><dt>construction</dt><dd>${esc(r.construction)}${r.separation ? ` · ${esc(human(r.separation))}` : ""}</dd>
      <dt>split</dt><dd>${esc(r.split)}</dd>
      <dt>gold</dt><dd>${chip(r.gold_type, r.gold_type === "conditional" ? "warn" : "")} ${r.gold_relation ? chip(r.gold_relation) : ""} ${r.gold_branches} branch(es)</dd></dl>`;
  const c = latest(s, "ingest", "collection");
  if (c) return `<p><b>${esc(c.name)}</b> · ${c.documents.length} document(s) · ${c.passages} passage(s)</p>
    <ul class="ins-list">${c.documents.map((d: any) => `<li>${esc(d.name)} <span class="muted">${d.passages} passages · ${esc(d.kind)}</span></li>`).join("")}</ul>`;
  const i = latest(s, "ingest", "inline passages");
  return i ? `<p>${i.count} passage(s) given inline.</p>` : empty("Waiting…");
}

function rankTable(title: string, rows: any[], scoreKey = "score", extra?: (r: any) => string) {
  if (!rows?.length) return "";
  const max = Math.max(...rows.map(r => Math.abs(r[scoreKey]) || 0), 1e-9);
  return `<div class="rank"><h5>${title}</h5><ol>${rows.map(r => `
    <li class="${r.is_instance ? "own" : ""}"><span class="rk">${r.rank}</span>
      <span class="rt" title="${esc(r.text)}"><code>${esc(r.id)}</code> ${esc(r.text)}</span>
      <span class="rs">${bar(Math.abs(r[scoreKey]), max)}<em>${fx(r[scoreKey], 3)}</em></span>${extra ? extra(r) : ""}</li>`).join("")}</ol></div>`;
}

function retrieve(s: RunState) {
  const t = allData(s, "retrieve").find(d => d.bm25);
  const dec = latest(s, "retrieve", "decision");
  const ps = latest(s, "retrieve", "passages");
  let html = skippedNote(s, "retrieve");
  if (t) {
    html += `<p class="ins-note">${t.corpus_size} passages indexed · pool ${t.pool} per ranker · RRF k = ${t.rrf_k} · top ${t.top_k}${t.dense_on ? "" : " · dense off (offline: BM25 only)"}</p>
      <div class="rank-grid">${rankTable("BM25 (lexical)", t.bm25)}${t.dense_on ? rankTable("Dense (Contriever)", t.dense) : ""}
      ${rankTable("Fused (reciprocal rank)", t.fused, "score", r => `<span class="rb">${r.both ? chip("both", "ok") : ""}</span>`)}</div>`;
  }
  if (dec) html += `<p class="ins-note">${chip(dec.mode === "oracle" ? "oracle passages" : dec.mode === "given" ? "given" : "retrieved", dec.mode === "retrieved" ? "ok" : "warn")} ${esc(dec.note || "")}${dec.found !== undefined ? ` Hybrid retrieval found <b>${dec.found} of ${dec.of}</b> of them.` : ""}</p>`;
  if (ps) html += `<div class="psgs">${ps.passages.map((p: any) => `<div class="psg2"><span class="pid">${esc(p.id)} · ${esc(human(p.source_type))}${p.date ? " · " + esc(p.date) : ""}</span><p>${esc(p.text)}</p></div>`).join("")}</div>`;
  return html || empty("Waiting…");
}

const FEATURE_ORDER = ["exception_cues_max", "exception_cues_asymmetry", "restriction_asymmetry", "specificity_asymmetry",
  "numeric_clash", "year_clash", "numeric_overlap", "token_jaccard", "length_ratio", "temporal_cues_max", "opinion_cues_max",
  "exception_cues_i", "exception_cues_j", "restriction_cues_i", "restriction_cues_j", "qualifiers_i", "qualifiers_j"];

function detect(s: RunState) {
  const pairs = allData(s, "detect").filter(d => d.pair && d.nli);
  const thr = latest(s, "setup", "components")?.thresholds || { low: 0.35, high: 0.75, decide: 0.5 };
  if (!pairs.length) return empty("Waiting for stage 1…");
  const verdicts = latest(s, "detect", "verdicts");
  const cards = pairs.map(p => {
    const n = p.nli;
    const feats = FEATURE_ORDER.filter(k => k in (p.features || {})).map(k => {
      const v = Number(p.features[k]) || 0;
      return `<li class="${v ? "" : "zero"}"><span>${esc(human(k))}</span>${bar(Math.min(v, 3), 3)}<em>${fx(v, 2)}</em></li>`;
    }).join("");
    const pos = (x: number) => `${(x * 100).toFixed(1)}%`;
    return `<div class="det">
      <h5>${esc(p.pair.join(" × "))} ${p.is_conflict ? chip("conflict", "bad") : chip("no conflict")} ${p.escalate ? chip("escalates to stage 2", "warn") : ""}</h5>
      <div class="nli" title="entailment ${fx(n.entailment, 2)} · neutral ${fx(n.neutral, 2)} · contradiction ${fx(n.contradiction, 2)}">
        <i class="e" style="flex:${n.entailment}"></i><i class="n" style="flex:${n.neutral}"></i><i class="c" style="flex:${n.contradiction}"></i></div>
      <p class="nli-k"><span class="e">entail ${fx(n.entailment, 2)}</span><span class="n">neutral ${fx(n.neutral, 2)}</span><span class="c">contradict ${fx(n.contradiction, 2)}</span></p>
      <div class="gauge"><i style="width:${pos(p.score)}"></i>
        <b class="band" style="left:${pos(thr.low)};width:${pos(thr.high - thr.low)}"></b>
        <b class="mk" style="left:${pos(thr.decide ?? 0.5)}"></b><span class="gv" style="left:${pos(p.score)}">${fx(p.score, 3)}</span></div>
      <p class="nli-k"><span>0</span><span>band ${fx(thr.low, 2)}–${fx(thr.high, 2)} · conflict ≥ ${fx(thr.decide ?? 0.5, 2)}</span><span>1</span></p>
      <details><summary>17 pair features</summary><ul class="feat">${feats}</ul></details></div>`;
  }).join("");
  const v = verdicts ? `<table class="data mini"><thead><tr><th>pair</th><th>conflict</th><th>type</th><th>decided by</th><th>confidence</th></tr></thead><tbody>${
    verdicts.pairs.map((p: any) => `<tr><td>${esc(p.pair.join(" × "))}</td><td>${p.is_conflict ? "yes" : "no"}</td><td>${esc(p.type)}</td><td>${esc(human(p.decided_by))}</td><td>${fx(p.confidence, 2)}</td></tr>`).join("")}</tbody></table>` : "";
  return cards + v;
}

function type(s: RunState) {
  const t = latest(s, "type", "types");
  if (!t) return empty("Waiting…");
  return `<ul class="ins-list">${t.pairs.map((p: any) => `<li>${esc(p.pair.join(" × "))} → ${chip(p.type, p.type === "conditional" ? "warn" : p.is_conflict ? "bad" : "")} <span class="muted">${esc(human(p.decided_by))}</span></li>`).join("")}</ul>
    <p class="ins-note">Five classes: no_conflict, factual, temporal, opinion, conditional. Only conditional conflicts go on to scope analysis and composition.</p>`;
}

function policy(s: RunState) {
  const st = s.stages.policy;
  if (!st) return empty("Waiting…");
  const w = st.data.find(d => d.payload._warn);
  if (w) {
    const p = w.payload;
    if (p.policy === "as_run") return `<div class="banner warn"><b>${esc(w.title)}</b><p>As-run: the detector's verdict stands, so the answer states the general rule and nothing else. This is the failure the project measures: a missed conditional conflict silently drops a branch.</p></div>`;
    return `<div class="banner warn"><b>${esc(w.title)}</b>
      <p>Stage 1 did not flag this pair. Under the <i>override</i> policy the run continues as if it had, so the rest of the pipeline can be shown. Everything downstream carries this badge.</p>
      <p>${chip(p.badge || "override", "warn")} pair ${esc((p.pair || []).join(" × "))} · ${esc(p.type)}${p.provisional_relation ? ` · provisional ${esc(p.provisional_relation)} (A4 decides)` : ""}</p></div>`;
  }
  const ok = latest(s, "policy");
  return `<div class="banner ok"><b>${ok?.applied === false ? "Gold agrees: no conflict here" : "The detector found the conflict"}</b><p>No override needed.</p></div>`;
}

function scope(s: RunState) {
  const sk = skippedNote(s, "scope");
  if (sk) return sk;
  const descs = latest(s, "scope", "descriptors");
  const decs = allData(s, "scope").filter(d => d.relation);
  if (!descs && !decs.length) return empty("Waiting…");
  let html = "";
  if (descs) html += `<div class="descs">${descs.descriptors.map((d: any) => `<div class="desc">
      <span class="pid">${esc(d.passage)} · ${esc(d.source)}</span>
      <p><b>applies to</b> ${esc(d.applies_to)}${d.default ? " " + chip("default", "ok") : ""}</p>
      <p><b>outcome</b> ${esc(d.outcome)}</p>
      <p>${(d.attributes || []).map((a: any) => chip(`${a.name}${a.values ? " ∈ {" + a.values.join(", ") + "}" : ""}${a.min_value !== undefined ? " ≥ " + a.min_value : ""}${a.max_value !== undefined ? " ≤ " + a.max_value : ""}`)).join(" ") || '<span class="muted">no typed attributes: undecidable scopes fall back to entailment</span>'}</p></div>`).join("")}</div>`;
  for (const d of decs) {
    const disagree = d.gold_relation && d.gold_relation !== d.relation;
    html += `<div class="rel">
      <div class="rel-big ${esc(d.relation)}">${esc(d.relation)}</div>
      <dl class="kv"><dt>pair</dt><dd>${esc(d.pair.join(" × "))}</dd>
        <dt>applicability</dt><dd>${esc(human(d.set_relation))} <span class="muted">(exception relative to default)</span></dd>
        <dt>outcomes</dt><dd>${d.outcomes_agree ? "agree" : "disagree"}</dd>
        <dt>evidence</dt><dd>${esc(human(d.evidence))} · confidence ${fx(d.confidence, 2)}</dd>
        <dt>roles</dt><dd>default ${esc(d.default_passage)} · exception ${esc(d.exception_passage)}${d.roles_swapped ? ' <span class="muted">(input order swapped; decision unchanged)</span>' : ""}</dd>
        ${d.gold_relation ? `<dt>gold</dt><dd>${chip(d.gold_relation, disagree ? "bad" : "ok")}${disagree ? " A4 disagrees with gold" : " A4 agrees with gold"}</dd>` : ""}</dl>
      <p class="ins-note">${esc(d.rationale)}</p></div>`;
  }
  return html;
}

function extract(s: RunState) {
  const sk = skippedNote(s, "extract");
  if (sk) return sk;
  const p = latest(s, "extract", "probe");
  if (!p) return empty("Waiting…");
  const tau = p.tau_ground ?? 0.5;
  return `<p class="ins-note">Base rule: <code>${esc(p.base_passage)}</code> · ${p.admitted} admitted · ${p.rejected} rejected · ${p.api_calls} LLM call(s)</p>
    <h5>Contrastive questions</h5><ol class="qs">${p.questions.map((q: string) => `<li>${esc(q)}</li>`).join("")}</ol>
    <h5>Candidates and the grounding gate (τ = ${fx(tau, 2)})</h5>
    ${p.candidates.length ? p.candidates.map((c: any) => `<div class="cand ${c.admitted ? "in" : "out"}">
      <p>${c.admitted ? chip("admitted", "ok") : chip("rejected", "bad")} ${chip(c.explicitness)} <span class="muted">from ${esc(c.source_passage)}</span></p>
      <p><b>if</b> ${esc(c.condition)}</p><p><b>then</b> ${esc(c.outcome || "—")}</p>
      <div class="gate"><i style="width:${(c.entailment * 100).toFixed(1)}%"></i><b style="left:${(tau * 100).toFixed(1)}%"></b><span>entailment ${fx(c.entailment, 2)}</span></div>
      ${c.rejection_reason ? `<p class="muted">${esc(c.rejection_reason)}</p>` : ""}
      ${c.evidence_span ? `<p class="ev">“${esc(c.evidence_span)}”</p>` : ""}</div>`).join("") : empty("No candidate conditions proposed.")}`;
}

function branchCards(bs: any[], kept?: string[]) {
  return `<div class="bcards">${bs.map(b => `<div class="bcard ${b.default ? "def" : "exc"} ${kept && !kept.includes(b.id) ? "gone" : ""}">
    <span class="pid">${esc(b.id)} · ${b.default ? "rule" : "exception"} · ${esc(b.passage || "—")}</span>
    <p class="bc-if">${b.default ? "applies generally" : "if " + esc(b.condition || b.applies_to)}</p>
    <p class="bc-then">${esc(b.outcome)}</p>
    ${(b.attributes || []).length ? `<p>${b.attributes.map((a: any) => chip(`${a.name}${a.values ? " ∈ {" + a.values.join(", ") + "}" : ""}${a.min_value !== undefined ? " ≥ " + a.min_value : ""}${a.max_value !== undefined ? " ≤ " + a.max_value : ""}`)).join(" ")}</p>` : ""}</div>`).join("")}</div>`;
}

function branches(s: RunState) {
  const b = latest(s, "branches", "branches");
  if (!b) return empty("Waiting…");
  const vs = latest(s, "branches", "B1 against gold");
  return `<p>${chip(b.origin, b.origin.startsWith("oracle") ? "warn" : "ok")}</p>${branchCards(b.branches)}
    ${vs ? `<p class="ins-note">For comparison, B1 recovered <b>${vs.recovered} of ${vs.gold_exceptions}</b> gold exception(s) (the project's lexical matcher; a paraphrase it misses reads as not recovered).</p>` : ""}`;
}

function compose(s: RunState) {
  const sk = skippedNote(s, "compose");
  if (sk) return sk;
  const c = latest(s, "compose", "composition");
  const b = latest(s, "branches", "branches");
  if (!c) return empty("Waiting…");
  return `<div class="rel"><div class="rel-big res-${esc(c.resolution)}">${esc(c.resolution)}</div>
    <dl class="kv"><dt>why</dt><dd>${esc(c.reason)}</dd>
    ${c.route ? `<dt>route</dt><dd>${chip(c.route.action, "solid")} ${esc(c.route.reason)}</dd>` : ""}
    <dt>flags</dt><dd>${c.flags?.nested ? chip("nested", "bad") : ""}${c.flags?.crossed ? chip("crossed", "bad") : ""}${!c.flags?.nested && !c.flags?.crossed ? "none" : ""}</dd>
    ${c.selected_passage ? `<dt>kept</dt><dd>${esc(c.selected_passage)} (selection)</dd>` : ""}
    ${c.merged_pairs ? `<dt>merged</dt><dd>${c.merged_pairs} restatement(s)</dd>` : ""}</dl></div>
    ${b ? branchCards(b.branches, c.kept) : ""}`;
}

function generate(s: RunState) {
  const sk = skippedNote(s, "generate");
  if (sk) return sk;
  const a = latest(s, "generate", "answer");
  if (!a) return empty("Waiting…");
  return `<blockquote class="answer">${cite(a.text)}</blockquote>
    <p class="ins-note">${chip(a.method)} ${a.api_calls ? `${a.api_calls} LLM call(s)` : "no model call: the deterministic template cannot drop a branch or invent a figure"} · cites ${a.citations.map((c: string) => `<span class="cite">${esc(c)}</span>`).join(" ")}</p>`;
}

function verify(s: RunState) {
  const sk = skippedNote(s, "verify");
  if (sk) return sk;
  const f = latest(s, "verify", "faithfulness");
  if (!f) return empty("Waiting…");
  const row = (ok: boolean, label: string, detail = "") => `<li class="${ok ? "ok" : "bad"}"><b>${ok ? "✓" : "✗"}</b> ${esc(label)}${detail ? ` <span class="muted">${esc(detail)}</span>` : ""}</li>`;
  return `<ul class="checks">
    ${row(!f.missing_branches.length, "every kept branch is stated", f.missing_branches.join(", "))}
    ${row(!f.unsupported_numbers.length, "no figure that is not in a branch", f.unsupported_numbers.join(", "))}
    ${row(!f.adjudicated, "no source ranked over another")}
    ${row(f.grounded !== false, "every exception is grounded in a retrieved passage")}
    ${row(!f.missing_attribution.length, "every branch is attributed", f.missing_attribution.length ? `missing for ${f.missing_attribution.join(", ")} (reported, not fatal)` : "")}
  </ul><p class="verdict ${f.ok ? "good" : "bad"}"><b>Faithfulness ${f.ok ? "passes" : "fails"}.</b></p>`;
}

function arena(s: RunState) {
  const a = latest(s, "arena", "answers");
  if (!a) return empty("Waiting…");
  const verdicts: Record<string, boolean | null> = {};
  for (const v of latest(s, "reader", "verdicts")?.verdicts || []) verdicts[v.system] = v.right;
  return `<div class="arena">${a.systems.map((x: any) => `<div class="ar ${x.system}">
    <span class="pid">${esc(x.name)}</span><p>${cite(x.text)}</p>
    <p>${chip(x.method)} ${x.system in verdicts ? (verdicts[x.system] ? chip("right for this reader", "ok") : verdicts[x.system] === false ? chip("wrong for this reader", "bad") : "") : ""}</p></div>`).join("")}</div>`;
}

function reader(s: RunState) {
  const sk = skippedNote(s, "reader");
  const v = latest(s, "reader", "verdicts");
  if (!v) return sk || empty("Waiting…");
  return `<p>Reader: ${Object.entries(v.reader).map(([k, x]) => chip(`${k} = ${x}`)).join(" ")} → branch <b>${esc(v.branch || "none")}</b> applies.</p>
    <table class="data mini"><thead><tr><th>branch</th><th>holds for this reader</th></tr></thead><tbody>${
      Object.entries(v.membership).map(([k, m]) => `<tr><td>${esc(k)}</td><td>${m === null ? "undecidable (free-text scope)" : m ? "yes" : "no"}</td></tr>`).join("")}</tbody></table>
    <ul class="checks">${v.verdicts.map((x: any) => `<li class="${x.right ? "ok" : "bad"}"><b>${x.right ? "✓" : "✗"}</b> ${esc(x.system)} is ${x.right ? "right" : "wrong"} for this reader</li>`).join("")}</ul>`;
}

function score(s: RunState) {
  const sk = skippedNote(s, "score");
  if (sk) return sk;
  const p = latest(s, "score", "preservation");
  if (!p) return empty("Waiting…");
  return `<table class="data mini score-t"><thead><tr><th>system</th><th>PR</th><th>SR</th><th>kept · lost · wrong</th></tr></thead><tbody>${
    p.rows.map((r: any) => `<tr><td>${esc(r.name)}<small>${esc(r.instrument)}</small></td><td><b>${pct(r.PR)}</b></td><td>${pct(r.SR)}</td><td>${r.preserved} · ${r.suppressed} · ${r.distorted}</td></tr>`).join("")}</tbody></table>
    ${p.gold_answer ? `<p class="ins-note"><b>Gold answer:</b> ${cite(p.gold_answer)}</p>` : ""}
    <p class="ins-note">PR: share of the ${p.gold_branches} gold branches the answer preserves. SR: share it suppresses. Textual judging is a lower bound: a paraphrase beyond the matcher reads as missing.</p>`;
}

const RENDER: Record<string, (s: RunState) => string> = {
  setup, ingest, retrieve, detect, type, policy, scope, extract, branches, compose, generate, verify, arena, reader, score,
};

export function renderInspector(host: HTMLElement, s: RunState, id: string) {
  const st = s.stages[id];
  const head = `<header class="ins-h"><h4>${esc(STAGE_LABEL[id] || id)}</h4>
    <span>${st ? `${esc(st.status)}${st.ms !== null ? ` · ${st.ms < 1 ? "<1" : Math.round(st.ms)} ms` : ""}${st.llm ? ` · ${st.llm} LLM call(s)` : ""}` : "pending"}</span></header>`;
  host.innerHTML = head + `<div class="ins-b">${(RENDER[id] || (() => empty("")))(s)}</div>`;
}

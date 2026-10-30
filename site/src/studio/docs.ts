/*  Your documents, and the conflict audit.
 *
 *  A collection is a set of documents (pasted text, a URL, or uploaded .txt,
 *  .md, .pdf, .html) cut into passages. Ask questions over it in the console,
 *  or audit it: rank the passage pairs that look like a rule and an exception
 *  to it -- the ones a selection-based RAG system will quietly get wrong.
 */
import { esc, fx, human } from "../ui/dom";
import { LiveSource, api } from "./api";
import type { StudioCtx } from "./console";

export class DocsView {
  private current = "";

  constructor(private host: HTMLElement, private ctx: StudioCtx) {
    host.innerHTML = `<div class="dv">
      <aside class="dv-list">
        <h5>Collections</h5><ul data-k="colls"></ul>
        <form data-k="new" class="dv-new"><input class="inp" placeholder="New collection name" data-k="name" required>
          <button class="btn ghost sm">Create</button></form>
        <p class="muted" data-k="off"></p>
      </aside>
      <section class="dv-main" data-k="main"><p class="muted">Create a collection, then add documents to it.</p></section>
    </div>`;
    this.q("new").addEventListener("submit", async e => {
      e.preventDefault();
      const name = this.q<HTMLInputElement>("name").value.trim();
      if (!name) return;
      try {
        const c = await api.createCollection(name);
        this.q<HTMLInputElement>("name").value = "";
        this.current = c.id;
        await this.refresh();
      } catch (err) { this.say(`Could not create it: ${(err as Error).message}`); }
    });
    ctx.onHealth(() => this.refresh());
  }

  private q<T extends HTMLElement = HTMLElement>(k: string, root: ParentNode = this.host): T {
    return root.querySelector(`[data-k="${k}"]`) as T;
  }

  private say(msg: string) { this.q("off").textContent = msg; }

  show() { this.refresh(); }

  private async refresh() {
    if (!this.ctx.online) {
      this.q("colls").innerHTML = "";
      this.q("main").innerHTML = `<div class="banner warn"><b>Documents need the engine server</b>
        <p>Collections are stored, chunked and indexed by the local engine server. Start it with <code>start.bat studio</code>
        (or <code>python -m service</code>) and this page connects by itself.</p></div>`;
      return;
    }
    this.say("");
    const cs = await api.collections().catch(() => []);
    if (!this.current && cs.length) this.current = cs[0].id;
    this.q("colls").innerHTML = cs.map(c => `<li><button class="${c.id === this.current ? "on" : ""}" data-c="${esc(c.id)}">
      <b>${esc(c.name)}</b><span>${c.passages} passages</span></button></li>`).join("") || `<li class="muted">none yet</li>`;
    this.q("colls").querySelectorAll<HTMLButtonElement>("[data-c]").forEach(b => b.addEventListener("click", () => { this.current = b.dataset.c!; this.refresh(); }));
    if (this.current) await this.renderCollection(this.current);
  }

  private async renderCollection(id: string) {
    const main = this.q("main");
    let c: any;
    try { c = await api.collection(id); } catch { main.innerHTML = `<p class="muted">That collection is gone.</p>`; this.current = ""; return; }
    main.innerHTML = `<header class="dv-h"><h3>${esc(c.name)}</h3><span class="muted">${c.documents.length} document(s) · ${c.passages} passage(s)</span>
        <a class="btn ghost sm" href="#/studio" data-k="ask">Ask a question ▸</a>
        <button class="btn ghost sm danger" data-k="del">Delete</button></header>
      <div class="dv-add">
        <div class="drop" data-k="drop" tabindex="0"><b>Drop files here</b><span>.txt · .md · .pdf · .html, or click to choose</span>
          <input type="file" multiple accept=".txt,.md,.pdf,.html,.htm" data-k="file" hidden></div>
        <form data-k="paste" class="dv-paste"><input class="inp" placeholder="document name" data-k="dname" value="pasted text">
          <textarea class="inp" rows="4" placeholder="Paste a policy, terms and conditions, an FAQ…" data-k="dtext"></textarea>
          <div class="row"><select class="sel sm" data-k="dtype">${["", "official_policy", "government_guidance", "product_terms", "faq", "amendment", "third_party"].map(t => `<option value="${t}">${t || "guess the source type"}</option>`).join("")}</select>
          <button class="btn ghost sm">Add text</button></div></form>
        <form data-k="urlf" class="dv-url"><input class="inp" type="url" placeholder="https://www.gov.uk/…" data-k="url"><button class="btn ghost sm">Fetch URL</button></form>
      </div>
      <table class="data mini"><thead><tr><th>document</th><th>kind</th><th>passages</th><th>characters</th></tr></thead><tbody>${
        c.documents.map((d: any) => `<tr><td>${d.url ? `<a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.name)}</a>` : esc(d.name)}</td><td>${esc(d.kind)}</td><td>${d.passages}</td><td>${d.chars.toLocaleString()}</td></tr>`).join("") || `<tr><td colspan="4" class="muted">no documents yet</td></tr>`}</tbody></table>
      <details class="dv-pass"><summary>Passages</summary><div data-k="plist" data-lenis-prevent><p class="muted">loading…</p></div></details>
      <section class="dv-audit"><h4>Conflict audit</h4>
        <p class="muted">Finds passage pairs that state a rule and an exception to it. Stage 1 screens every candidate pair, then the strongest are analysed for their scope relation. A screening tool: it says “worth a look”, not “verified”.</p>
        <div class="row"><label class="field"><span>analyse top</span><input class="inp" type="number" min="1" max="100" value="12" data-k="top"></label>
          <label class="field"><span>mode</span><select class="sel" data-k="amode"><option value="offline">offline</option><option value="live" ${this.ctx.health?.live_ready ? "" : "disabled"}>live</option></select></label>
          <button class="btn run" data-k="audit" ${c.passages < 2 ? "disabled" : ""}>Run audit ▸</button></div>
        <div data-k="aprog" class="muted"></div><div data-k="findings"></div></section>`;

    const drop = this.q("drop", main), file = this.q<HTMLInputElement>("file", main);
    const upload = async (files: File[]) => {
      if (!files.length) return;
      drop.classList.add("busy");
      try { await api.upload(id, files); } catch (e) { alert(`Upload failed: ${(e as Error).message}`); }
      drop.classList.remove("busy");
      this.refresh();
    };
    drop.addEventListener("click", () => file.click());
    drop.addEventListener("keydown", e => { if (e.key === "Enter") file.click(); });
    file.addEventListener("change", () => upload([...(file.files || [])]));
    drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); upload([...(e.dataTransfer?.files || [])]); });

    this.q("paste", main).addEventListener("submit", async e => {
      e.preventDefault();
      const text = this.q<HTMLTextAreaElement>("dtext", main).value.trim();
      if (!text) return;
      try {
        await api.addText(id, { name: this.q<HTMLInputElement>("dname", main).value || "pasted text", text,
                                source_type: this.q<HTMLSelectElement>("dtype", main).value || undefined });
        this.refresh();
      } catch (err) { alert((err as Error).message); }
    });
    this.q("urlf", main).addEventListener("submit", async e => {
      e.preventDefault();
      const url = this.q<HTMLInputElement>("url", main).value.trim();
      if (!url) return;
      try { await api.addText(id, { url }); this.refresh(); } catch (err) { alert((err as Error).message); }
    });
    this.q("del", main).addEventListener("click", async () => {
      if (!confirm(`Delete “${c.name}” and its documents?`)) return;
      await api.deleteCollection(id); this.current = ""; this.refresh();
    });
    this.q("ask", main).addEventListener("click", () => sessionStorage.setItem("sc.ask", id));
    (main.querySelector(".dv-pass") as HTMLDetailsElement).addEventListener("toggle", async ev => {
      if (!(ev.target as HTMLDetailsElement).open) return;
      const ps = await api.passages(id).catch(() => []);
      this.q("plist", main).innerHTML = ps.map((p: any) => `<div class="psg2"><span class="pid">${esc(p.id)} · ${esc(human(p.source_type))}${p.date ? " · " + esc(p.date) : ""}</span><p>${esc(p.text)}</p></div>`).join("");
    });
    this.q("audit", main).addEventListener("click", () => this.audit(id, main));
  }

  private async audit(id: string, main: HTMLElement) {
    const prog = this.q("aprog", main), out = this.q("findings", main);
    out.innerHTML = ""; prog.textContent = "starting…";
    const findings: any[] = [];
    try {
      const src = await LiveSource.start({ analyse_top: Number(this.q<HTMLInputElement>("top", main).value) || 12,
                                           mode: this.q<HTMLSelectElement>("amode", main).value }, "audit", id);
      src.attach({
        event: ev => {
          if (ev.stage === "candidates" && ev.title === "prefilter") prog.textContent = `${ev.payload.candidates} of ${ev.payload.pairs_total} pairs share enough vocabulary to compare`;
          if (ev.stage === "screen" && ev.title === "progress") prog.textContent = `stage 1: ${ev.payload.done} / ${ev.payload.total} candidate pairs screened`;
          if (ev.title === "finding") { findings.push(ev.payload); out.innerHTML = this.findingsHtml(findings); }
          if (ev.stage === "run" && ev.kind === "end") {
            const s = ev.payload.summary || {};
            prog.innerHTML = `<b>${s.high ?? 0}</b> high-risk pair(s) among ${s.analysed ?? 0} analysed · ${s.candidates ?? 0} candidates from ${s.passages ?? 0} passages`;
          }
          if (ev.stage === "run" && ev.kind === "error") prog.textContent = `failed: ${ev.title}`;
        },
        done: () => {},
      });
    } catch (e) { prog.textContent = `The server refused: ${(e as Error).message}`; }
  }

  private findingsHtml(fs: any[]) {
    return fs.map((f, i) => `<article class="finding2 risk-${esc(f.risk)}">
      <header><b>${i + 1}</b> <span class="chip ${f.risk === "high" ? "bad" : f.risk === "medium" ? "warn" : ""}">${esc(f.risk)} risk</span>
        ${f.relation ? `<span class="chip">${esc(f.relation)}</span>` : ""}
        <span class="muted">stage-1 ${fx(f.score, 2)} · overlap ${fx(f.overlap, 2)} · exception cues ${fx(f.exception_cues, 1)} · ${esc(human(f.evidence || ""))}</span></header>
      <div class="pair">${["a", "b"].map(k => `<div class="${f.default && f[k].id === f.default ? "rule" : f.exception && f[k].id === f.exception ? "exc" : ""}">
          <span class="pid">${esc(f[k].id)}${f.default && f[k].id === f.default ? " · rule" : f.exception && f[k].id === f.exception ? " · exception" : ""}</span><p>${esc(f[k].text)}</p></div>`).join("")}</div>
      ${f.rationale ? `<p class="muted">${esc(f.rationale)}</p>` : ""}</article>`).join("");
  }
}

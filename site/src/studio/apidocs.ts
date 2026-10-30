/*  The API, in the site's own words: what each endpoint does, an example to
 *  copy in three languages, and a panel that calls the running server. */
import { esc } from "../ui/dom";
import { serverBase } from "./api";
import type { StudioCtx } from "./console";

const ENDPOINTS: [string, string, string][] = [
  ["POST", "/v1/resolve", "Answer a question in one call: the scoped answer, its branches, citations and checks."],
  ["POST", "/v1/runs", "Start a run; returns run_id and an events_url to stream."],
  ["GET", "/v1/runs/{id}/events", "Server-sent events: every stage as it happens (stored events replay first)."],
  ["GET", "/v1/runs/{id}", "One run: request, status, summary."],
  ["GET", "/v1/runs/{id}/events.json", "Every event of a run, as JSON (replay, export)."],
  ["GET", "/v1/runs", "Run history (search, filter by kind)."],
  ["POST", "/v1/collections", "Create a document collection."],
  ["POST", "/v1/collections/{id}/documents", "Add a document from {text} or {url}."],
  ["POST", "/v1/collections/{id}/upload", "Upload .txt, .md, .pdf or .html files."],
  ["POST", "/v1/collections/{id}/audit", "Scan a collection for hidden conditional conflicts (streamed)."],
  ["POST", "/v1/evals", "Evaluate on a benchmark split (streamed)."],
  ["GET", "/v1/corpus", "Benchmark records (with gold)."],
  ["GET", "/v1/costs", "LLM spend and latency by pipeline step and model."],
  ["GET", "/v1/health", "Status, and what live mode needs."],
];

export class ApiView {
  constructor(private host: HTMLElement, private ctx: StudioCtx) {
    this.render();
    ctx.onHealth(() => this.render());
  }

  show() { this.render(); }

  private render() {
    const base = serverBase();
    const body = `{
  "query": "Is there a fee on international card payments?",
  "documents": [
    "A 3% fee applies to all international card payments.",
    "Premium cardholders pay no fee on international card payments."
  ],
  "detector_policy": "override",
  "reader": {}
}`;
    const curl = `curl -X POST ${base}/v1/resolve \\\n  -H "Content-Type: application/json" \\\n  -d '${body.replace(/\n\s*/g, " ")}'`;
    const py = `from service.client import ScopeCompose

sc = ScopeCompose("${base}")
r = sc.resolve(
    "Is there a fee on international card payments?",
    documents=["A 3% fee applies to all international card payments.",
               "Premium cardholders pay no fee on international card payments."],
)
print(r["answer"])          # the scoped answer, with (per pX) citations
print(r["relation"], r["resolution"], r["faithful"])

# or watch a run stage by stage
for event in sc.stream(sc.start(source="benchmark", record_id="wp1_fin_imp_003")):
    print(event["stage"], event["kind"], event["title"])`;
    const js = `// start a run, then stream every stage as it happens
const r = await fetch("${base}/v1/runs", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ source: "benchmark", record_id: "wp1_fin_imp_003", mode: "offline" }),
}).then(r => r.json());

const es = new EventSource("${base}" + r.events_url);
es.onmessage = m => {
  const ev = JSON.parse(m.data);          // {run_id, seq, t_ms, stage, kind, title, payload}
  console.log(ev.stage, ev.kind, ev.title);
  if (ev.stage === "run" && ev.kind === "end") es.close();
};`;
    this.host.innerHTML = `<div class="api">
      <div class="api-intro"><p>The studio is a client of this API, and so can anything else be. Every run streams its stages as
        <a href="https://developer.mozilla.org/docs/Web/API/Server-sent_events" target="_blank" rel="noopener">server-sent events</a>,
        each <code>{run_id, seq, t_ms, stage, kind, title, payload}</code>, and every run is stored for replay.</p>
        <p>${this.ctx.online ? `<span class="chip ok">connected</span> <a href="${esc(base)}/docs" target="_blank" rel="noopener">Interactive docs (Swagger) ↗</a> · <a href="${esc(base)}/redoc" target="_blank" rel="noopener">ReDoc ↗</a>`
          : `<span class="chip warn">no server</span> start it with <code>start.bat studio</code> or <code>python -m service</code>`}</p></div>
      <table class="data mini api-t"><thead><tr><th>method</th><th>path</th><th>what it does</th></tr></thead><tbody>${
        ENDPOINTS.map(([m, p, d]) => `<tr><td><span class="verb ${m.toLowerCase()}">${m}</span></td><td><code>${esc(p)}</code></td><td>${esc(d)}</td></tr>`).join("")}</tbody></table>
      <div class="api-code">
        ${[["curl", curl], ["Python SDK", py], ["JavaScript", js]].map(([t, c]) => `<figure><figcaption>${t}<button class="btn ghost sm" data-copy>copy</button></figcaption><pre><code>${esc(c)}</code></pre></figure>`).join("")}
      </div>
      <section class="api-try"><h5>Try it: POST /v1/resolve</h5>
        <textarea class="inp code" rows="12" data-k="body" spellcheck="false">${esc(body)}</textarea>
        <button class="btn run" data-k="send" ${this.ctx.online ? "" : "disabled"}>Send ▸</button>
        <pre class="json" data-k="out" data-lenis-prevent>${this.ctx.online ? "" : "connect the engine server to try it"}</pre></section>
    </div>`;
    this.host.querySelectorAll<HTMLButtonElement>("[data-copy]").forEach(b => b.addEventListener("click", () => {
      const code = b.closest("figure")!.querySelector("code")!.textContent || "";
      navigator.clipboard?.writeText(code).then(() => { b.textContent = "copied"; setTimeout(() => (b.textContent = "copy"), 1200); });
    }));
    this.host.querySelector("[data-k=send]")?.addEventListener("click", async () => {
      const out = this.host.querySelector("[data-k=out]") as HTMLElement;
      out.textContent = "running…";
      try {
        const payload = JSON.parse((this.host.querySelector("[data-k=body]") as HTMLTextAreaElement).value);
        const t = performance.now();
        const r = await fetch(serverBase() + "/v1/resolve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const j = await r.json();
        out.textContent = `HTTP ${r.status} · ${Math.round(performance.now() - t)} ms\n\n` + JSON.stringify(j, null, 2);
      } catch (e) { out.textContent = String((e as Error).message); }
    });
  }
}

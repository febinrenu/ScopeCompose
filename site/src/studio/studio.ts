/*  ScopeCompose Studio: the pipeline as a working product.
 *
 *    #/studio              the live console
 *    #/studio/run/<id>     a stored run, opened in the console
 *    #/studio/eval         batch evaluation, streamed
 *    #/studio/docs         document collections and the conflict audit
 *    #/studio/runs[/<id>]  history: replay, compare, export
 *    #/studio/api          the API, with examples to copy
 */
import "./studio.css";
import { esc } from "../ui/dom";
import { api, serverBase, setServerBase, type Health } from "./api";
import { Console, type StudioCtx } from "./console";

type Tab = "console" | "eval" | "docs" | "runs" | "api";

const TABS: [Tab, string, string][] = [
  ["console", "Console", "#/studio"],
  ["eval", "Evaluate", "#/studio/eval"],
  ["docs", "Documents & audit", "#/studio/docs"],
  ["runs", "History", "#/studio/runs"],
  ["api", "API", "#/studio/api"],
];

export function mountStudio(host: HTMLElement) {
  host.classList.add("studio-view");
  host.innerHTML = `<div class="studio">
    <header class="st-h">
      <a class="back" href="#act-00">← Back to the story</a>
      <div class="st-title">
        <h2>ScopeCompose <em>Studio</em></h2>
        <p>The pipeline, running: every stage streams in as it computes. Real retrieval, real detection,
          real composition, every model call accounted for.</p>
      </div>
      <nav class="st-tabs" aria-label="Studio">${TABS.map(([t, label, href]) => `<a href="${href}" data-tab="${t}">${label}</a>`).join("")}</nav>
      <form class="st-server" data-k="srvform">
        <label><span class="mono">engine server</span><input class="inp" data-k="srvurl" value="${esc(serverBase())}" spellcheck="false"></label>
        <button class="btn ghost sm">connect</button>
      </form>
    </header>
    ${TABS.map(([t]) => `<div class="st-view" data-view="${t}" hidden></div>`).join("")}
  </div>`;

  const listeners: (() => void)[] = [];
  const ctx: StudioCtx = { health: null, online: false, recorded: [], onHealth: fn => listeners.push(fn) };

  const probe = async () => {
    let h: Health | null = null;
    try { h = await api.health(); } catch { h = null; }
    const was = ctx.online;
    ctx.health = h;
    ctx.online = !!h?.ok;
    host.querySelector(".st-server")!.setAttribute("data-on", String(ctx.online));
    if (was !== ctx.online || !listeners.length) listeners.forEach(fn => fn());
  };

  host.querySelector("[data-k=srvform]")!.addEventListener("submit", e => {
    e.preventDefault();
    const v = (host.querySelector("[data-k=srvurl]") as HTMLInputElement).value.trim();
    setServerBase(v || null);
    probe();
  });

  const views: Partial<Record<Tab, { show?: (arg?: string) => void }>> = {};
  let con: Console | null = null;

  const view = async (t: Tab, arg?: string) => {
    host.querySelectorAll<HTMLElement>(".st-view").forEach(v => (v.hidden = v.dataset.view !== t));
    host.querySelectorAll<HTMLElement>(".st-tabs a").forEach(a => a.classList.toggle("on", a.dataset.tab === t));
    const el = host.querySelector<HTMLElement>(`.st-view[data-view="${t}"]`)!;
    if (t === "console") {
      con ??= new Console(el, ctx);
      if (arg) con.openRun(arg, false);
      const ask = sessionStorage.getItem("sc.ask");   // "Ask a question" from a collection
      if (ask) { sessionStorage.removeItem("sc.ask"); con.useCollection(ask); }
      return;
    }
    if (!views[t]) {
      if (t === "eval") views[t] = new (await import("./eval")).EvalView(el, ctx);
      else if (t === "docs") views[t] = new (await import("./docs")).DocsView(el, ctx);
      else if (t === "runs") views[t] = new (await import("./runs")).RunsView(el, ctx);
      else views[t] = new (await import("./apidocs")).ApiView(el, ctx);
    }
    views[t]!.show?.(arg);
  };

  fetch("./data/runs/index.json").then(r => (r.ok ? r.json() : [])).then(list => {
    ctx.recorded = Array.isArray(list) ? list : [];
    (con as unknown as { refreshRecorded?: () => void } | null)?.refreshRecorded?.();
  }).catch(() => {});

  probe();
  setInterval(() => { if (!host.hidden) probe(); }, 8000);

  return {
    route(hash: string) {
      const parts = hash.replace(/^#\/studio\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
      const [head, arg] = parts;
      if (!head) view("console");
      else if (head === "run") view("console", arg);
      else if (head === "eval" || head === "docs" || head === "runs" || head === "api") view(head, arg);
      else view("console");
    },
  };
}

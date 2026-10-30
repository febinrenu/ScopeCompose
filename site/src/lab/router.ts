/*  Hash routing. `#/bench`, `#/byo` and `#/architecture` are lab views,
 *  rendered once, lazily. Everything else is the story; anchors from the
 *  old single-page site (`#relations`, `#claims`, ...) still land on the
 *  act that now holds that content.
 */
import { $, $$ } from "../ui/dom";

const VIEWS: Record<string, string> = {
  "#/bench": "view-bench",
  "#/byo": "view-byo",
  "#/architecture": "view-architecture",
  "#/studio": "view-studio",
};

/** `#/studio` and its sub-routes (`#/studio/eval`, `#/studio/runs/…`) share one view. */
function viewFor(hash: string): string | undefined {
  if (hash === "#/studio" || hash.startsWith("#/studio/")) return VIEWS["#/studio"];
  return VIEWS[hash];
}

export const LEGACY: Record<string, string> = {
  "#hero": "#act-06", "#relations": "#act-04", "#position": "#act-03", "#pipeline": "#act-05",
  "#corpus": "#explorer", "#findings": "#findings", "#claims": "#act-09", "#repro": "#colophon",
};

/** Only plain ids are ever looked up; anything else (a stray or hostile hash) is ignored. */
export function anchorTarget(hash: string): HTMLElement | null {
  const h = LEGACY[hash] || hash;
  return /^#[A-Za-z][\w-]*$/.test(h) ? $(h) : null;
}

export interface RouterHooks {
  enterLab: () => void;
  leaveLab: () => void;
  anchor: (target: HTMLElement) => void;
}

export function startRouter(h: RouterHooks) {
  let inLab = false;
  let storyY = 0;
  const built = new Set<string>();
  let studio: { route: (hash: string) => void } | null = null;

  const render = async (id: string, host: HTMLElement) => {
    if (built.has(id)) return;
    built.add(id);
    if (id === "view-bench") (await import("./bench")).renderBench(host);
    else if (id === "view-byo") (await import("./byo")).renderByo(host);
    else if (id === "view-studio") studio = (await import("../studio/studio")).mountStudio(host);
    else (await import("./architecture")).renderArchitecture(host);
  };

  const sync = async () => {
    const hash = location.hash;
    const view = viewFor(hash);
    $$(".bar-lab a, .menu-foot a.route").forEach(a => {
      const href = a.getAttribute("href") || "";
      a.classList.toggle("on", href === hash || (href === "#/studio" && view === "view-studio"));
    });
    if (view) {
      const sameStudio = inLab && view === "view-studio" && studio !== null && !document.getElementById(view)!.hidden;
      if (!inLab) { storyY = scrollY; h.enterLab(); }
      inLab = true;
      document.documentElement.classList.add("lab-on");
      document.documentElement.classList.toggle("studio-on", view === "view-studio");
      for (const id of Object.values(VIEWS)) { const el = document.getElementById(id); if (el) el.hidden = id !== view; }
      const host = document.getElementById(view)!;
      await render(view, host);
      if (view === "view-studio") {
        studio?.route(hash);
        if (sameStudio) return;          // switching tabs inside the studio keeps scroll and focus
      }
      scrollTo(0, 0);
      (host.querySelector("h2") as HTMLElement | null)?.setAttribute("tabindex", "-1");
      (host.querySelector("h2") as HTMLElement | null)?.focus({ preventScroll: true });
      return;
    }
    const wasLab = inLab;
    if (inLab) {
      inLab = false;
      document.documentElement.classList.remove("lab-on", "studio-on");
      for (const id of Object.values(VIEWS)) { const el = document.getElementById(id); if (el) el.hidden = true; }
      h.leaveLab();
    }
    const target = hash && hash.length > 1 ? anchorTarget(hash) : null;
    if (target) requestAnimationFrame(() => h.anchor(target));
    else if (wasLab) requestAnimationFrame(() => scrollTo(0, storyY));
  };

  // In-page anchors scroll smoothly through the story's scroller.
  document.addEventListener("click", e => {
    const a = (e.target as HTMLElement).closest("a");
    if (!a) return;
    const href = a.getAttribute("href") || "";
    if (!href.startsWith("#") || href.startsWith("#/") || href === "#") return;
    const target = anchorTarget(href);
    if (!target) return;
    e.preventDefault();
    if (location.hash !== href) history.pushState(null, "", href);
    if (inLab) sync(); else h.anchor(target);
  });
  addEventListener("hashchange", sync);
  addEventListener("popstate", sync);
  sync();
}

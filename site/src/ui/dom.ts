export const $ = <T extends Element = HTMLElement>(s: string, r: ParentNode = document) => r.querySelector(s) as T | null;
export const $$ = <T extends Element = HTMLElement>(s: string, r: ParentNode = document) => Array.from(r.querySelectorAll(s)) as T[];

const ESC: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (s: unknown) => String(s ?? "").replace(/[&<>"']/g, c => ESC[c]);

export const pct = (x: number, d = 1) => (x * 100).toFixed(d) + "%";
export const fx = (x: number, d = 3) => Number.isFinite(x) ? x.toFixed(d) : "—";
export const human = (s: string | null | undefined) => String(s ?? "—").replace(/_/g, " ");

/** Wrap each line of a heading in a mask so it can rise into view. */
export function splitLines(el: HTMLElement) {
  if (el.dataset.split) return;
  el.dataset.split = "1";
  const parts = el.innerHTML.split(/<br\s*\/?>/i);
  el.innerHTML = parts.map(p => `<span class="split-line"><span>${p.trim()}</span></span>`).join("");
}

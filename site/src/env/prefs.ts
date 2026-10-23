/*  Preferences that decide how the page runs. Read once, at boot; reduced
 *  motion is also followed live.
 */
const q = new URLSearchParams(location.search);

export type QualityOverride = "off" | "low" | "mid" | "high" | null;

export const prefs = {
  /** ?q=off|low|mid|high forces a quality tier (demos and tests). */
  quality: (["off", "low", "mid", "high"].includes(q.get("q") || "") ? q.get("q") : null) as QualityOverride,
  debug: q.has("debug"),
  /** Strict determinism: frozen time, no pointer -- the reversibility check uses this. */
  still: q.has("still"),
  saveData: Boolean((navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData),
  coarse: matchMedia("(pointer: coarse)").matches,
};

const rmQuery = matchMedia("(prefers-reduced-motion: reduce)");

/** The inline head script already set html.rm before first paint. */
export const reducedMotion = (): boolean => document.documentElement.classList.contains("rm");

/** Follow a live change of the OS setting, unless the URL forced a mode. */
export function watchReducedMotion(onChange: (rm: boolean) => void): void {
  if (q.get("motion")) return;
  rmQuery.addEventListener("change", e => {
    document.documentElement.classList.toggle("rm", e.matches);
    onChange(e.matches);
  });
}

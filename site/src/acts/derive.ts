/*  Values derived from WorldParams rather than stored in keyframes.
 *  Pure, so the world stays a function of the story cursor.
 */
import type { WorldParams } from "./keyframes";
import { STATION_IDS } from "./keyframes";

const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const bump = (x: number, c: number, w: number) => Math.max(0, 1 - Math.abs(x - c) / w);

/** The instant of impact: peaks as the shatter begins, gone once it has. */
export const flash = (p: WorldParams) => bump(p.burn, 0.06, 0.06) * p.duel * (1 - p.compose);

/** Camera shake on impact, a deterministic function of burn (not of time). */
export function shake(p: WorldParams): [number, number] {
  const k = flash(p) * 0.06;
  return [Math.sin(p.burn * 211.3) * k, Math.cos(p.burn * 173.9) * k * 0.7];
}

/** How lit station i is, 0..1, from the journey's progress. */
export function stationActivation(flow: number, i: number): number {
  const at = flow * (STATION_IDS.length - 1);
  return clamp(1 - Math.abs(at - i) * 1.25);
}

/** Index of the station the journey is at. */
export const stationIndex = (flow: number) => Math.round(clamp(flow) * (STATION_IDS.length - 1));

/** Where the retrieved cards settle (the middle card), in world space. */
export const CARD_FOCUS: [number, number, number] = [0, -0.72, 2.4];

/** Focus distance: always the act's subject, so depth of field never blurs what is meant to be read.
 *  While the retrieved cards are up, focus moves from the target onto them. */
export function focusDistance(p: WorldParams): number {
  const toTarget = Math.hypot(p.camX - p.tgtX, p.camY - p.tgtY, p.camZ - p.tgtZ);
  const toCards = Math.hypot(p.camX - CARD_FOCUS[0], p.camY - CARD_FOCUS[1], p.camZ - CARD_FOCUS[2]);
  const k = clamp(p.cards * 1.5);
  return toTarget + (toCards - toTarget) * k;
}

/** Chromatic aberration: none at rest (even a little splits type into fringes), a spike on impact. */
export const aberration = (p: WorldParams) => flash(p) * 1.6 + p.crack * (1 - p.burn) * 0.08;

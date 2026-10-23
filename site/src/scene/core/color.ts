/*  The palette, in linear light.
 *
 *  The glow rule: only things meant to glow go above 1.0. Text and glass
 *  faces stay at or below 1.0; rims, beams, embers and sparks sit at 2–6, so
 *  bloom (threshold 1.0) finds exactly them and never the type.
 */
import { Color } from "three";

export const HEX = {
  ink: "#050608",
  ivory: "#f4f2ec",
  cyan: "#7ff0ff",
  amber: "#ffb45c",
  coral: "#ff6a5c",
  violet: "#a98bff",
  mint: "#7dffc4",
} as const;

/** A colour in linear light, optionally pushed into HDR for bloom. */
export const lin = (hex: string, intensity = 1) => new Color(hex).multiplyScalar(intensity);

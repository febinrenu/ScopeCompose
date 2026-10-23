/*  Glass, by quality tier.
 *  Tiers 2–3: real transmission (the scene refracts through it), iridescence
 *  and clearcoat. Tier 1: "fake glass" -- reflections and a fresnel edge, no
 *  transmission pass -- which costs almost nothing.
 */
import { Color, MeshPhysicalMaterial, type Texture } from "three";

export type GlassKind = "clear" | "amber" | "cyan" | "prism";

export function makeGlass(kind: GlassKind, env: Texture, transmission: boolean): MeshPhysicalMaterial {
  const tint = kind === "amber" ? "#ffcf95" : kind === "cyan" ? "#c9fbff" : "#ffffff";
  const atten = kind === "amber" ? "#ff9a3c" : kind === "cyan" ? "#3fd8f0" : "#e8f4ff";
  const m = new MeshPhysicalMaterial({
    color: new Color(tint),
    envMap: env,
    envMapIntensity: kind === "amber" ? 1.4 : 1.25,
    roughness: kind === "prism" ? 0.0 : 0.035,
    metalness: 0,
    clearcoat: 1,
    clearcoatRoughness: 0.04,
    iridescence: kind === "clear" ? 0.55 : 0.25,
    iridescenceIOR: 1.32,
    iridescenceThicknessRange: [120, 480],
    specularIntensity: 1,
  });
  if (transmission) {
    m.transmission = 1;
    m.thickness = kind === "amber" ? 0.9 : 1.4;
    m.ior = kind === "prism" ? 1.62 : 1.48;
    m.attenuationColor = new Color(atten);
    m.attenuationDistance = kind === "amber" ? 0.9 : 3.2;
    if (kind === "prism") m.dispersion = 6;
    if (kind === "amber") { m.emissive = new Color("#ff8a2a"); m.emissiveIntensity = 0.12; }
  } else {
    m.transparent = true;
    m.opacity = kind === "amber" ? 0.55 : 0.2;
    m.depthWrite = false;
    if (kind === "amber") { m.emissive = new Color("#ff8a2a"); m.emissiveIntensity = 0.35; }
  }
  return m;
}

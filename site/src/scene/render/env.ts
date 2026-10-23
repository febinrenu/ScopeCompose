/*  A dark studio, baked into a reflection map.
 *
 *  Glass is only as good as what it reflects. This is an ink room with
 *  three emissive panels -- a cyan key, an amber rim, a cool softbox above --
 *  so every glossy edge in the world carries the story's colours.
 */
import {
  BackSide, BoxGeometry, Mesh, MeshBasicMaterial, PMREMGenerator, PlaneGeometry, Scene, type Texture, type WebGLRenderer,
} from "three";
import { HEX, lin } from "../core/color";

export function buildEnvironment(renderer: WebGLRenderer): { texture: Texture; dispose: () => void } {
  const scene = new Scene();
  const room = new Mesh(new BoxGeometry(20, 12, 20), new MeshBasicMaterial({ color: lin("#0a0d14"), side: BackSide }));
  scene.add(room);
  const panel = (w: number, h: number, color: string, k: number, x: number, y: number, z: number, ry = 0, rx = 0) => {
    const m = new Mesh(new PlaneGeometry(w, h), new MeshBasicMaterial({ color: lin(color, k) }));
    m.position.set(x, y, z); m.rotation.set(rx, ry, 0);
    scene.add(m);
  };
  panel(4, 6, HEX.cyan, 6, -9.5, 1, 1, Math.PI / 2);          // cyan key, left
  panel(3, 7, HEX.amber, 5, 9.5, 0, -2, -Math.PI / 2);         // amber rim, right
  panel(9, 3, "#dfe8ff", 3.5, 0, 5.8, 0, 0, Math.PI / 2);      // cool softbox above
  panel(6, 1.2, HEX.coral, 1.6, 0, -2, -9.8);                  // faint coral kicker behind
  const pmrem = new PMREMGenerator(renderer);
  const rt = pmrem.fromScene(scene, 0.035);
  pmrem.dispose();
  scene.traverse(o => {
    const m = o as Mesh;
    m.geometry?.dispose();
    (m.material as MeshBasicMaterial | undefined)?.dispose?.();
  });
  return { texture: rt.texture, dispose: () => rt.dispose() };
}

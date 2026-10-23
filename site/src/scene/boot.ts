/*  Entry point of the 3D chunk, loaded with import() once the quality tier
 *  is known to be above zero. Post-processing is its own chunk, fetched in
 *  parallel only where it will be used.
 */
import { World, type WorldInput } from "./world";
import { buildTargetsAsync, netNodes } from "./targets";
import { wordPoints } from "./textures";
import { WORD_WIDTH } from "./actors/particles";

export type BootInput = Omit<WorldInput, "targets" | "nodes" | "makePost"> & { nodeCount: number };

export async function bootWorld(input: BootInput): Promise<World> {
  const postChunk = input.quality.post ? import("./render/post") : Promise.resolve(null);
  // The textures and the wordmark are drawn with the page's own fonts.
  try {
    await Promise.race([
      Promise.all([
        document.fonts.load("800 120px 'Hubot Sans Variable'"),
        document.fonts.load("500 92px 'Mona Sans Variable'"),
        document.fonts.load("italic 400 120px 'Newsreader Variable'"),
        document.fonts.load("600 40px 'Martian Mono Variable'"),
      ]),
      new Promise(r => setTimeout(r, 1800)),
    ]);
  } catch { /* fonts are a nicety here, not a requirement */ }
  const nodes = netNodes(input.nodeCount);
  const n = input.quality.particles;
  const targets = await buildTargetsAsync(n, wordPoints("SCOPECOMPOSE", WORD_WIDTH, Math.min(n, 65536)), nodes, 3, yieldToMain);
  const post = await postChunk;
  const world = new World({
    ...input, targets, nodes,
    makePost: post ? (r, s, c, o) => new post.Post(r, s, c, o) : null,
  });
  await world.warm();
  return world;
}

/** Hand the main thread back (scheduler.yield where it exists). */
function yieldToMain(): Promise<void> {
  const s = (globalThis as unknown as { scheduler?: { yield?: () => Promise<void> } }).scheduler;
  return s?.yield ? s.yield() : new Promise(r => setTimeout(r, 0));
}

export type { World };

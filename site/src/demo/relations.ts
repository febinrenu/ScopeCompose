/*  Act 04: the four relations, explorable.
 *
 *  Each tab runs its example through the real chain and shows what the
 *  engine decided -- including if it ever disagreed with the relation the
 *  example was written to show. Selecting a tab scrolls the story to that
 *  relation's keyframe, so the geometry behind the panel is the same thing
 *  the panel describes.
 */
import { RELATION_EXAMPLES } from "../data/examples";
import { runChain } from "../engine/pipeline";
import type { Relation } from "../engine/types";
import { $, $$, esc, human } from "../ui/dom";
import { countBy } from "../data/corpus";

/** Verbatim from contract/models.py:55-66 (the ScopeRelation docstring). */
const DEFINITIONS: Record<Relation, string> = {
  refinement: "One applicability set is a proper subset of the other, and the branches are outcome-INCOMPATIBLE on the subset. A genuine exception. Compose.",
  disjoint: "Applicability sets do not intersect. Outcome comparison is vacuous. Compose trivially.",
  redundant: "Applicability sets overlap or nest, but the branches AGREE on outcome throughout the overlap. Not a conflict at all. Merge into one branch.",
  opposed: "Applicability sets overlap, neither contains the other, and outcomes disagree on the overlap. A real contradiction. Fall back to selection.",
};

const CELL: Record<string, string> = {
  disjoint: "disjoint", subset: "nested", superset: "nested", equal: "overlap", overlapping: "overlap", unknown: "overlap",
};

export const RELATIONS: Relation[] = ["refinement", "disjoint", "redundant", "opposed"];

export class RelationExplorer {
  private current: Relation = "refinement";
  private tabs: HTMLButtonElement[];
  private panel: HTMLElement;

  constructor(private onSelect: (rel: Relation) => void) {
    this.tabs = $$<HTMLButtonElement>("#geo [role=tab]");
    this.panel = $("#panel-geo")!;
    this.tabs.forEach((t, i) => {
      t.addEventListener("click", () => { this.show(t.dataset.rel as Relation); this.onSelect(this.current); });
      t.addEventListener("keydown", e => {
        let k = -1;
        if (e.key === "ArrowRight") k = (i + 1) % this.tabs.length;
        else if (e.key === "ArrowLeft") k = (i + this.tabs.length - 1) % this.tabs.length;
        else if (e.key === "Home") k = 0;
        else if (e.key === "End") k = this.tabs.length - 1;
        if (k < 0) return;
        e.preventDefault();
        this.tabs[k].focus();
        this.show(this.tabs[k].dataset.rel as Relation);
        this.onSelect(this.current);
      });
    });
    this.show("refinement", true);
  }

  /** Called by the story as the cursor passes each relation. */
  follow(rel: Relation) { if (rel !== this.current) this.show(rel); }

  show(rel: Relation, force = false) {
    if (rel === this.current && !force) return;
    this.current = rel;
    this.tabs.forEach(t => {
      const on = t.dataset.rel === rel;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
    });
    this.panel.setAttribute("aria-labelledby", "tab-" + rel);
    const ex = RELATION_EXAMPLES[rel];
    const r = runChain(ex, ex.b);
    const counts = countBy("rel");
    const tone = rel === "opposed" ? "bad" : rel === "refinement" ? "warn" : "ok";
    this.panel.innerHTML = `
      <p class="def">${esc(DEFINITIONS[rel])}</p>
      <p class="def-src mono">contract/models.py:55-66 · ${counts[rel] || 0} instance${counts[rel] === 1 ? "" : "s"} in the benchmark</p>
      <div class="geo-ex">
        ${ex.b.map(b => `<div class="row"><span class="mono">${b.def ? "rule" : esc(b.at?.[0]?.n ? human(b.at[0].n) : "scope")}</span>
          <span>${b.def ? "In all cases" : "For " + esc(b.desc)}, ${esc(b.out)}.</span></div>`).join("")}
        <div class="geo-result">
          <span class="chip">set: ${esc(r.setRel)}</span>
          <span class="chip">outcomes ${r.agree ? "agree" : "disagree"}</span>
          <span class="chip ${tone}">${esc(r.relation)}</span>
          <span class="chip solid">→ ${esc(r.action)}</span>
          <span class="mono muted">${esc(ex.provenance)} · engine output</span>
        </div>
        ${r.relation !== ex.intended ? `<p class="state err">The engine returned ${esc(r.relation)} for an example written to show ${esc(ex.intended || "")}.</p>` : ""}
      </div>`;
    document.documentElement.style.setProperty("--tabc", rel === "opposed" ? "var(--coral)" : rel === "refinement" ? "var(--amber)" : "var(--cyan)");
    $$("[data-cell]").forEach(c => c.classList.remove("hit"));
    const row = CELL[r.setRel] || "overlap";
    $(`[data-cell="${row}-${r.agree ? "a" : "d"}"]`)?.classList.add("hit");
  }
}

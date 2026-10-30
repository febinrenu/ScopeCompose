"""The generated half of the datasheet: every number comes from the data, none is typed.

    python -m benchmark.datasheet_v7 --data benchmark/data/corpus.jsonl --out docs/datasheet_generated.md

``benchmark/datasheet.md`` is the narrative (motivation, uses, limitations). This module writes the parts
that drift when typed by hand:

* the **strata x split** table, and whether the S1 narrowing rule applies;
* the **tier** table (natural / split / same-guide) and the conditional count;
* **label provenance**: how many gold branch structures are still model proposals, how many a person has
  reviewed, and how many have none. A model proposal is never counted as human-annotated;
* **agreement and hours** per review task, from the review files: decisions per annotator, hours recorded,
  and Cohen's kappa between each pair of annotators (undefined batches are labelled, not scored);
* the **release** statement (passages withheld, by licence), when a release manifest is given.

A section with nothing behind it says so ("no review decisions recorded yet") rather than printing zeros that
look like results.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Sequence

from benchmark.corpus_plan import datasheet_table
from benchmark.preannotate import branch_provenance
from benchmark.review import DEFAULT_DIR, KINDS, ReviewStore, agreement
from benchmark.strata import quota_report
from contract.gold import GoldInstance


def provenance_counts(instances: Sequence[GoldInstance]) -> Counter:
    """``model-proposed`` / ``reviewed`` / ``no branches`` / ``annotated by hand`` over instances."""
    c: Counter = Counter()
    for i in instances:
        if not i.gold_branches:
            c["no branches"] += 1
            continue
        p = branch_provenance(i)
        if p is None:
            c["annotated by hand (no proposal recorded)"] += 1
        elif p.startswith("reviewed:"):
            c["reviewed by a person"] += 1
        else:
            c["model-proposed, not yet reviewed"] += 1
    return c


def annotators_for(directory: Path, kind: str) -> list[str]:
    prefix = f"{kind}_"
    return sorted(p.stem[len(prefix):] for p in Path(directory).glob(f"{prefix}*.jsonl"))


def review_rows(directory: Path) -> list[dict]:
    """One row per (kind, annotator) with decisions and hours, plus a kappa row per annotator pair."""
    rows: list[dict] = []
    for kind in KINDS:
        names = annotators_for(directory, kind)
        stores = {n: ReviewStore(directory, kind, n) for n in names}
        for n, st in stores.items():
            rows.append({"kind": kind, "annotator": n, "decisions": len(st.load()), "hours": st.hours()})
        for a, b in combinations(names, 2):
            try:
                k = agreement(stores[a], stores[b])
                rows.append({"kind": kind, "pair": f"{a} / {b}", "n": k.n, "kappa": k.kappa,
                             "band": k.band, "degenerate": k.degenerate})
            except ValueError as e:
                rows.append({"kind": kind, "pair": f"{a} / {b}", "error": str(e)})
    return rows


def render(instances: Sequence[GoldInstance], review_dir: Path | None = None, *,
           contrast_ids: frozenset[str] = frozenset(), release_manifest: dict | None = None) -> str:
    quota = quota_report(list(instances), contrast_ids=contrast_ids)
    out = ["# Datasheet: generated sections", "",
           "Every number below is computed from the data by `python -m benchmark.datasheet_v7`.", "",
           "## Composition by tier", "", datasheet_table(list(instances)), "",
           "## Strata by split", "", "```", quota.render(), "```", "",
           "## Label provenance", ""]
    prov = provenance_counts(instances)
    out += [f"- {k}: **{v}**" for k, v in sorted(prov.items())] or ["- no instances"]
    out += ["", "Model proposals are shown as proposals. They count as human-annotated only after a person "
                "has reviewed them.", "", "## Annotation: decisions, hours and agreement", ""]
    rows = review_rows(review_dir or DEFAULT_DIR)
    if not rows:
        out.append("No review decisions recorded yet.")
    else:
        out += ["| task | annotator | decisions | hours |", "|---|---|---:|---:|"]
        out += [f"| {r['kind']} | {r['annotator']} | {r['decisions']} | {r['hours']:.2f} |"
                for r in rows if "annotator" in r]
        pairs = [r for r in rows if "pair" in r]
        out += ["", "| task | pair | n | kappa | reading |", "|---|---|---:|---:|---|"]
        for r in pairs:
            if "error" in r:
                out.append(f"| {r['kind']} | {r['pair']} | | | not computed: {r['error']} |")
            elif r["degenerate"]:
                out.append(f"| {r['kind']} | {r['pair']} | {r['n']} | undefined | {r['band']} |")
            else:
                out.append(f"| {r['kind']} | {r['pair']} | {r['n']} | {r['kappa']:.3f} | {r['band']} |")
        if not pairs:
            out.append("| | | | | only one annotator so far: no agreement can be reported |")
        total = sum(r["hours"] for r in rows if "annotator" in r)
        out += ["", f"Annotation time recorded: **{total:.1f} hours**."]
    if release_manifest is not None:
        out += ["", "## Release", "",
                f"- instances: **{release_manifest['n_instances']}**; passages: **{release_manifest['n_passages']}**; "
                f"withheld (not redistributable): **{release_manifest['n_redacted']}**",
                *[f"- open passages ({lic}): **{n}**"
                  for lic, n in release_manifest.get("open_passages_by_licence", {}).items()],
                f"- unrestorable after redaction: **{len(release_manifest.get('unrestorable', []))}**"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/datasheet_generated.md"))
    ap.add_argument("--review-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--contrast", type=Path, default=None)
    ap.add_argument("--release", type=Path, default=None, help="a release bundle directory (reads manifest.json)")
    args = ap.parse_args(argv)

    from benchmark.strata import _contrast_ids
    from experiments.run_matrix import load_instances

    manifest = json.loads((args.release / "manifest.json").read_text(encoding="utf-8")) if args.release else None
    text = render(load_instances(args.data), args.review_dir, contrast_ids=_contrast_ids(args.contrast),
                  release_manifest=manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E4: where does end-to-end loss arise? Substitute gold, stage by stage (RQ2).

The pilot said the composition operator was nearly perfect given gold inputs (81 of 83 branches)
and the detector found 3% of conflicts. That is two measurements on different data. The cascade
puts them on *one* axis: take a system's end-to-end answer, then replace its internal stages with
gold one at a time and score every level with the same metric.

====  ================================================  ==================================
L0    the system's own end-to-end answer                everything predicted
L1    **gold frames**, composed by the system's         isolates extraction and alignment:
      partition operator and verbalised                 L1 - L0 is what finding the frames costs
L2    **gold structure**, verbalised directly           isolates composition: L2 - L1 is what the
      (the reference answer)                            operator and its typed scopes cost
====  ================================================  ==================================

and ``1 - L2`` is the share of instances **no structure can recover under this reader**: the
ceiling the metric itself imposes. Reporting it keeps stage losses from being blamed on a system
when they belong to the instrument.

Each delta is a paired comparison on the same instances (exact McNemar, BCa interval), so a stage
that appears to cost a lot on 40 instances is shown with its uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from baselines.registry import verbalise_branches
from composition.partition import pieces_from_branches
from contract.gold import GoldInstance
from experiments.analysis import Paired, cca_by_instance, paired
from metrics.case_accuracy import DEFAULT_NLI_THRESHOLD, NLIScorerLike, Reader
from pipelines.scopecompose import compose_pieces

LEVELS = ("L0", "L1", "L2")
DESCRIPTIONS = {"L0": "end-to-end", "L1": "gold frames, composed", "L2": "gold structure, verbalised"}


def gold_frames_answer(inst: GoldInstance) -> str:
    """Compose the gold branches with the system's operator, then verbalise.

    Differs from :func:`gold_structure_answer` only where the operator does something to the gold:
    a scope it cannot type is shown verbatim, an overridden branch is recorded, overlapping scopes are
    split into regions. That difference *is* the composition stage."""
    branches, _ = compose_pieces(pieces_from_branches(inst.gold_branches), "g")
    return verbalise_branches(branches)


def gold_structure_answer(inst: GoldInstance) -> str:
    """The gold branches verbalised directly: the reference answer."""
    return verbalise_branches(inst.gold_branches)


@dataclass
class Cascade:
    levels: dict[str, dict[str, bool]]
    deltas: dict[str, Paired]

    def accuracy(self, level: str) -> float:
        v = self.levels[level]
        return sum(v.values()) / len(v) if v else 0.0

    @property
    def ceiling_gap(self) -> float:
        """The share of instances not recovered even with gold structure: the instrument's limit."""
        return 1.0 - self.accuracy("L2")

    def render(self) -> str:
        lines = ["Oracle cascade (CCA-all strict)", "=" * 72]
        for lv in LEVELS:
            lines.append(f"  {lv}  {DESCRIPTIONS[lv]:<28}{self.accuracy(lv):.3f}")
        names = {"L1-L0": "extraction + alignment", "L2-L1": "composition + verbalisation"}
        lines.append("")
        for k, p in self.deltas.items():
            lines.append(f"  {k}  ({names[k]}): {p.render()}")
        lines.append(f"  ceiling: {self.ceiling_gap:.3f} of instances are not recovered even from gold structure "
                     "(a property of the reader and metric, not of the system)")
        return "\n".join(lines)


def cascade(instances: Sequence[GoldInstance], end_to_end: Mapping[str, str], reader: Reader,
            nli: NLIScorerLike, *, threshold: float = DEFAULT_NLI_THRESHOLD, seed: int = 0) -> Cascade:
    """Score all three levels on the same instances and pair each adjacent step."""
    eligible = [i for i in instances if i.personas and i.gold_branches]
    if not eligible:
        raise ValueError("the cascade needs instances with gold branches and personas")
    answers = {
        "L0": {i.instance_id: end_to_end[i.instance_id] for i in eligible},
        "L1": {i.instance_id: gold_frames_answer(i) for i in eligible},
        "L2": {i.instance_id: gold_structure_answer(i) for i in eligible},
    }
    levels = {lv: {iid: r.cca_all for iid, r in cca_by_instance(eligible, a, reader, nli, threshold=threshold).items()}
              for lv, a in answers.items()}
    deltas = {"L1-L0": paired(levels["L1"], levels["L0"], "L1", "L0", seed=seed),
              "L2-L1": paired(levels["L2"], levels["L1"], "L2", "L1", seed=seed)}
    return Cascade(levels, deltas)

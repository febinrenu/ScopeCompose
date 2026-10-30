"""O3: the hybrid system (proposal v8 section 5.6).

The question this system answers is *what happens if the structured parse is
checked against explicit scope reasoning?* It is the paper's resolution
hypothesis (H3): O3 is compared with the best structured prompting (B7c / B8)
on the same backbone and the same passages.

Steps, all on the cell's own backbone:

1. **Parse** -- the same compact structured parse as B7c (one call).
2. **Frames** -- QNFA frame extraction with verbatim spans and decision-variable
   signatures (one batched call).
3. **Group** -- frames whose signatures unify (equivalent, subsumes, overlap)
   govern the same decision; distinct decisions are never composed together.
4. **Fold and compose** -- each group's signature differences become typed
   applicability, and :func:`composition.partition.partition_compose` assigns an
   outcome to every region of the case space (Proposition 2). Nothing a detector
   decided can delete a frame (Proposition 3).
5. **Repair against the parse, and log it** -- draft branches that no frame
   supports are *kept* (a recall backstop) and logged as ``parse_only``; frames
   the parse missed are logged as ``injected``. Nothing is repaired silently.
6. **Verbalise** with the same verbaliser as every structured system.

If extraction yields no usable frame the answer **falls back to the system's own
parse** and says so in the trace, so O3 is never worse than B7c merely because a
second call failed.

Hedged frames never become an assigned outcome: their span is stated verbatim
with its source, because "may receive a waiver" determines nothing for anyone.
"""

from __future__ import annotations

from typing import Sequence

from api_budget import Tier
from api_budget.client import LLMClient
from baselines.registry import (
    StructuredCompact,
    SystemAnswer,
    _Meter,
    verbalise_branches,
)
from composition.partition import Piece, TooManyFrames, partition_compose
from contract.frames import Modality, NormativeFrame
from contract.gold import Applicability, Branch
from contract.models import Passage
from detection.frames import FrameExtractor
from detection.signature import coreference_components, fold_component
from metrics.values import values_match


class _Metered:
    """A client proxy that routes every call through a :class:`_Meter`."""

    def __init__(self, client: LLMClient, meter: _Meter):
        self._client, self._meter = client, meter

    def complete(self, **kw):
        return self._meter.run(self._client, **kw)


def _branch(bid: str, passage: str | None, *, is_default: bool, condition: str | None, outcome: str,
            descriptor: str) -> Branch | None:
    try:
        return Branch(branch_id=bid, condition=None if is_default else (condition or "unspecified"),
                      outcome=outcome[:300],
                      applicability=Applicability(descriptor=(descriptor or "everyone")[:200],
                                                  is_default=is_default),
                      supporting_passage=passage)
    except Exception:
        return None


def compose_pieces(pieces: list[Piece], tag: str = "c") -> tuple[list[Branch], dict]:
    """Partition-compose one group of co-referential pieces into verbalisable branches.

    Shared by O3 (pieces from extracted frames) and the oracle cascade (pieces from gold
    branches), so a difference between the two can only come from what they were *given*, never
    from how they were composed. Uncovered regions are not stated (no source speaks to them);
    contested regions state both sources; untyped scopes are shown verbatim.
    """
    info: dict = {"frames": len(pieces), "regions": 0, "contested": 0, "indeterminate": 0,
                  "overridden": 0, "discarded": 0}
    try:
        part = partition_compose(pieces)
    except TooManyFrames:
        # Too many to partition exhaustively: report each piece as its own statement.
        out = [b for i, p in enumerate(pieces)
               if (b := _branch(f"{tag}x{i}", p.passage_id, is_default=p.applicability.is_default,
                                condition=p.applicability.descriptor, outcome=p.outcome,
                                descriptor=p.applicability.descriptor))]
        info["fallback"] = "too_many_frames"
        return out, info

    passage_of = {p.piece_id: p.passage_id for p in pieces}
    outcome_of = {p.piece_id: p.outcome for p in pieces}
    default_ids = {p.piece_id for p in pieces if p.applicability.is_default}
    branches: list[Branch] = []
    n = 0

    def add(**kw):
        nonlocal n
        b = _branch(f"{tag}b{n}", **kw)
        if b is not None:
            branches.append(b)
            n += 1

    for r in sorted(part.regions, key=lambda r: (len(r.inside - default_ids) != 0, sorted(r.inside))):
        if r.status == "uncovered":
            continue
        info["regions"] += 1
        base = not (r.inside - default_ids)
        if r.status == "assigned":
            add(passage=passage_of[r.governing[0]], is_default=base, condition=r.description,
                outcome=r.outcome or "", descriptor=r.description or "everyone")
        else:
            info["contested"] += 1
            said = "; ".join(f"{passage_of[g]} states {outcome_of[g]}" for g in r.governing)
            add(passage=None, is_default=False,
                condition=f"the sources disagree{' for ' + r.description if r.description else ''}",
                outcome=said, descriptor=r.description or "everyone")
    for p in part.indeterminate:
        info["indeterminate"] += 1
        add(passage=p.passage_id, is_default=False, condition=p.applicability.descriptor,
            outcome=p.outcome, descriptor=p.applicability.descriptor)
    info["overridden"], info["discarded"] = len(part.overridden), len(part.discarded)
    return branches, info


class ScopeCompose:
    """O3: parse + frames + partition composition, repaired against the parse."""

    id = "O3"

    def __init__(self, *, tier: Tier, domain: str, client: LLMClient | None = None):
        self.tier, self.domain, self._client = tier, domain, client

    @property
    def client(self) -> LLMClient:
        from api_budget import get_client

        return self._client if self._client is not None else get_client()

    # ------------------------------------------------------------------ #

    def _compose_component(self, frames: list[NormativeFrame], tag: str) -> tuple[list[Branch], dict]:
        sigs = [f.signature for f in frames]
        apps = fold_component(sigs, [f.applicability for f in frames])
        pieces = [Piece(f"{tag}f{i}", apps[i], f.outcome, f.passage_id) for i, f in enumerate(frames)]
        return compose_pieces(pieces, tag)

    def answer(self, query: str, passages: Sequence[Passage], *, conflicts=()) -> SystemAnswer:
        m = _Meter()
        draft, _ = StructuredCompact(tier=self.tier, client=self.client)._parse(m, query, passages)
        extractor = FrameExtractor(self.domain, tier=self.tier, client=_Metered(self.client, m))
        ext = extractor.extract(query, passages)

        trace: dict = {"parse_ok": bool(draft), "n_draft": len(draft), "frames": len(ext.frames),
                       "extraction": {"yield": ext.stats.frame_yield, "dropped_span": ext.stats.dropped_span,
                                      "oov_rate": ext.stats.oov_rate},
                       "components": [], "injected": [], "parse_only": [], "hedged": []}

        determining = [f for f in ext.frames if f.modality is not Modality.HEDGED]
        hedged = [f for f in ext.frames if f.modality is Modality.HEDGED]

        if not ext.frames:
            trace["fallback"] = "no_frames: answer is the system's own parse"
            return SystemAnswer(self.id, verbalise_branches(draft), m.calls, m.prompt, m.completion,
                                tuple(draft), trace)

        groups, uncertain = coreference_components([f.signature for f in determining])
        trace["uncertain_pairs"] = len(uncertain)

        # One section per decision. They are verbalised SEPARATELY: merging two decisions into one
        # branch list would let the verbaliser present one decision's outcome as the other's
        # exception, which is exactly the misattachment this system exists to prevent.
        sections: list[list[Branch]] = []
        for gi, idxs in enumerate(groups):
            bs, info = self._compose_component([determining[i] for i in idxs], f"c{gi}")
            sections.append(bs)
            trace["components"].append(info)

        # Hedged frames are stated verbatim with their source and never given an assigned outcome.
        hedged_sentences: list[str] = []
        for f in hedged:
            trace["hedged"].append(f.passage_id)
            hedged_sentences.append(f"{f.span.strip().rstrip('.')} (per {f.passage_id}).")

        # Repair against the parse -- logged, never silent.
        composed_outcomes = [b.outcome for bs in sections for b in bs]
        extras: list[Branch] = []
        for i, d in enumerate(draft):
            if not any(values_match(d.outcome, o) or values_match(o, d.outcome) for o in composed_outcomes):
                trace["parse_only"].append(d.outcome)
                kept = _branch(f"d{i}", d.supporting_passage, is_default=False,
                               condition=d.condition or d.applicability.descriptor, outcome=d.outcome,
                               descriptor=d.applicability.descriptor)
                if kept:
                    extras.append(kept)
        sections.append(extras)
        for f in determining:
            if not any(values_match(f.outcome, d.outcome) or values_match(d.outcome, f.outcome) for d in draft):
                trace["injected"].append(f.outcome)

        text = " ".join([t for t in (verbalise_branches(bs) for bs in sections) if t] + hedged_sentences)
        return SystemAnswer(self.id, text, m.calls, m.prompt, m.completion,
                            tuple(b for bs in sections for b in bs), trace)

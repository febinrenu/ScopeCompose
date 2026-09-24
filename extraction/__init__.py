"""B1: Contrastive Scope Probing and the grounding gate.

Recovers the applicability condition a passage implies, including when the
passage never marks itself as an exception -- which is the hard case and the
one the method is judged on.

Two pieces, and the second is what makes the first safe to use:

* :class:`~extraction.probing.ContrastiveScopeProbe` asks *contrastive*
  questions ("under what circumstances does the rule not apply?") rather than
  open ones, because a question carrying a presupposition is harder to answer
  vacuously.
* The **grounding gate** scores ``passage |= (condition AND outcome)`` with a
  local NLI cross-encoder and drops what the cited passage does not support. A
  condition proposer without a gate is a hallucination engine; the gate's
  rejection rate is reported rather than assumed.
"""

from extraction.probing import (
    TAU_GROUND,
    CandidateCondition,
    ContrastiveScopeProbe,
    ProbeResult,
    ProbeStats,
)

__all__ = [
    "TAU_GROUND",
    "CandidateCondition",
    "ContrastiveScopeProbe",
    "ProbeResult",
    "ProbeStats",
]

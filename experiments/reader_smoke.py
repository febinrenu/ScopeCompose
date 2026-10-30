"""A five-answer smoke test of the real reader and the real NLI model.

This is **not** validation -- gate V2 is validated on real data against the human entailment
audit. It is the check to run first after the reader model changes, because it catches the
failure that matters most in one minute: a reader so conservative that the *gold* answer scores
zero. (The first reader prompt did exactly that: told only "do not assume the person qualifies for
anything the answer does not say", an 8B model answered NOT_DETERMINED for a standard-card holder
even though the answer plainly stated the general rule.)

Expected behaviour, one row per answer:

=============  ==========================================================================
``scoped``     CCA strict = True  (both people get their outcome, with a quoted span)
``selection``  CCA strict = False (the exception person is told the general rule: wrong)
``hedge``      strict = False but lenient = True (the safeguards remove the gap-filling)
``unattached`` False  (every outcome listed, none attached to anyone)
``null``       False  (a refusal says nothing)
=============  ==========================================================================

Needs Ollama running with ``scopecompose-reader`` (``ollama create scopecompose-reader -f
config/ollama/reader.Modelfile``) and the cached DeBERTa NLI model. No API tokens are spent.
"""

from __future__ import annotations

import sys

from contract.gold import Applicability, Branch, GoldInstance, Persona
from contract.models import ConflictType, Construction, Domain, Passage, ScopeRelation

ANSWERS = {
    "scoped": "A 3% fee applies (per p0). However, if the holder has a premium-tier card, no fee applies (per p1).",
    "selection": "A 3% fee applies.",
    "hedge": "A 3% fee applies. Premium customers may receive fee waivers.",
    "unattached": "Depending on your circumstances, either a 3% fee applies or no fee applies.",
    "null": "I'm sorry, I don't have enough information to answer that question.",
}

#: (strict, lenient) the reader should produce for each answer.
EXPECTED = {"scoped": (True, True), "selection": (False, False), "hedge": (False, True),
            "unattached": (False, False), "null": (False, False)}


def fixture() -> GoldInstance:
    return GoldInstance(
        instance_id="reader_smoke", query="Do I pay a fee on transfers?", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL,
        passages=[Passage(id="p0", text="Transfers incur a 3% fee."),
                  Passage(id="p1", text="The fee is waived for premium-tier cardholders.")],
        gold_conflict_type=ConflictType.CONDITIONAL, gold_scope_relation=ScopeRelation.REFINEMENT,
        gold_branches=[
            Branch(branch_id="b0", condition=None, outcome="a 3% fee applies",
                   applicability=Applicability(descriptor="everyone", is_default=True), supporting_passage="p0"),
            Branch(branch_id="b1", condition="the holder has a premium-tier card", outcome="no fee applies",
                   applicability=Applicability(descriptor="premium"), supporting_passage="p1")],
        personas=[Persona(persona_id="u_default", branch_id="b0",
                          statement="I hold a standard card and I am sending a transfer."),
                  Persona(persona_id="u_exc", branch_id="b1",
                          statement="I hold a premium-tier card and I am sending a transfer.")])


def run(reader, nli) -> dict[str, tuple[bool, bool]]:
    from metrics.case_accuracy import score_instance

    inst = fixture()
    return {name: (r.cca_all, r.cca_all_lenient)
            for name, a in ANSWERS.items() for r in [score_instance(inst, a, reader, nli)]}


def main() -> int:
    from detection.nli import CrossEncoderNLI
    from metrics.case_accuracy import LLMReader

    got = run(LLMReader(step="reader_smoke"), CrossEncoderNLI())
    bad = 0
    for name, (strict, lenient) in got.items():
        ok = (strict, lenient) == EXPECTED[name]
        bad += not ok
        print(f"  {name:<11} strict={strict!s:<5} lenient={lenient!s:<5} expected={EXPECTED[name]}  "
              f"{'ok' if ok else 'UNEXPECTED'}")
    print("reader smoke test:", "PASS" if not bad else f"{bad} UNEXPECTED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

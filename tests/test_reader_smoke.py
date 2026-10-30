"""The reader smoke fixture is valid and, with an idealised reader, produces the expected table.

The real model is not touched here (tests are offline); ``experiments/reader_smoke.py`` is run by hand."""

from __future__ import annotations

from detection.nli import NLIScores
from experiments.reader_smoke import ANSWERS, EXPECTED, fixture, run
from metrics.case_accuracy import NOT_DETERMINED, ReaderOutput
from metrics.case_accuracy import READER_PROMPT_SHA, READER_SYSTEM


def test_the_fixture_is_a_valid_refinement_with_two_personas():
    inst = fixture()
    assert inst.default_branch.outcome == "a 3% fee applies" and len(inst.exception_branches) == 1
    assert {p.persona_id for p in inst.personas} == {"u_default", "u_exc"}
    assert set(ANSWERS) == set(EXPECTED)


def test_an_idealised_reader_reproduces_the_expected_table():
    """Quotes the sentence that addresses the person; says nothing when none does. This pins the
    *scoring* half of the expectation table independently of any model."""

    def reader(*, query, answer, persona):
        sents = [s.strip() for s in answer.replace(" However, ", ". However, ").split(". ") if s.strip()]
        if "premium" in persona:
            pick = next((s for s in sents if "premium" in s.lower() or "waive" in s.lower() or "no fee" in s.lower()),
                        None)
            if pick is None and "3%" in answer and "However" not in answer and "Premium" not in answer:
                pick = sents[0]                                    # a selection answer: the only rule stated
            if pick is None:
                return ReaderOutput(NOT_DETERMINED, "")
            outcome = "no fee applies" if ("waive" in pick.lower() or "However" in pick or "no fee" in pick.lower()) \
                else pick
            return ReaderOutput("fee waived" if "may receive" in pick else outcome, pick)
        pick = next((s for s in sents if "3%" in s), None)
        return ReaderOutput(pick or NOT_DETERMINED, pick or "")

    class NLI:
        def score_batch(self, pairs):
            # entails only when the quoted span states what the hypothesis says
            out = []
            for premise, hyp in pairs:
                said = ("no fee" in premise.lower() or "waive" in premise.lower()) == ("no fee" in hyp.lower())
                firm = "may receive" not in premise.lower()
                out.append(NLIScores(0.95 if said and firm else 0.05, 0.05, 0.0))
            return out

    got = run(reader, NLI())
    assert got["scoped"] == (True, True)
    assert got["null"] == (False, False) and got["unattached"] == (False, False)
    assert got["hedge"][0] is False, "the hedge never passes strict"


def test_the_reader_prompt_defines_defaults_and_keeps_the_guard():
    """Regression: the prompt must tell the reader that a person who meets no exception gets the general
    rule, otherwise an 8B model scores the gold answer zero -- and must still forbid outside knowledge."""
    assert "covered by the general rule" in READER_SYSTEM
    assert "Do not use outside knowledge" in READER_SYSTEM and "COPIED WORD FOR WORD" in READER_SYSTEM
    assert "hedges" in READER_SYSTEM and len(READER_PROMPT_SHA) == 64

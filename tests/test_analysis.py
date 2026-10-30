"""The confirmatory analysis on synthetic outcomes with known answers."""

from __future__ import annotations

import pytest

from contract.mock import generate
from detection.nli import NLIScores
from experiments.analysis import (
    Paired,
    bootstrap_p_at_most,
    cca_by_instance,
    confirmatory,
    paired,
    persona_setting_accuracy,
)
from metrics.case_accuracy import NOT_DETERMINED, ReaderOutput


def outcomes(n, correct_ids):
    return {f"i{k:03d}": (k in correct_ids) for k in range(n)}


# --------------------------------------------------------------------------- #
# paired
# --------------------------------------------------------------------------- #


def test_a_clear_paired_difference_is_found_with_every_summary_consistent():
    n = 60
    a = outcomes(n, set(range(50)))            # 50/60
    b = outcomes(n, set(range(25, 35)))        # 10/60, overlapping on 25..34
    r = paired(a, b, "A", "B")
    assert r.n == 60 and r.acc_a == pytest.approx(50 / 60) and r.acc_b == pytest.approx(10 / 60)
    assert r.b == 40 and r.c == 0 and r.p_exact < 1e-9
    assert r.diff == pytest.approx(40 / 60) and r.ci_low > 0.4 and r.sign_flip_p < 0.001
    assert r.discordance == pytest.approx(40 / 60)


def test_identical_systems_have_no_difference_and_no_power_figure():
    a = outcomes(40, set(range(20)))
    r = paired(a, dict(a))
    assert r.diff == 0 and r.p_exact == 1.0 and r.discordance == 0 and r.mdd_points is None
    assert r.sign_flip_p == 1.0


def test_the_exact_test_is_used_even_when_the_default_would_approximate():
    """30 discordant items: the repo default would switch to chi-square here."""
    from metrics.power_sim import _two_sided_binom_p

    n = 60
    a = outcomes(n, set(range(0, 21)) | set(range(40, 60)))
    b = outcomes(n, set(range(0, 5)) | set(range(21, 40)) | set(range(40, 60)))
    r = paired(a, b)
    assert r.b + r.c >= 25
    assert r.p_exact == pytest.approx(_two_sided_binom_p(r.b, r.b + r.c))


def test_a_small_n_states_what_it_cannot_detect():
    a = outcomes(20, set(range(12)))
    b = outcomes(20, set(range(8)))
    r = paired(a, b)
    assert "cannot detect a true difference smaller than" in r.underpowered_for or "no effect" in r.underpowered_for
    assert r.mdd_points is None or r.mdd_points > 20, "20 instances cannot see a small effect"


def test_systems_scored_on_different_instances_cannot_be_paired():
    with pytest.raises(ValueError, match="different instances"):
        paired({"x": True, "y": False}, {"x": True, "z": False})
    with pytest.raises(ValueError, match="share no"):
        paired({}, {})


def test_pairing_is_seeded_and_order_independent():
    a, b = outcomes(50, set(range(30))), outcomes(50, set(range(10, 40)))
    assert paired(a, b, seed=3) == paired(a, b, seed=3)
    assert paired(dict(reversed(list(a.items()))), b, seed=3) == paired(a, b, seed=3)


def test_render_mentions_the_test_the_interval_and_the_power_limit():
    text = paired(outcomes(40, set(range(30))), outcomes(40, set(range(10)))).render()
    assert "exact McNemar" in text and "sign-flip" in text and "discordant" in text


# --------------------------------------------------------------------------- #
# the confirmatory family
# --------------------------------------------------------------------------- #


def family(n=80, *, b3_correct, b0_correct, o3_correct, comp_correct):
    return {"B0": outcomes(n, b0_correct), "B3": outcomes(n, b3_correct),
            "O3": outcomes(n, o3_correct), "B7c": outcomes(n, comp_correct), "B8": outcomes(n, comp_correct)}


def test_a_clean_win_on_both_hypotheses_is_supported_when_s1_is_large_enough():
    cca = family(b0_correct=set(range(0, 50)), b3_correct=set(range(0, 15)),
                 o3_correct=set(range(0, 70)), comp_correct=set(range(0, 40)))
    c = confirmatory(cca, comparator="B7c", s1_test=80)
    assert not c.exploratory
    assert c.supported("H1") and c.supported("H3")
    assert all(rej for _, rej in c.adjusted.values())


def test_the_narrowing_rule_makes_every_result_exploratory_even_if_significant():
    cca = family(b0_correct=set(range(0, 50)), b3_correct=set(range(0, 15)),
                 o3_correct=set(range(0, 70)), comp_correct=set(range(0, 40)))
    c = confirmatory(cca, comparator="B7c", s1_test=45)
    assert c.exploratory and "below the pre-registered floor of 60" in c.exploratory_reason
    assert c.adjusted["H1"][1] and not c.supported("H1"), "significant, but not confirmatory"
    assert "EXPLORATORY" in c.render()


def test_a_significant_difference_in_the_wrong_direction_does_not_support_h1():
    cca = family(b0_correct=set(range(0, 15)), b3_correct=set(range(0, 50)),   # B3 BETTER than B0
                 o3_correct=set(range(0, 70)), comp_correct=set(range(0, 40)))
    c = confirmatory(cca, comparator="B7c", s1_test=80)
    assert c.adjusted["H1"][1] and not c.directions_supported["H1"] and not c.supported("H1")
    assert "OPPOSITE to prediction" in c.render()


def test_holm_adjusts_the_family_so_a_borderline_pair_can_fail_after_correction():
    cca = family(b0_correct=set(range(0, 40)), b3_correct=set(range(0, 30)),
                 o3_correct=set(range(0, 52)), comp_correct=set(range(0, 40)))
    c = confirmatory(cca, comparator="B7c", s1_test=80)
    for h, r in c.results.items():
        assert c.adjusted[h][0] >= r.p_exact - 1e-12, "Holm never lowers a p-value"


def test_the_h3_comparator_is_the_one_pre_registered():
    cca = family(b0_correct=set(range(40)), b3_correct=set(range(20)), o3_correct=set(range(60)),
                 comp_correct=set(range(30)))
    cca["B8"] = outcomes(80, set(range(58)))
    assert confirmatory(cca, comparator="B7c", s1_test=80).results["H3"].name_b == "B7c"
    assert confirmatory(cca, comparator="B8", s1_test=80).results["H3"].name_b == "B8"


def test_h2_joins_the_family_when_supplied_and_is_adjusted_with_it():
    cca = family(b0_correct=set(range(50)), b3_correct=set(range(15)), o3_correct=set(range(70)),
                 comp_correct=set(range(40)))
    without = confirmatory(cca, comparator="B7c", s1_test=80)
    with_h2 = confirmatory(cca, comparator="B7c", s1_test=80, h2_p=0.02)
    assert set(without.adjusted) == {"H1", "H3"} and set(with_h2.adjusted) == {"H1", "H2", "H3"}
    assert with_h2.adjusted["H1"][0] >= without.adjusted["H1"][0]


def test_a_missing_system_is_an_error():
    with pytest.raises(KeyError, match="missing system"):
        confirmatory({"B0": {}}, comparator="B7c", s1_test=80)


# --------------------------------------------------------------------------- #
# bootstrap p for H2
# --------------------------------------------------------------------------- #


def test_bootstrap_p_is_small_when_the_statistic_clears_the_threshold_and_large_when_not():
    units = [1.0] * 40 + [0.0] * 10
    mean_ = lambda u: sum(u) / len(u)  # noqa: E731
    assert bootstrap_p_at_most(units, mean_, 0.5, n_resamples=2000) < 0.01
    assert bootstrap_p_at_most(units, mean_, 0.8, n_resamples=2000) > 0.3
    assert bootstrap_p_at_most(units, mean_, 0.5, seed=1) == bootstrap_p_at_most(units, mean_, 0.5, seed=1)
    with pytest.raises(ValueError):
        bootstrap_p_at_most([], mean_, 0.5)


def test_an_undefined_resample_counts_against_the_hypothesis_not_for_it():
    units = [0, 1]

    def undefined(u):
        raise ZeroDivisionError

    assert bootstrap_p_at_most(units, undefined, 0.1, n_resamples=50) == 1.0


# --------------------------------------------------------------------------- #
# per-instance CCA and the persona setting
# --------------------------------------------------------------------------- #


class ConstNLI:
    def score_batch(self, pairs):
        return [NLIScores(0.95, 0.05, 0.0) for _ in pairs]


def test_cca_by_instance_scores_gold_answers_and_refuses_missing_ones():
    insts = [i for i in generate(120, seed=0) if i.personas][:3]

    def reader(*, query, answer, persona):
        sents = [s.strip() for s in answer.replace(" However, ", ". However, ").split(". ") if s.strip()]
        pick = sents[1] if "-tier" in persona and len(sents) > 1 else sents[0]
        return ReaderOutput(pick, pick)

    answers = {i.instance_id: i.gold_scoped_answer for i in insts}
    got = cca_by_instance(insts, answers, reader, ConstNLI())
    assert set(got) == {i.instance_id for i in insts}
    with pytest.raises(KeyError, match="not a wrong one"):
        cca_by_instance(insts, {}, reader, ConstNLI())


def test_the_persona_setting_is_scored_deterministically_against_each_personas_gold_outcome():
    inst = next(i for i in generate(60, seed=0) if i.personas)
    answers = {}
    for p in inst.personas:
        gold = inst.branch_for_persona(p.persona_id).outcome
        answers[f"{inst.instance_id}|{p.persona_id}"] = f"For you, {gold}."
    acc = persona_setting_accuracy([inst], answers)
    assert all(acc.values()) and len(acc) == len(inst.personas)
    wrong = {k: "a 99% fee applies" for k in answers}
    assert not any(persona_setting_accuracy([inst], wrong).values())
    with pytest.raises(KeyError):
        persona_setting_accuracy([inst], {})

"""The paper's claims, each tied to the place in ``results.json`` that holds its value.

Adding a quantitative statement to the manuscript means adding it here first. The
assembler (``experiments.make_results``) and this list are held in step by a test that
builds results end to end and requires every path below to resolve.
"""

from __future__ import annotations

from reporting.claims import Claim, Registry

SYSTEMS = ("B0", "B3", "B7c", "B8", "O3")


def default_registry() -> Registry:
    r = Registry()
    add = r.add

    add(Claim("s1.n", "s1.n_test", "S1 instances in the test split", "int"))
    add(Claim("s1.floor", "s1.floor", "pre-registered S1 floor", "int"))
    add(Claim("s1.status", "s1.status", "confirmatory or exploratory, by the narrowing rule", "raw"))
    add(Claim("nli.tau", "nli_threshold.value", "entailment threshold for CCA-strict", "f2"))

    for s in SYSTEMS:
        add(Claim(f"cca.{s}", f"cca.{s}.cca_all", f"CCA-all strict, {s}", "pct"))
        add(Claim(f"cca.{s}.low", f"cca.{s}.cca_all_low", f"CCA-all strict, {s}, 95% interval low", "pct"))
        add(Claim(f"cca.{s}.high", f"cca.{s}.cca_all_high", f"CCA-all strict, {s}, 95% interval high", "pct"))
        add(Claim(f"cca.{s}.lenient", f"cca.{s}.cca_all_lenient", f"CCA-all lenient, {s}", "pct"))
        add(Claim(f"cca.{s}.exception", f"cca.{s}.cca_exception", f"accuracy on exception personas, {s}", "pct"))
        add(Claim(f"cca.{s}.default", f"cca.{s}.cca_default", f"accuracy on default personas, {s}", "pct"))
        add(Claim(f"cca.{s}.overgen", f"cca.{s}.over_generalisation",
                  f"default personas given the exception outcome, {s}", "pct"))

    for h, what in (("h1", "B3 vs B0"), ("h3", "O3 vs the pre-registered comparator")):
        add(Claim(f"{h}.diff", f"{h}.diff", f"{what}: accuracy difference (points)", "pts"))
        add(Claim(f"{h}.ci_low", f"{h}.ci_low", f"{what}: 95% BCa interval low", "pts"))
        add(Claim(f"{h}.ci_high", f"{h}.ci_high", f"{what}: 95% BCa interval high", "pts"))
        add(Claim(f"{h}.p", f"{h}.p_exact", f"{what}: exact McNemar p", "p"))
        add(Claim(f"{h}.p_flip", f"{h}.sign_flip_p", f"{what}: sign-flip p", "p"))
        add(Claim(f"{h}.n", f"{h}.n", f"{what}: instances compared", "int"))
        add(Claim(f"{h}.discordant_a", f"{h}.b", f"{what}: first system right, second wrong", "int"))
        add(Claim(f"{h}.discordant_b", f"{h}.c", f"{what}: second system right, first wrong", "int"))
        add(Claim(f"{h}.mdd", f"{h}.mdd_text", f"{what}: minimum detectable difference, as a sentence fragment",
                  "raw"))
        add(Claim(f"{h}.holm", f"confirmatory.adjusted_p.{h.upper()}", f"{what}: Holm-adjusted p", "p"))

    add(Claim("cascade.l0", "cascade.accuracy.L0", "end-to-end CCA-all (cascade level L0)", "pct"))
    add(Claim("cascade.l1", "cascade.accuracy.L1", "gold frames composed (L1)", "pct"))
    add(Claim("cascade.l2", "cascade.accuracy.L2", "gold structure verbalised (L2)", "pct"))
    add(Claim("cascade.ceiling", "cascade.ceiling_gap", "instances not recovered even from gold structure", "pct"))
    add(Claim("cascade.extraction", "cascade.extraction_and_alignment.diff",
              "accuracy gained by gold frames (L1 - L0, points)", "pts"))
    add(Claim("cascade.composition", "cascade.composition_and_verbalisation.diff",
              "accuracy gained by gold structure over gold frames (L2 - L1, points)", "pts"))
    return r

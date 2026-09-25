# WP0 — Prior-art review and positioning memo

**Status: COMPLETE as a first deliverable, 2026-09-25.** SG-DT verified against
the source; closed-venue sweep done and it narrowed the claim. Residual gaps
named in §4.
**Model sweep 2026-09-24. Human verification and venue sweep 2026-09-25.**

---

## Verdict

> **The bridging claim survives — but the distinction is narrower than the
> first draft of this memo stated, and must be worded precisely.**

This project targets the **resolution operator** in conflict-aware RAG: the step
deciding whether two separately retrieved, credibility-ranked passages are
selected between or composed, for the *conditional* case.

Two things narrow it, and both belong in §4 in the paper's own voice:

1. **Chen et al. named the failure first.** Silent Scope Omission — a system
   applying a general rule while silently dropping a nested exception — is
   their term, arrived at independently.
2. **They are not retrieval-innocent.** Verified against the source: the paper
   *does* discuss RAG and retrieval in its motivation, explicitly noting that
   retrieval can determine whether a decisive exception is visible at all. Its
   downstream statutory-reasoning experiment supplies a statute snippet as
   evidence alongside a case fact pattern.

### The sentence to write, and the one not to

> ❌ **"SG-DT does not involve retrieval"** / "prior work does not consider
> retrieval" / "the pipeline never has two documents."
>
> A reviewer points at their RAG discussion and the distinction collapses. The
> first draft of this memo made exactly this overclaim.

> ✅ **"Span-Grounded Deontic Trees address silent scope omission within an
> identified provision, by explicitly representing defeaters and their
> attachment to source spans. Our setting introduces an additional
> retrieval-level challenge: the governing rule and its scope-narrowing
> exception may be surfaced from separately retrieved documents. The system
> must therefore first recognise that the passages constitute a conditional
> relation, and then resolve their scope relation before composing an answer."**

The precise distinction, stated once:

| | SG-DT | this project |
|---|---|---|
| structural target | an **already-supplied** provision or executable unit | K separately retrieved passages |
| retrieval's role | discussed as a surrounding issue; may determine what is visible | **part of the phenomenon being benchmarked** |
| benchmark task | intra-provision defeasible scope parsing | cross-document scope resolution |

Their own §4 describes NormBench as benchmarking *intra-provision defeasible
scope parsing*, and their task definition is explicit: *given a provision X,
systems output a Span-Grounded Deontic Tree*, with nodes grounded to spans of
that current text scope (a full provision, or an executable unit when
segmented). That is the load-bearing fact, and it is verified.

**A third narrowing, from the venue sweep (§4).** The four-way relation is not
a new formalism. Specificity-based override has formalised `refinement` since
Poole and Bochman; rule-analysis literature already uses `disjoint` and
`redundant` as rule relations. What survives is the **operationalisation for
separately retrieved documents, and the benchmark** — plus `opposed`, which is
a genuine gap in specificity rather than an application of it.

**Confidence: high.** Both things that could have forced the fallback were
checked. Neither did; both narrowed the claim, and the narrowed claim is
defensible against the literature that narrowed it.

---


## How this pass was conducted, and what it is not

Every citation carried over from the proposal was checked against arXiv or the
ACL Anthology, and the title, authors and claims were compared against what the
memo asserted. That found four discrepancies, recorded below.

**What this pass could not do:**

- **The closed-venue sweep is not discharged.** KR, TPLP, AAAI, AIJ and JAR
  proceedings are not reachable by web search in the way arXiv is. The
  checklist below remains open, and it is the part most likely to contain a
  classical-KR anticipation of the four-way scope relation.
- **Full texts were mostly not read.** Conclusions rest on abstracts and
  anthology pages. The exception is SG-DT, the one paper where the answer
  decides the framing: it was read against the source on 2026-09-25, and that
  read corrected an overclaim this memo had made (see the Verdict).
- **A model reviewed a model's citations.** Shared failure modes; a claim we
  both misread in the same direction survives unchallenged. The SG-DT
  correction is a worked example — the model pass concluded "the pipeline never
  has two documents", which the paper's own RAG discussion contradicts.

---

## 1. Citation verification

All seven verified as real, with titles and authors matching. No fabricated
references.

| cited as | verified | status |
|---|---|---|
| Chen et al., SG-DT/NormBench, arXiv:2606.08932 | ✅ Jian Chen, Siyuan Li, Chucheng Wan, Zixuan Yuan | exact |
| ConditionalQA, arXiv:2110.06884 | ✅ Haitian Sun, William W. Cohen, Ruslan Salakhutdinov | exact |
| ConflictRAG, Wang et al., arXiv:2605.17301 | ✅ Chenyu Wang, Yueyuan Li, Yingmin Liu, Yang Shu | exact |
| DRAGged into Conflicts, arXiv:2506.08500 | ✅ Cattan, Jacovi, Ram, Herzig, Aharoni, Goldshtein, Ofek, Szpektor, Caciularu | exact |
| TCR, Ye et al., arXiv:2601.06842 | ✅ Hua Ye et al. | **see §2a** |
| Detecting Is Not Resolving, arXiv:2605.27157 | ✅ Zhe Yu, Wenpeng Xing, Chen Ye, Xuyang Teng, Bo Yang, Changting Lin, Meng Han | exact |
| Claim-Selective Certification, arXiv:2605.21949 | ✅ Shao Kan | exact |
| Defeasible NLI, Rudinger et al. | ✅ Rudinger, Shwartz, Hwang, Bhagavatula, Forbes, Le Bras, Smith, Choi — Findings of EMNLP 2020, pp. 4661–4675 | full author list recovered |
| S2G-RAG, Li et al., ACL 2026 | ✅ Li, Zou, Lv, Zhang, Zhou — arXiv:2604.23783, ACL 2026 (2026.acl-long.1185) | **ID recovered; memo had none** |

### Four discrepancies to fix in the write-up

**a. TCR is described imprecisely.** The memo calls it "makes the detect-resolve
process contestable." The paper is *"Seeing through the Conflict: Transparent
Knowledge Conflict Handling in RAG"*, and it addresses **internal (parametric)
knowledge versus retrieved external content** using dual contrastive encoders.
That is a *context–memory* conflict, not an inter-source conflict. The memo's
"orthogonal to which operator is used" is right, but for a stronger reason than
stated: it is a different conflict type.

**b. The "37%" figure is VERIFIED — with a wording constraint.** Confirmed
against the dataset-statistics table on 2026-09-25:

| complexity | count | share |
|---|---:|---:|
| Level 1 — simple | 834 | 36% |
| Level 2 — nested | 610 | 27% |
| **Level 3+ — recursive** | **846** | **37%** |

The paper states directly that *a significant portion of the dataset is
recursive (Level 3+, 37%)*. Citable.

The constraint is on phrasing. Their table classifies items by level; it does
not assert that 37% *require* deep resolution to answer.

- ❌ "37% of their benchmark needs deep exception nesting"
- ✅ "37% of NormBench items are classified as Level 3+ (recursive),
  corresponding to deep/counter-exception structures"

**c. Two further named findings, both usable.** The memo had only "Recursion
Decay". The abstract adds the **Auditability Trap** — *"models retrieve relevant
spans but fail to assemble correct control flow"* — and the results section
frames the same dissociation as the **Structure-Grounding Gap**: finding the
relevant span and attaching it to the correct logical parent are separate
abilities, and models have the first without the second.

That is this project's thesis observed *inside* a single provision, by an
independent group, and it is the strongest external support available for the
argument that locating evidence and composing it correctly come apart. Cite it
as support rather than treating it as competition.

**d. ConflictRAG's credibility mechanism.** The memo says "Entropy-TOPSIS". The
abstract describes an "entropy-based source credibility assessment" and a
diagnostic metric, **CARS**. TOPSIS is not confirmed. Verify before the related-
work section describes the mechanism.

---

## 2. The gating question, answered

> Does SG-DT anywhere define a resolution operator over *separately retrieved*
> competing sources?

**No — verified against the source, 2026-09-25.** Their task is defined as
*given a provision X, output a Span-Grounded Deontic Tree*, with nodes grounded
to spans of that current text scope; §4 calls NormBench a benchmark for
*intra-provision defeasible scope parsing*. Competing separately retrieved
documents are never the input unit.

**But the first version of this section overclaimed and has been corrected.**
It read: *"the pipeline never has two, because the retrieval in its Auditability
Trap is retrieval of spans inside the provision."* The paper discusses RAG and
retrieval directly in its motivation, and its downstream statutory-reasoning
experiment supplies a statute snippet as evidence. Writing "SG-DT does not
involve retrieval" would hand a reviewer an easy correction.

The surviving distinction is about **what the benchmark task takes as its
input unit**, not about whether retrieval is mentioned:

| | SG-DT | this project |
|---|---|---|
| structural input | an already-supplied provision or executable unit | K passages from retrieval, separately ranked |
| retrieval | discussed as a surrounding concern | part of the phenomenon benchmarked |
| failure | exception present in the text, dropped during parsing | exception present in the corpus, **discarded before parsing** |
| fix | structured intermediate representation | composition operator replacing selection |

A perfect SG-DT parser still exhibits this project's failure, because the
credibility ranker discards the exception passage before the parser sees both
together. That sentence is the positioning argument, and it is worth putting in
the paper close to verbatim.

**Confirmed 2026-09-25.** SG-DT §3–§4 read against the source. Input is a
single provision or executable unit; the bridging claim stands, with the wording
tightened as above.

---

## 3. New work found in this sweep

Five items not in the proposal. None anticipates the claim; two must be cited.

### CARE-RAG — the nearest neighbour, and the one a reviewer will raise
*Rethinking All Evidence: Enhancing Trustworthy RAG via Conflict-Driven
Summarization*, arXiv:2507.01281.

**Conflict-driven summarization** is the closest published thing to
"composition", and the name alone will prompt the question. The distinction is
real and must be stated:

- CARE-RAG **synthesises one reliable answer** from all evidence, having removed
  what it judges misleading. The output is a single answer with conflict handled
  *during* construction.
- This project **preserves each branch with its applicability condition
  attached**, and the output is structurally plural. Nothing is removed for being
  in tension with something else — that tension is the content.

CARE-RAG also mixes parametric and retrieved evidence, so like TCR it is partly
a context–memory method. **Cite it in related work; do not let a reviewer raise
it first.**

### Conflict-Aware RAG (ConScore) — ACM Web Conference 2026
doi:10.1145/3774904.3792289. Multi-stage learning with conflict signals; ConScore
compares generative probabilities across sources. A *detection and training*
signal, not a resolution operator. Cite alongside ConflictRAG.

### Ragability Benchmark — LREC 2026
ACL Anthology 2026.lrec-1.182. Inter-context conflicts requiring implicit
reasoning. Checked specifically for the conditional case: **it does not
distinguish conflicts where both sources are true under different scopes from
flat contradictions.** That absence supports the benchmark contribution and is
worth one sentence.

### ArbGraph — arXiv:2604.18362
*Conflict-Aware Evidence Arbitration for Reliable Long-Form RAG*. "Arbitration"
is selection vocabulary. Likely a direct instance of the operator this project
argues against — **worth reading properly**, as it may be a strong baseline or a
sharp foil.

### DeFAb — arXiv:2606.18557
*A Verifiable Benchmark for Defeasible Abduction in Foundation Models*.
Defeasible reasoning benchmark, no retrieval component. Relevant to the
defeasible-logic lineage rather than the RAG line.

---

## 4. Closed-venue sweep — done 2026-09-25, and it narrowed the claim

**Result: prior art found. The novelty is narrower than the proposal assumed,
and the paper is better for knowing it now.**

### What was found

The underlying logic is emphatically not new. Reasoning about rules with
exceptions has been formalised for decades:

| area | anchor | bearing on this project |
|---|---|---|
| default logic | Reiter (1980) | exceptions to default assumptions |
| specificity | Poole; Bochman | **the `refinement` relation, essentially** |
| defeasible logic | Pollock; Prakken & Sartor | override and priority between rules |
| argumentation | Dung (1995) and after | conflict resolution between arguments |
| rule analysis | rule-base verification literature | uses **disjoint**, **redundant** and **conflicting** as rule relations |
| KB refinement | refinement-operator literature | uses **refinement** for condition specialisation |
| AI & Law | *lex specialis* | a more specific rule overrides a more general one |

Bochman states the specificity principle in terms close enough to be quoted
against a novelty claim: *more specific default rules override less specific
rules in conflict*, with the canonical pattern `A → C` against `A ∧ B → ¬C`.
That is `refinement`, named and formalised decades ago.

Verified in the KR proceedings directly:

- **A Rule-Based Approach to Specifying Preferences over Conflicting Facts and
  Querying Inconsistent Knowledge Bases** — Bienvenu, Bourgaux, Inoue, Jean.
  KR 2025 main track.
- **Reasoning in Defeasible Description Logics with System W and Lexicographic
  Inference** — Casini, Haldimann, Meyer. KR 2025 main track.
- **Deontic Reasoning Based on Inconsistency Measures** — Arieli, van Berkel,
  Raddaoui, Strasser. KR 2024. Normative conflict, nonmonotonic and
  paraconsistent entailment.

IJCAI-ECAI 2026's KRR track carries neighbouring work (non-monotonic reasoning,
ASP, argumentation, description logics) but a title-and-abstract scan found no
direct match for the four-way classification.

### What survives

No prior work was found that says: *given two retrieved passages, classify
their relationship as refinement / disjoint / redundant / opposed.* The
vocabulary is old; the **operational combination, applied to separately
retrieved documents, as an annotated benchmark task** is where the contribution
now sits.

One part is stronger than the rest and worth leading with. Ordinary specificity
answers the nested case:

```
    B ⊂ A                      → specificity decides
```

It does **not** answer:

```
    A ∩ B ≠ ∅,  A ⊄ B,  B ⊄ A,  outcomes conflict on the overlap
```

That is `opposed`, and it is a structural gap in specificity-based override
rather than an application of it. `wp1_fin_opp_053` and `wp1_imm_opp_055` are
built precisely on that shape — and 055 deliberately removes the credibility
and recency crutches, so a system has nothing but scope to reason with.

`redundant` is the other operationally useful one: `B ⊂ A` **and**
`outcome(B) = outcome(A)` means the narrower passage is a restatement, not an
exception. A system that branches on every narrower passage fabricates
exceptions, and `wp1_fin_red_049` exists to catch exactly that.

### The framing this forces

> **Prior work has extensively formalised defeasible reasoning, exceptions,
> specificity-based priority, and conflicts among rules. Our contribution is not
> a new defeasible logic; rather, we operationalise scope relations between
> separately retrieved rule-bearing documents, and introduce a benchmark that
> distinguishes nested exception/refinement, redundant specialisation, disjoint
> applicability, and non-nested overlapping opposition.**

**Do not write:**

- ❌ "No previous work has studied rule-exception relationships."
- ❌ "We are the first to define refinement, disjointness, redundancy and opposition."
- ❌ "Existing defeasible logic cannot distinguish these relations."
- ❌ "Specificity has not considered scope overlap."

**Write instead:**

- ✅ "Existing formalisms provide mechanisms for defeasibility, specificity and conflict resolution."
- ✅ "We operationalise applicability-set relations for retrieved textual rules."
- ✅ "We study these relations as an information-retrieval and language-understanding problem."
- ✅ "Our benchmark evaluates whether systems preserve the applicability conditions and outcomes of cross-document rules."

### Residual

The sweep covered the publicly indexed portions of KR 2024–2025, IJCAI-ECAI
2026 KRR, and the classical default/defeasible/specificity literature. TPLP,
AIJ and JAR were not searched exhaustively, and a specificity paper that
happens to enumerate the same four cases could still exist. The framing above
survives that discovery, which is the point of adopting it now: it claims the
operationalisation and the benchmark, not the taxonomy.

---

## 5. Consequences for the paper

1. **Credit Chen et al. for Silent Scope Omission explicitly**, in the project's
   own voice, then draw the retrieval-stage distinction. Reads as scholarship
   when volunteered; reads as concealment when a reviewer finds it.
2. **Never write that SG-DT does not involve retrieval.** It does, in the
   motivation. The distinction is about the benchmark's input unit, and the
   exact sentence to use is in the Verdict.
3. **Never claim the four relations as a new formalism.** §4 lists the prior art
   and the sentence to use instead. Lead on the operationalisation and the
   benchmark, and on `opposed` as the case specificity does not cover.
4. **Cite the 37% figure using their terminology** — "classified as Level 3+
   (recursive)", not "needs deep nesting". Verified; see §1b.
5. **Use the Structure-Grounding Gap as support, not competition.** An
   independent group found that locating the relevant span and attaching it to
   the right logical parent are separate abilities. That is this project's
   thesis observed inside a single provision, and the strongest external
   evidence available for it.
6. **Add a related-work paragraph on conflict-driven summarization** (CARE-RAG),
   distinguishing synthesis-into-one from composition-preserving-branches.
7. **Fix the TCR description** — it is context–memory, not inter-source.
8. **Verify ConflictRAG's mechanism** before describing it as Entropy-TOPSIS.
9. **Claim the benchmark gap explicitly** — CONFLICTS and Ragability both
   taxonomise conflicts without separating "both true under different scopes"
   from "contradiction". That is the hole this corpus fills.

---

## 6. What would overturn this verdict

Stated up front so the fallback is not a retreat:

- ~~**SG-DT turns out to evaluate over a retrieved set**~~ — **checked and
  ruled out** on 2026-09-25. Its structural input is a single provision or
  executable unit. This was the largest single risk and it is closed.
- ~~**A KR/TPLP paper already formalises the four-way relation**~~ — **checked
  2026-09-25, and substantially confirmed.** Specificity already covers
  `refinement`; rule-analysis literature already uses `disjoint` and
  `redundant`. The response was not to defend the taxonomy but to stop claiming
  it: the contribution is the operationalisation and the benchmark. That framing
  is stable even if a paper enumerating all four cases turns up later.
- **Largest remaining risk is now empirical, not positional** — the decisive
  experiment currently shows the pipeline behind the baseline at -12.5%
  [-37.5%, +12.5%] on 24 branches, with almost no power. A properly sized corpus
  could confirm that.
- **ArbGraph already composes rather than arbitrates** (the name suggests not,
  but the name is not the paper) → the operator claim narrows to the conditional
  case specifically.

None of the three would sink the project. All three would change §3 and §4, and
all three are cheap to check now and expensive to discover in week 14.

---

## 7. Outstanding actions, ranked

Both gating checks are done. What remains is tidy-up, not risk.

1. Read ArbGraph (arXiv:2604.18362) — "arbitration" is selection vocabulary, so
   it is a likely foil or baseline. The only unread paper that could still
   change a claim.
2. Confirm ConflictRAG's credibility mechanism before describing it as
   Entropy-TOPSIS (§1d).
3. Read CARE-RAG (arXiv:2507.01281) closely enough to write the distinguishing
   paragraph (§3).
4. Optional: exhaustive TPLP / AIJ / JAR search. §4 explains why the framing
   survives without it.

**Done:** SG-DT §3–§4 read against the source; NormBench 37% verified; KR
2024–2025 and IJCAI-ECAI 2026 KRR swept; all nine original citations plus three
KR citations verified against their sources.

# WP0 — Prior-art review and positioning memo

**Status: template. Weeks 1–3. GATING — this can reshape the paper.**
**Owners: joint (A takes the RAG / conflict-aware line, B takes the defeasible / legal-NLP line).**

---

## What this memo has to decide

One question, stated so it can come back "no":

> Is this the first work to connect the conditional-conflict problem to the
> **conflict-aware RAG resolution operator specifically** — replacing selection
> with composition at the point where credibility ranking currently discards
> the exception?

Not "is this the first work to notice exceptions get dropped." Chen et al. got
there first, in a different setting, and the memo should say so plainly.

**If the bridging claim is anticipated**, the fallback is already agreed: anchor
the contribution on the extraction method, the formal four-way scope-relation
definition, and the validated metric suite. Write that down here if it happens,
rather than quietly softening the abstract later.

---

## Three confirmed starting points

These are not search terms. They are papers already identified as close
relatives, to be read and positioned against directly.

### 1. Span-Grounded Deontic Trees / NormBench — the closest work overall
Chen et al., *"From Statute to Control Flow: Span-Grounded Deontic Trees for
Defeasible Scope Parsing"*, arXiv:2606.08932, KDD 2026.

- Names **Silent Scope Omission (SSO)** — a model applying a general rule while
  silently dropping nested exceptions. That is this project's "suppression",
  under a different name, found independently.
- NormBench: 2,290 provisions, Chinese and English statutory/policy text.
- SG-DT anchors every logical branch to a source span and requires explicit
  exclusion guards — conceptually close to this project's grounding gate.
- **Already handles the recursive case this project explicitly excludes**: 37%
  of NormBench needs Level-3-or-deeper exception-to-exception resolution, and
  they report "Recursion Decay" as defeater depth increases.

**The distinction to defend:** SG-DT is a *structure-recovery* diagnostic — given
a provision, does a model reconstruct which clause overrides which? It holds
retrieval fixed or oracular. This project targets an earlier point: the
*resolution operator* that decides, for two separately retrieved and
credibility-ranked passages, whether to select between them or compose them. A
perfect SG-DT parser still exhibits this failure if the pipeline discards the
exception passage before the parser ever sees both together.

**Check while reading:** do they anywhere define a resolution operator over
*separately retrieved* competing sources? If yes, the bridging claim is gone and
the fallback applies.

### 2. ConditionalQA — the closest work on extraction
Sun, Cohen, and Salakhutdinov, ACL 2022, arXiv:2110.06884.

- Established "conditional answers" as a recognized hard problem in reading
  comprehension over long policy documents, including unstated conditions.
- Genuinely close to the *extraction* half of this project.
- **Structurally different in a way that matters:** it is a **single-document**
  task. The condition and the outcome it governs are both already in the one
  document the model reads, so there is no upstream selection step that could
  discard the condition first.

**Action:** Contrastive Scope Probing must be benchmarked against
ConditionalQA-style condition-selection baselines during WP1. Not optional.

### 3. Defeasible NLI — the NLP defeasible-reasoning line
Rudinger et al., EMNLP Findings 2020.

- Formalises defeasibility as a *graded NLI judgement over sentence pairs* —
  how a new premise strengthens or weakens a hypothesis's plausibility.
- Not an extracted, source-attributed, composable branch structure grounded in
  retrieved documents.
- Cited by name specifically so WP0 checks it directly, rather than gesturing at
  "the defeasible reasoning literature" and moving on.

---

## Venues a general web search does not reach

The arXiv-and-web sweep behind the proposal cannot see these. Search each
directly:

- [ ] **KR** (Principles of Knowledge Representation and Reasoning)
- [ ] **IJCAI**
- [ ] **TPLP** (Theory and Practice of Logic Programming)
- [ ] **AAAI**
- [ ] **Artificial Intelligence Journal**
- [ ] **Journal of Automated Reasoning**
- [ ] **ACL Anthology** — beyond what surfaces via general search

Classical lineage to check while there: default logic (Reiter 1980), defeasible
logic (Pollock, Prakken), truth-maintenance systems.

---

## Adjacent work already surveyed (confirm, do not re-derive)

| Work | Relation to this project |
|---|---|
| ConflictRAG (Wang et al., arXiv:2605.17301) | The direct target. detect-classify-resolve-generate with Entropy-TOPSIS credibility selection. |
| DRAGged into Conflicts (Cattan et al., arXiv:2506.08500) | Five-category taxonomy, CONFLICTS benchmark, three-stage annotation, judge validated at 0.89. **The annotation-rigor precedent this project adopts.** |
| — its "complementary information" category | The entry a reviewer points to first. It covers **symmetric** relations between co-equal answers to an underspecified query. A conditional conflict is **asymmetric**: a default plus a scope-narrowing exception whose condition must be extracted, grounded, and attached. |
| TCR (Ye et al., arXiv:2601.06842) | Makes the detect-resolve process contestable. Orthogonal to which operator is used. |
| Detecting Is Not Resolving (Yu et al., arXiv:2605.27157) | 50,000 turn-level evaluations showing models acknowledge contradictory evidence without letting it constrain output. Independent evidence that detection and safe resolution are dissociated. |
| Claim-Selective Certification (Kan, arXiv:2605.21949) | Structurally close — notes evidence may "require conditions" — but maps claims to a certification action rather than extracting and composing the condition. |
| S2G-RAG (Li et al., ACL 2026) | Structured "gap items" governing what to retrieve next. Same tool (structure), different object. |

---

## Deliverable

A positioning memo that does one of three things, explicitly:

1. **Confirms** the bridging claim → §4 of the paper stands as written.
2. **Narrows** it → rewrite §4 with the narrower claim, and say what was found.
3. **Finds it anticipated** → invoke the fallback (extraction method + formal
   scope-relation definition + validated metric suite), and rewrite §3.

Whichever it is, record it here with citations. A memo that says "we looked and
found nothing" without naming what was searched is not a WP0 deliverable.

---

## Why this is gating

Everything downstream — the benchmark design, which baselines are decisive, how
§4 is framed — depends on the answer. It is cheap in week 2 and very expensive
in week 14, which is the entire reason it sits at the front of the plan.

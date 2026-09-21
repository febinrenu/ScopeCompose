# Baselines

## Member A owns (implemented, `retrieval_side.py`)

| Baseline | What it does | Why it is here |
|---|---|---|
| `standard_rag` | concatenate top-K | the floor: no conflict handling at all |
| `rerank_top1` | keep the single top passage | selection in its crudest form |
| `nli_filter` | drop the passage judged inconsistent | **the demonstrative one** |

The NLI-filter is the one to put on a slide. It is a reasonable-looking design
someone might actually ship, and on a general-rule/valid-exception pair it
reliably deletes the exception — because the exception is exactly the passage
that reads as inconsistent. The system then answers confidently from the
general rule alone, scores well on answer correctness, and has erased the
branch that applied to the user. That is the project's central failure mode,
reproducible in about thirty lines.

## Member B owns (not implemented here)

- credibility-based selection resolution (a faithful reimplementation)
- single-prompt chain-of-thought detection-and-resolution
- full-context concatenation
- taxonomy-aware prompting, adapted from Cattan et al. (arXiv:2506.08500)

## Joint — and the paper's decisive result

**The structured long-context baseline.** A long-context model gets the full
concatenated passage set and is instructed to emit an explicit SG-DT-style
branch/tree output — condition, outcome, applicability, per-branch source span —
rather than a free-text "mention any exceptions" answer. It is scored with the
*identical* PR/SR/HCR/SCR metrics as the main pipeline, so the comparison is
same-metric and direct.

This is not a formality alongside the others. If it matches the composition
pipeline's Preservation Rate, the central architectural claim does not hold as
stated, and that outcome gets reported as prominently as a positive one.

**Read README section 5 before interpreting it.** At zero frontier spend this
runs on open weights, which makes the result interpretable in only one
direction: if the baseline wins, that is decisive falsification and fully
valid; if it loses, a reviewer will attribute the gap to model tier rather than
architecture. Budget the strongest available model here, and say so in the
write-up.

A weaker free-text variant ("state any conditions or exceptions that qualify
the answer") is retained as a secondary check on how much of any observed
effect depends on requiring structured output at all, separately from the
pipeline architecture.

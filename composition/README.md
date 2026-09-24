# `composition/`

## What goes here

**B2: the composition operator**, with nested / crossed multi-exception flagging.

```
1. classify each (default, exception) pair by the four-way relation   [comes from A4]
2. if any pair is 'opposed': fall back to Selection, report as such
3. merge any 'redundant' pair into the default (NOT a separate branch)
4. for every remaining pair of exception branches (b_i, b_j):
       if their applicability sets overlap:
           if one is a subset of the other -> flag 'nested'    (exception to an exception)
           elif they disagree on the overlap -> flag 'crossed' (co-occurring, conflicting)
           else -> merge (outcome-compatible overlap, not a flag)
5. if 'nested' or 'crossed' was flagged: fall back to Selection with the flag recorded
   (this populates multi_exception_flags in the contract)
6. else: compose -- the default restricted to the region no exception covers,
   plus every exception branch, each attributed to its supporting passage
```

## Why nested and crossed are two flags, not one

They say different things about the data, and the paper reports both rates
separately:

- **nested** is exception-to-exception structure — the recursive case that
  Span-Grounded Deontic Trees (Chen et al., KDD 2026) already handles in the
  single-document setting, where 37% of their benchmark needs Level-3-or-deeper
  resolution.
- **crossed** is two independent exceptions that can co-occur and disagree — a
  structurally different pattern that a recursive-tree framing does not directly
  name.

Collapsing them into one "complex" flag throws away the distinction that makes
the future-work argument concrete.

## Implementation

`operator.py` implements this. `CompositionOperator.compose()` runs steps 1-6;
`classify_exception_pair()` is step 4 and is exported separately because it is
the part worth testing directly.

## What this builds on

- `contract/routing.py` — `route()` and `route_with_flags()` implement steps 1–2
  and 5 already, with the reasoning attached to each decision. Use them rather
  than re-deriving the table.
- `scope/attributes.py` — `compare()`, `intersects()` and `narrower()` do set
  algebra over typed applicability attributes. Step 4's subset and overlap tests
  are already written; you need the outcome-agreement half.
- `contract/gold.py` — `Branch`, `Applicability`, `ScopeAttribute`.
- `python -m contract.mock` generates records including deliberately nested and
  crossed cases, so B2's flagging logic has test data before real annotation.

## Metric

Correctness of branch-structure assembly, plus the nested-flagged and
crossed-flagged rates — reported separately, on both the natural corpus and the
synthetic stress test.

## Scope discipline

This project handles **first-order** conditional conflicts only: a default and
its direct exceptions. Nested and crossed structures are detected and flagged,
never composed recursively and never silently picked between. Recursive
composition via an SG-DT-style tree is named future work. Flagging honestly is
the deliverable; getting it right on second-order structure is not.

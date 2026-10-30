# Datasheet: generated sections

Every number below is computed from the data by `python -m benchmark.datasheet_v7`.

## Composition by tier

  | Tier | Definition | Count |
  |---|---|---|
  | Tier 1 `natural` | rule and exception in separately retrievable documents | **59** |
  | Tier 2 `split` | single-document pair split across two synthetic documents | **54** |
  | Same-guide | two URLs of one underlying guide; real retrieval separation, no authored split | **15** |
  | **Total** | | **128** |

  Of these, **67** are conditional and **61** are distractors.

## Strata by split

```
Strata x split
====================================================================
  stratum      train     dev    test    total
  -----------------------------------------
  S1              12       1       1       14
  S2               4       0       0        4
  S3               0       0       0        0
  S4               0       0       0        0
  S5              22      15      17       54
  control         39      10       7       56

  S1 conditional conflicts in test: 1  (target 80, floor 60)
  status: BELOW FLOOR: narrowing rule applies

  Narrowing rule (proposal section 3.1): every cross-document hypothesis (H1-H3)
  is reported as EXPLORATORY and the title/abstract claim narrows to
  'exception suppression in RAG, with exploratory natural cross-document
  evidence'. S3 never substitutes for S1.

  WARNING: 17 S5 probe(s) share a document group with a protected stratum and landed in test; they are excluded from every test population by stratum: ['wp1_fin_imp_007', 'wp1_fin_imp_008', 'wp1_fin_imp_009', 'wp1_fin_imp_010']
```

## Label provenance

- annotated by hand (no proposal recorded): **42**
- no branches: **86**

Model proposals are shown as proposals. They count as human-annotated only after a person has reviewed them.

## Annotation: decisions, hours and agreement

No review decisions recorded yet.

## Release

- instances: **128**; passages: **256**; withheld (not redistributable): **48**
- open passages (Open Government Licence v3.0): **100**
- open passages (authored for this benchmark): **108**
- unrestorable after redaction: **6**

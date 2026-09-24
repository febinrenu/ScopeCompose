"""Annotation tooling: blind passes, Cohen's kappa per axis, gold assembly.

The protocol this implements is in ``benchmark/manual/annotation_manual.md``.
Two properties are enforced in code rather than by convention, because both are
unrecoverable once violated:

* **Passes are blind.** An annotator can read only their own labels while
  labelling (:mod:`.store`).
* **Kappa needs two independent humans.** It is refused, with an explanation,
  when the two label sets could not have been independent -- the same person
  twice, or anything model-generated (:mod:`.agreement`).
"""

"""B3: conflict-aware generation.

Verbalises a resolved branch structure into a scoped answer with per-branch
attribution. The model explains the structure; it never decides which branch is
true -- that was settled upstream by A4's scope relation and B2's composition.

A generator given room to adjudicate will adjudicate, and the resulting failure
is invisible to answer-correctness scoring: the same blind spot this project is
about, reappearing at the last step. So the default renderer uses no model at
all, and every answer is checked afterwards for dropped branches, invented
figures, and credibility language.
"""

from generation.scoped_answer import (
    FaithfulnessReport,
    GeneratedAnswer,
    GenerationStats,
    ScopedAnswerGenerator,
    check_faithfulness,
    render_template,
)

__all__ = [
    "FaithfulnessReport",
    "GeneratedAnswer",
    "GenerationStats",
    "ScopedAnswerGenerator",
    "check_faithfulness",
    "render_template",
]

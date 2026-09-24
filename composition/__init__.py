"""B2: the composition operator.

The step that replaces selection. Prior conflict-aware RAG reaches this point,
ranks the sources and emits the winner; this assembles a structure keeping
every branch that is true, each attached to the scope it is true under.

Second-order structure -- an exception carved back by a further exception
(``nested``), or two exceptions that co-occur and disagree (``crossed``) -- is
outside this project's first-order scope. It is detected and flagged, never
composed recursively and never silently picked between. The two flags are
reported separately because they say different things about the data.
"""

from composition.operator import (
    ComposedAnswer,
    CompositionOperator,
    CompositionStats,
    Resolution,
    classify_exception_pair,
)

__all__ = [
    "ComposedAnswer",
    "CompositionOperator",
    "CompositionStats",
    "Resolution",
    "classify_exception_pair",
]

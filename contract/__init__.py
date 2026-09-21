"""The frozen interface contract between Member A and Member B.

See ``contract/models.py`` for the wire format and ``contract/gold.py`` for the
annotation format. Changing either requires bumping ``CONTRACT_VERSION``,
updating ``contract/mock.py``, and regenerating the JSON Schema -- all in the
same commit. See ``CLAUDE.md``.
"""

from contract.models import (
    CONTRACT_VERSION,
    ConflictPair,
    ConflictType,
    Construction,
    ContractVersionError,
    DecidedBy,
    Domain,
    MultiExceptionFlags,
    Passage,
    QueryRecord,
    ScopeRelation,
    SourceType,
    check_version,
)

__all__ = [
    "CONTRACT_VERSION",
    "ConflictPair",
    "ConflictType",
    "Construction",
    "ContractVersionError",
    "DecidedBy",
    "Domain",
    "MultiExceptionFlags",
    "Passage",
    "QueryRecord",
    "ScopeRelation",
    "SourceType",
    "check_version",
]

"""ScopeCompose Studio: the pipeline as a local service.

Runs a query end to end (retrieval -> detection -> scope -> extraction ->
composition -> generation -> verification) and streams every step as an
event, so a run can be watched live, stored, replayed, and called as an API.

    python -m service            # http://localhost:8008, docs at /docs
"""

"""CLI for the API cost log.

A thin entry point kept separate from ``costlog.py`` so that
``python -m api_budget.report`` does not trigger Python's double-import
warning -- ``api_budget/__init__.py`` imports the library module, and running
that same module with ``-m`` makes runpy complain.

Usage::

    python -m api_budget.report
    python -m api_budget.report --json
"""

from __future__ import annotations

from api_budget.costlog import main

if __name__ == "__main__":
    raise SystemExit(main())

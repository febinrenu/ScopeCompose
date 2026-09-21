"""Live end-to-end check against the configured provider.

Run after setting up `.env` or changing `config/models.yaml`. It costs a
handful of calls and verifies the things that silently break a corpus run:

1. each tier resolves and returns non-empty content
2. structured JSON output parses (A4 and B1 both depend on it)
3. truncation raises instead of returning an empty string
4. the cache serves an identical second call for free

Usage::

    python scripts/smoke_test.py               # BULK only, cheapest
    python scripts/smoke_test.py --all-tiers   # every configured tier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import settings  # noqa: E402

PASSAGE = "The international transaction fee is waived for premium-tier cardholders."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-tiers", action="store_true",
                    help="exercise every configured tier, not just BULK")
    args = ap.parse_args(argv)

    settings.reload()
    from api_budget import LLMClient, Tier, TierNotConfigured
    from api_budget.backends import TruncatedResponseError
    from api_budget.cache import RequestCache
    from api_budget.costlog import CostLog

    client = LLMClient(
        cache=RequestCache("api_budget/smoke.sqlite"),
        cost_log=CostLog("logs/smoke.jsonl"),
    )
    failures: list[str] = []

    # 1. content ------------------------------------------------------------- #
    tiers = list(Tier) if args.all_tiers else [Tier.BULK]
    print("1. Tier reachability and non-empty content")
    print("-" * 72)
    for tier in tiers:
        try:
            r = client.complete(
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                tier=tier, step="smoke_content",
            )
        except TierNotConfigured as exc:
            print(f"  {tier.value:<17} SKIP  {exc}".rstrip())
            continue
        except Exception as exc:
            print(f"  {tier.value:<17} FAIL  {type(exc).__name__}: {str(exc)[:70]}")
            failures.append(f"{tier.value} unreachable")
            continue

        ok = bool(r.text.strip())
        print(f"  {tier.value:<17} {'OK  ' if ok else 'FAIL'}  {r.model:<24} "
              f"{r.prompt_tokens}/{r.completion_tokens} tok  text={r.text.strip()[:24]!r}")
        if not ok:
            failures.append(f"{tier.value} returned empty content")

    # 2. structured JSON ------------------------------------------------------ #
    print("\n2. Structured JSON output (A4 descriptor extraction depends on this)")
    print("-" * 72)
    prompt = (
        f"Passage: {PASSAGE}\n\n"
        'Return a JSON object with keys "descriptor", "is_default", "outcome".'
    )
    try:
        r = client.complete(
            messages=[{"role": "user", "content": prompt}],
            tier=Tier.BULK, step="smoke_json",
            response_format={"type": "json_object"},
        )
        parsed = r.json()
        print(f"  OK    parsed {len(parsed)} key(s): {sorted(parsed)[:5]}")
        print(f"        {str(parsed)[:120]}")
    except Exception as exc:
        print(f"  FAIL  {type(exc).__name__}: {str(exc)[:110]}")
        failures.append("structured JSON output")

    # 3. truncation ----------------------------------------------------------- #
    print("\n3. Truncation fails loudly rather than returning an empty string")
    print("-" * 72)
    try:
        client.complete(
            messages=[{"role": "user", "content": "Explain retrieval-augmented generation."}],
            tier=Tier.BULK, step="smoke_truncation",
            max_tokens=16, use_cache=False,
        )
        print("  FAIL  no error raised - an empty response would reach the parser")
        failures.append("truncation not detected")
    except TruncatedResponseError as exc:
        print(f"  OK    TruncatedResponseError: {str(exc)[:100]}")
    except Exception as exc:
        print(f"  FAIL  wrong exception type {type(exc).__name__}: {str(exc)[:80]}")
        failures.append("truncation raised the wrong error")

    # 4. cache ---------------------------------------------------------------- #
    print("\n4. Cache serves an identical second call for free")
    print("-" * 72)
    msg = [{"role": "user", "content": "Reply with exactly: cached"}]
    a = client.complete(messages=msg, tier=Tier.BULK, step="smoke_cache")
    b = client.complete(messages=msg, tier=Tier.BULK, step="smoke_cache")
    if not a.cached and b.cached and a.text == b.text:
        print(f"  OK    first call live ({a.latency_s:.2f}s), "
              f"second from cache ({b.latency_s:.2f}s)")
    else:
        print(f"  FAIL  cached flags were {a.cached} then {b.cached}")
        failures.append("cache")

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("All checks passed. The provider is wired up correctly.")
    print("\nRun `python -m api_budget.report` to see what these calls cost,")
    print("and `python scripts/budget_estimate.py` before any full-corpus run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

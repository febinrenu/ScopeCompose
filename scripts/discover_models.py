"""Populate ``config/models.yaml`` from the provider's live model list.

Model catalogues change. Hardcoding an ID means the pipeline breaks silently
months later when that model is retired, usually mid-experiment. This script
asks the provider what it actually serves and writes the tier map from the
answer.

Run once at setup, and again whenever a tier stops resolving::

    python scripts/discover_models.py
    python scripts/discover_models.py --dry-run     # show, don't write
    python scripts/discover_models.py --show        # just list what's available

Tier assignment is heuristic, and deliberately visible rather than clever: it
prints what it picked and why, so a wrong guess is obvious and can be
overridden by hand. Hand edits survive -- ``--force`` is required to overwrite
a tier that already has a model.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

import settings  # noqa: E402

# Substrings that suggest a model's size class. Heuristic, and printed so a bad
# guess is visible rather than silent.
_SMALL_HINTS = ("8b", "7b", "9b", "mini", "small", "instant", "lite", "flash", "haiku")
_LARGE_HINTS = ("70b", "120b", "405b", "large", "maverick", "opus", "sonnet", "pro", "k2")
_REASONING_HINTS = ("r1", "reason", "think", "qwq", "o1", "o3")
_SKIP_HINTS = ("whisper", "tts", "guard", "embed", "rerank", "vision", "moderation")

_PARAM_RE = re.compile(r"(\d+)\s*b\b", re.IGNORECASE)


def _param_billions(model_id: str) -> int:
    m = _PARAM_RE.search(model_id)
    return int(m.group(1)) if m else 0


def _family(model_id: str) -> str:
    """Coarse family name, used to keep SECOND_BACKBONE genuinely different."""
    lowered = model_id.lower()
    for fam in ("llama", "qwen", "deepseek", "mixtral", "mistral", "gemma",
                "kimi", "gpt-oss", "compound", "allam", "gpt"):
        if fam in lowered:
            return fam
    return lowered.split("-")[0] if "-" in lowered else lowered


def list_models(provider: str = "groq") -> list[str]:
    spec = settings.models().provider(provider)
    if not spec.configured:
        raise SystemExit(
            f"provider {provider!r} is not configured.\n"
            f"  base_url: {spec.base_url or '(unset)'}\n"
            f"  api_key:  {'set' if spec.api_key else '(unset)'}\n"
            f"Copy .env.example to .env and fill in your key."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")

    client = OpenAI(base_url=spec.base_url, api_key=spec.api_key, timeout=30.0)
    return sorted(m.id for m in client.models.list().data)


def classify(model_ids: list[str]) -> dict[str, list[str]]:
    """Bucket models into rough capability classes."""
    usable = [m for m in model_ids if not any(h in m.lower() for h in _SKIP_HINTS)]

    buckets: dict[str, list[str]] = {"small": [], "large": [], "reasoning": [], "other": []}
    for m in usable:
        low = m.lower()
        if any(h in low for h in _REASONING_HINTS):
            buckets["reasoning"].append(m)
        elif any(h in low for h in _LARGE_HINTS) or _param_billions(m) >= 30:
            buckets["large"].append(m)
        elif any(h in low for h in _SMALL_HINTS):
            buckets["small"].append(m)
        else:
            buckets["other"].append(m)

    buckets["large"].sort(key=lambda m: (-_param_billions(m), m))
    buckets["small"].sort(key=lambda m: (_param_billions(m) or 99, m))
    return buckets


def choose(buckets: dict[str, list[str]]) -> dict[str, tuple[str | None, str]]:
    """Pick a model per tier. Returns {tier: (model_id, rationale)}."""
    small = buckets["small"] or buckets["other"] or buckets["large"]
    large = buckets["large"] or buckets["other"] or buckets["small"]

    picks: dict[str, tuple[str | None, str]] = {}

    picks["BULK"] = (
        small[0] if small else None,
        "smallest usable model: high volume, low difficulty, wants throughput over quality",
    )
    picks["JUDGE"] = (
        large[0] if large else None,
        "largest available: load-bearing, and validated against human labels before trusted",
    )
    picks["LONG_CONTEXT"] = (
        large[0] if large else None,
        "largest available. NOTE: at K=5 the input is only ~2-5K tokens, so capability -- "
        "not window size -- is what this tier is short of. See README section 5.",
    )

    # SECOND_BACKBONE must be a DIFFERENT family, or the robustness ablation
    # shows nothing.
    judge_model = picks["JUDGE"][0]
    judge_family = _family(judge_model) if judge_model else ""
    alt = next((m for m in large if _family(m) != judge_family), None)
    if alt is None:
        alt = next(
            (m for m in buckets["reasoning"] + buckets["other"] if _family(m) != judge_family),
            None,
        )
    picks["SECOND_BACKBONE"] = (
        alt,
        f"a different family from JUDGE ({judge_family or 'unknown'}) -- same-family would "
        "make the robustness ablation vacuous",
    )
    return picks


def apply(picks: dict[str, tuple[str | None, str]], *, force: bool) -> tuple[dict, list[str]]:
    path = settings.MODELS_YAML
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    changes: list[str] = []

    for tier, (model, _why) in picks.items():
        if model is None:
            continue
        entry = raw["tiers"].get(tier)
        if entry is None:
            continue
        existing = entry.get("model")
        if existing and not force:
            changes.append(f"  {tier:<16} kept {existing}  (hand-set; use --force to replace)")
            continue
        if existing == model:
            changes.append(f"  {tier:<16} unchanged {model}")
            continue
        entry["model"] = model
        changes.append(f"  {tier:<16} {existing or '(unset)'} -> {model}")

    return raw, changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, write nothing")
    ap.add_argument("--show", action="store_true", help="list available models and exit")
    ap.add_argument("--force", action="store_true", help="overwrite tiers that already have a model")
    args = ap.parse_args(argv)

    print(f"querying {args.provider} for available models ...")
    model_ids = list_models(args.provider)
    print(f"  {len(model_ids)} model(s) returned\n")

    buckets = classify(model_ids)
    for name in ("large", "small", "reasoning", "other"):
        if buckets[name]:
            print(f"  {name:<10} {', '.join(buckets[name])}")
    print()

    if args.show:
        return 0

    picks = choose(buckets)
    print("tier assignment:")
    for tier, (model, why) in picks.items():
        print(f"  {tier:<16} -> {model or '(none found)'}")
        print(f"  {'':<16}    {why}")
    print()

    unresolved = [t for t, (m, _) in picks.items() if m is None]
    if unresolved:
        print(f"WARNING: no model found for {', '.join(unresolved)} -- "
              f"set these by hand in {settings.MODELS_YAML.name}\n")

    raw, changes = apply(picks, force=args.force)
    print("changes:")
    print("\n".join(changes) if changes else "  (none)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    settings.MODELS_YAML.write_text(
        yaml.safe_dump(raw, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nwrote {settings.MODELS_YAML}")
    print("NOTE: this rewrites the file through the YAML serialiser, so the "
          "explanatory comments in the original are lost. They are in git.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())

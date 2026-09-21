"""Validate a JSONL file against the contract.

The integration check both members run before handing anything over::

    python -m contract.validate mock.jsonl
    python -m contract.validate data.jsonl --format gold --strict-version

Reports every failure with its line number rather than stopping at the first,
because when a schema change breaks a file it usually breaks it in one
systematic way, and seeing all of them at once identifies the cause faster than
fixing them one at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from contract.gold import GoldInstance
from contract.models import (
    CONTRACT_VERSION,
    ConflictType,
    Construction,
    ContractVersionError,
    QueryRecord,
    ScopeRelation,
    check_version,
)


@dataclass
class Report:
    path: str
    total: int = 0
    valid: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)
    conflict_types: dict[str, int] = field(default_factory=dict)
    scope_relations: dict[str, int] = field(default_factory=dict)
    constructions: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"{self.path}: {self.valid}/{self.total} valid"]

        if self.errors:
            lines.append("")
            lines.append(f"{len(self.errors)} error(s):")
            for lineno, msg in self.errors[:50]:
                first = msg.strip().splitlines()[0] if msg.strip() else msg
                lines.append(f"  line {lineno}: {first}")
            if len(self.errors) > 50:
                lines.append(f"  ... and {len(self.errors) - 50} more")

        if self.valid:
            lines.append("")
            lines.append("distribution:")
            for label, counts in (
                ("conflict_type", self.conflict_types),
                ("scope_relation", self.scope_relations),
                ("construction", self.constructions),
            ):
                body = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
                lines.append(f"  {label:16s} {body}")

            # Tier blending is a reporting error, not a schema error, so it can
            # never be caught by validation alone -- flag it here instead.
            if len(self.constructions) > 1:
                lines.append("")
                lines.append(
                    "  NOTE: this file mixes Tier 1 (natural) and Tier 2 (split) instances. "
                    "They must be reported separately, never blended into one headline number."
                )

            missing = {r.value for r in ScopeRelation} - set(self.scope_relations)
            if missing and self.scope_relations:
                lines.append(
                    f"  NOTE: no instances with scope_relation in {sorted(missing)} -- "
                    "a fixture missing a relation cannot catch confusions involving it."
                )

        return "\n".join(lines)


def validate_file(
    path: Path,
    *,
    fmt: str = "record",
    strict_version: bool = False,
) -> Report:
    model = QueryRecord if fmt == "record" else GoldInstance
    rep = Report(path=str(path))

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            rep.total += 1

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                rep.errors.append((lineno, f"malformed JSON: {exc}"))
                continue

            try:
                inst = model.model_validate(obj)
            except ValidationError as exc:
                rep.errors.append((lineno, str(exc)))
                continue

            try:
                check_version(inst.contract_version, strict=strict_version)
            except ContractVersionError as exc:
                rep.errors.append((lineno, str(exc)))
                continue

            rep.valid += 1

            if isinstance(inst, QueryRecord):
                for pair in inst.conflict_pairs:
                    rep.conflict_types[pair.type.value] = rep.conflict_types.get(pair.type.value, 0) + 1
                    if pair.scope_relation:
                        k = pair.scope_relation.value
                        rep.scope_relations[k] = rep.scope_relations.get(k, 0) + 1
                c = inst.construction.value
            else:
                k = inst.gold_conflict_type.value
                rep.conflict_types[k] = rep.conflict_types.get(k, 0) + 1
                if inst.gold_scope_relation:
                    s = inst.gold_scope_relation.value
                    rep.scope_relations[s] = rep.scope_relations.get(s, 0) + 1
                c = inst.construction.value
            rep.constructions[c] = rep.constructions.get(c, 0) + 1

    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="JSONL file(s) to validate")
    ap.add_argument("--format", choices=["record", "gold"], default="record",
                    help="'record' = wire contract (default); 'gold' = annotation format")
    ap.add_argument("--strict-version", action="store_true",
                    help=f"require an exact contract_version match with v{CONTRACT_VERSION}")
    args = ap.parse_args(argv)

    failed = False
    for path in args.paths:
        if not path.exists():
            print(f"{path}: not found", file=sys.stderr)
            failed = True
            continue
        rep = validate_file(path, fmt=args.format, strict_version=args.strict_version)
        print(rep.render())
        if not rep.ok:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export the contract as JSON Schema.

Run after ANY change to ``contract/models.py``::

    python -m contract.export_schema

The exported file is committed. That makes a schema change show up as a diff a
human can review -- including Member B, who should not have to read Python to
check what changed in the agreement they build against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract.gold import GoldInstance
from contract.models import CONTRACT_VERSION, QueryRecord

SCHEMA_DIR = Path(__file__).parent / "schema"


def export(out_dir: Path = SCHEMA_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for name, model in (("query_record", QueryRecord), ("gold_instance", GoldInstance)):
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["title"] = f"conflict-rag {name} v{CONTRACT_VERSION}"
        path = out_dir / f"{name}-v{CONTRACT_VERSION}.json"
        path.write_text(
            json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out-dir", type=Path, default=SCHEMA_DIR)
    args = ap.parse_args(argv)
    for p in export(args.out_dir):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

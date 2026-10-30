"""Record real runs for the static site.

    python -m service.record            # offline runs only (no keys needed)
    python -m service.record --live     # also a live run (models + LLM)

Runs the pipeline exactly as the server does and writes every event to
``site/public/data/runs/<name>.json`` with an ``index.json`` beside it. The
studio plays these back when no engine server is running -- so the published
site shows genuine traces, labelled as recordings, never simulated ones.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from service import tracing_client
from service.events import Tracer
from service.orchestrator import RunRequest, run_query

OUT = Path(__file__).resolve().parent.parent / "site" / "public" / "data" / "runs"

RECIPES = [
    ("fee-override", "Monthly fee: the detector misses, override continues",
     "offline · gold labels after a detector miss",
     dict(source="benchmark", record_id="wp1_fin_imp_003", mode="offline", detector_policy="override",
          reader={"scheme": "graduate", "years_since_registration": 1})),
    ("fee-as-run", "Monthly fee, as-run: the branch is silently dropped",
     "offline · the failure the project measures",
     dict(source="benchmark", record_id="wp1_fin_imp_003", mode="offline", detector_policy="as_run",
          reader={"scheme": "graduate", "years_since_registration": 1})),
    ("card-passages", "Your own passages: a card fee and its carve-out",
     "offline · pasted passages",
     dict(source="passages", query="Do I pay a fee on card payments abroad?", mode="offline",
          detector_policy="override", passages=[
              {"text": "A non-sterling transaction fee of 2.99% applies to all card payments made in a foreign currency.",
               "source_type": "official_policy"},
              {"text": "Premium Plus cardholders pay no non-sterling transaction fee on card payments abroad.",
               "source_type": "product_terms"}])),
]

LIVE = ("fee-live", "Monthly fee, live: DeBERTa, Contriever and the LLM stages",
        "live · real models, every LLM call shown",
        dict(source="benchmark", record_id="wp1_fin_imp_003", mode="live", detector_policy="override",
             generation="llm", reader={"scheme": "graduate", "years_since_registration": 1}))


def record(name: str, title: str, note: str, req: dict) -> dict:
    tracer = Tracer(f"recorded_{name}")
    tracer.emit("run", "start", "run", {"kind": "run", "request": req})
    summary = run_query(RunRequest.from_dict(req), tracer)
    tracer.emit("run", "end", "done", {"summary": summary})
    body = {"name": name, "title": title, "note": note, "recorded": time.strftime("%Y-%m-%d"),
            "events": [e.to_dict() for e in tracer.events]}
    (OUT / f"{name}.json").write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {name}: {len(tracer.events)} events, {tracer.llm['calls']} LLM call(s), {summary.get('resolution')}")
    # what actually happened goes on the label, so a recording never promises an outcome it did not have
    outcome = " · ".join(x for x in (summary.get("relation"), summary.get("resolution", "").replace("_", " ")) if x)
    return {"id": name, "title": title, "note": f"{note} · {outcome}" if outcome else note,
            "file": f"{name}.json", "mode": req["mode"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="also record a live run (needs the models and an LLM key)")
    args = ap.parse_args(argv)
    tracing_client.install()
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    if args.live:
        index.append(record(*LIVE))
    else:
        old = OUT / "index.json"
        if old.exists():  # keep a live recording made earlier
            index += [r for r in json.loads(old.read_text(encoding="utf-8")) if r.get("mode") == "live"]
    for recipe in RECIPES:
        index.append(record(*recipe))
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(index)} recording(s) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

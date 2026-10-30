"""Run the ScopeCompose engine server.

    python -m service                  # http://127.0.0.1:8008
    python -m service --lan            # also reachable from phones on the same Wi-Fi
    python -m service --warm-live      # load the live models at start-up

API docs at /docs. The studio (the site's #/studio page) finds the server on
port 8008 by default.
"""

from __future__ import annotations

import argparse
import threading


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=None, help="bind address (default 127.0.0.1; --lan for 0.0.0.0)")
    ap.add_argument("--lan", action="store_true", help="listen on the local network too")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--warm-live", action="store_true", help="load DeBERTa / Contriever at start-up")
    args = ap.parse_args(argv)

    import uvicorn

    from service import orchestrator, tracing_client
    from service.app import create_app

    tracing_client.install()
    app = create_app()

    def warm() -> None:
        orchestrator.benchmark()
        orchestrator.engines(False)
        if args.warm_live:
            e = orchestrator.engines(True)
            recs = list(orchestrator.benchmark().values())[:1]
            if recs:
                e.stage1.score_pairs(recs[0].passages[:2])  # loads the NLI model
            orchestrator.benchmark_retriever(True).dense  # loads Contriever and embeds the corpus

    threading.Thread(target=warm, daemon=True, name="warm").start()
    host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
    print(f"ScopeCompose engine server on http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{args.port}"
          f"  (docs: /docs)", flush=True)

    import asyncio

    async def serve() -> None:
        # A browser closing an event stream resets the socket; on Windows the
        # proactor loop reports that as an error. It is not one.
        loop = asyncio.get_running_loop()
        default = loop.get_exception_handler()

        def quiet(lp, ctx):
            if isinstance(ctx.get("exception"), (ConnectionResetError, ConnectionAbortedError)):
                return
            (default or lp.default_exception_handler)(ctx) if default else lp.default_exception_handler(ctx)

        loop.set_exception_handler(quiet)
        await uvicorn.Server(uvicorn.Config(app, host=host, port=args.port, log_level="warning")).serve()

    asyncio.run(serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# ==========================================================================
#  ScopeCompose - one-command launcher for macOS and Linux
#  (start.bat is the Windows twin; they do the same things).
#
#    ./start.sh          build the site and open it   (http://localhost:4173, and on your network)
#    ./start.sh studio   the site + the engine server, opened on the live Studio (#/studio)
#    ./start.sh dev      live-reloading dev server    (http://localhost:5173, and on your network)
#    ./start.sh verify   check the browser engines against the recorded Python
#
#  Needs only Node.js 20.19 or newer. Python is NOT needed to run the site.
#  SC_NO_OPEN=1 stops it opening a browser; SC_LOCAL_ONLY=1 keeps the server
#  on this computer only (for example on public Wi-Fi).
# ==========================================================================
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
SITE="$ROOT/site"
MODE="${1:-run}"

fail() { echo "[x] $*"; exit 1; }

[ -f "$SITE/package.json" ] || fail "Could not find $SITE/package.json. Keep start.sh next to the \"site\" folder."

# ---- 1. Node.js ---------------------------------------------------------------
if ! command -v node >/dev/null 2>&1; then
  echo "[x] Node.js is not installed (or not on PATH)."
  if [ "$(uname)" = "Darwin" ]; then
    echo "    Install the LTS from https://nodejs.org (or: brew install node), then run ./start.sh again."
  else
    echo "    Install Node.js 20.19+ from https://nodejs.org or your package manager"
    echo "    (for example: https://github.com/nodesource/distributions), then run ./start.sh again."
  fi
  exit 1
fi
node -e "const [a,b]=process.versions.node.split('.').map(Number);process.exit(a>20||(a===20&&b>=19)?0:1)" \
  || fail "Node.js $(node -v) is too old. ScopeCompose needs 20.19 or newer: https://nodejs.org"
echo "[ok] Node.js $(node -v)"

cd "$SITE" || fail "Could not enter $SITE"

# ---- 2. Dependencies (first run only, or after package-lock changes) ----------------
needs_install=1
if [ -f node_modules/vite/bin/vite.js ]; then
  node -e "const f=require('fs');try{process.exit(f.statSync('package-lock.json').mtimeMs>f.statSync('node_modules/.package-lock.json').mtimeMs+1000?1:0)}catch{process.exit(1)}" && needs_install=0
fi
if [ "$needs_install" = 1 ]; then
  echo "[..] Installing dependencies (first run takes a minute)..."
  if ! npm ci --no-audit --no-fund; then
    echo "[!] npm ci failed - retrying with npm install..."
    npm install --no-audit --no-fund || fail "Could not install dependencies. Check your internet connection and try again."
  fi
  echo "[ok] Dependencies installed"
else
  echo "[ok] Dependencies already installed"
fi

OPEN="--open"; [ "${SC_NO_OPEN:-}" = "1" ] && OPEN=""
HOST="0.0.0.0"; [ "${SC_LOCAL_ONLY:-}" = "1" ] && HOST="localhost"

# A free port, so the links printed are the ones that work.
pick_port() { PORT="$(node tools/free_port.mjs "$1" "$HOST")" || fail "Could not find a free port near $1. Close other servers and try again."; }

links() {
  echo
  echo " ==================================================================="
  echo "  ScopeCompose"
  echo
  echo "  On this computer:   http://localhost:$1"
  if [ "$HOST" != "localhost" ]; then
    echo
    echo "  On your phone or tablet (same Wi-Fi as this computer):"
    node tools/lan_links.mjs "$1"
    echo
    echo "  If the phone cannot connect:"
    if [ "$(uname)" = "Darwin" ]; then
      echo "    - macOS may ask whether node may accept incoming connections: choose Allow."
    else
      echo "    - If a firewall is on (ufw, firewalld), allow port $1 for the local network."
    fi
    echo "    - Check that the phone is on the same Wi-Fi, not on mobile data."
    echo "    - Some office or college networks stop devices seeing each other;"
    echo "      a phone hotspot that this computer joins works instead."
  fi
  echo
  echo "  Keep this window open while you use the site. Ctrl+C stops it."
  echo " ==================================================================="
  echo
}

case "$MODE" in
  run)
    echo "[..] Checking the browser engines against the recorded Python answers..."
    if node tools/verify_engines.js >/dev/null; then
      echo "[ok] Engines agree with the Python implementation"
    else
      echo "[!] The engine check reported a disagreement. Run \"./start.sh verify\" for details."
      echo "    Continuing so the site can still be viewed."
    fi
    echo "[..] Building the site..."
    node node_modules/vite/bin/vite.js build --logLevel error || fail "The build failed - see the messages above."
    echo "[ok] Built to site/dist"
    pick_port 4173
    links "$PORT"
    exec node node_modules/vite/bin/vite.js preview --port "$PORT" --strictPort --host "$HOST" $OPEN
    ;;
  dev)
    pick_port 5173
    links "$PORT"
    exec node node_modules/vite/bin/vite.js --port "$PORT" --strictPort --host "$HOST" $OPEN
    ;;
  studio)
    # Python for the engine server: the repo's .venv, created on first use
    PY="$ROOT/.venv/bin/python"
    if [ ! -x "$PY" ]; then
      BASEPY=""
      for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys;sys.exit(0 if sys.version_info>=(3,11) else 1)" 2>/dev/null; then BASEPY="$c"; break; fi
      done
      if [ -z "$BASEPY" ]; then
        echo "[!] Python 3.11+ was not found, so the studio will use the browser's engines only."
        echo "    Install it from https://www.python.org and run ./start.sh studio again for the engine server."
        exec "$0" run
      fi
      echo "[..] Creating a Python environment for the engine server (first run only)..."
      "$BASEPY" -m venv "$ROOT/.venv" || fail "Could not create $ROOT/.venv"
    fi
    if ! "$PY" -c "import fastapi, uvicorn, multipart, rank_bm25, pypdf, yaml, dotenv, numpy, pydantic" 2>/dev/null; then
      echo "[..] Installing the engine server's packages (a few megabytes; first run only)..."
      "$PY" -m pip install --disable-pip-version-check -q -r "$ROOT/requirements-service.txt" || fail "Could not install the engine server's packages."
    fi
    echo "[ok] Python engine environment ready"
    echo "[..] Building the site..."
    node node_modules/vite/bin/vite.js build --logLevel error || fail "The build failed - see the messages above."
    echo "[ok] Built to site/dist"
    LANFLAG=""; [ "$HOST" = "0.0.0.0" ] && LANFLAG="--lan"
    ENGINE_PID=""
    if [ "$(node tools/free_port.mjs 8008 "$HOST")" = "8008" ]; then
      (cd "$ROOT" && "$PY" -m service --port 8008 $LANFLAG) &
      ENGINE_PID=$!
      trap '[ -n "$ENGINE_PID" ] && kill "$ENGINE_PID" 2>/dev/null' EXIT INT TERM
      echo "[..] Engine server starting: http://localhost:8008  (API docs: /docs)"
    else
      echo "[!] Port 8008 is in use: assuming an engine server is already running there."
    fi
    pick_port 4173
    links "$PORT"
    echo "  Studio:   http://localhost:$PORT/#/studio     API docs: http://localhost:8008/docs"
    echo
    [ -n "$OPEN" ] && OPEN="--open /#/studio"
    node node_modules/vite/bin/vite.js preview --port "$PORT" --strictPort --host "$HOST" $OPEN
    ;;
  verify)
    exec node tools/verify_engines.js
    ;;
  *)
    fail "Unknown option \"$MODE\". Use: ./start.sh  |  ./start.sh studio  |  ./start.sh dev  |  ./start.sh verify"
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if pgrep -f "python3 -m demo.app_server" >/dev/null 2>&1; then
  pkill -f "python3 -m demo.app_server" || true
  sleep 1
fi

python3 -m demo.app_server --host "${MEDICAL_DEMO_HOST:-127.0.0.1}" --port "${MEDICAL_DEMO_PORT:-8088}"

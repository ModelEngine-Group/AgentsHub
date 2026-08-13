#!/usr/bin/env bash
# dki 运行入口（Linux/macOS/Git-Bash）
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
PY=".venv/bin/python"; [ -f ".venv/Scripts/python.exe" ] && PY=".venv/Scripts/python.exe"
exec "$PY" -m dki "$@"

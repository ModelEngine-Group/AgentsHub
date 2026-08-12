#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MEDINSIGHT_MCP_HOST="${MEDINSIGHT_MCP_HOST:-0.0.0.0}"
export MEDINSIGHT_MCP_PORT="${MEDINSIGHT_MCP_PORT:-18100}"
export MEDINSIGHT_MCP_TRANSPORT="${MEDINSIGHT_MCP_TRANSPORT:-sse}"

exec python "${SCRIPT_DIR}/server.py"

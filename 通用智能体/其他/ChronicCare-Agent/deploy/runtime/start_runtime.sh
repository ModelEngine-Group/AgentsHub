#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/app"
LOG_DIR="${PROJECT_ROOT}/logs/runtime"
mkdir -p "${LOG_DIR}"

# The Streamlit console entry point places /usr/local/bin at sys.path[0].
# Keep the application root importable so app/streamlit_app.py can resolve
# the sibling tool_server package in the unified runtime image.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CHRONICCARE_TOOL_SERVER_URL="${CHRONICCARE_TOOL_SERVER_URL:-http://127.0.0.1:18088}"
export CHRONICCARE_MCP_HOST="${CHRONICCARE_MCP_HOST:-0.0.0.0}"
export CHRONICCARE_MCP_PORT="${CHRONICCARE_MCP_PORT:-18188}"
export CHRONICCARE_MCP_TRANSPORT="${CHRONICCARE_MCP_TRANSPORT:-streamable-http}"
export CHRONICCARE_TRACE_DIR="${CHRONICCARE_TRACE_DIR:-/app/outputs/mcp_traces}"
export CHRONICCARE_TRACE_FILE="${CHRONICCARE_TRACE_FILE:-/app/outputs/mcp_traces/mcp_tool_calls.jsonl}"
export CHRONICCARE_TRACE_SUMMARY_FILE="${CHRONICCARE_TRACE_SUMMARY_FILE:-/app/outputs/mcp_traces/mcp_trace_summary.json}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,172.17.0.1,172.19.0.1,172.20.0.1,172.21.0.1,10.0.0.0/8,172.16.0.0/12}"
export no_proxy="${no_proxy:-$NO_PROXY}"

TOOL_SERVER_PID=""
MCP_ADAPTER_PID=""
STREAMLIT_PID=""

cleanup() {
  for pid in "${STREAMLIT_PID}" "${MCP_ADAPTER_PID}" "${TOOL_SERVER_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

python3 -m uvicorn tool_server.app:app --host 0.0.0.0 --port 18088 \
  >> "${LOG_DIR}/tool_server.log" 2>&1 &
TOOL_SERVER_PID=$!

python3 scripts/run_mcp_adapter.py \
  >> "${LOG_DIR}/mcp_adapter.log" 2>&1 &
MCP_ADAPTER_PID=$!

streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 18501 \
  >> "${LOG_DIR}/streamlit.log" 2>&1 &
STREAMLIT_PID=$!

wait -n "${TOOL_SERVER_PID}" "${MCP_ADAPTER_PID}" "${STREAMLIT_PID}"
exit $?

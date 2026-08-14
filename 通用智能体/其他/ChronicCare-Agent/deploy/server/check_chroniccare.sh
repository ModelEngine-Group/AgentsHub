#!/usr/bin/env bash
set -euo pipefail

TOOL_SERVER_URL="${CHRONICCARE_TOOL_SERVER_URL:-http://127.0.0.1:18088}"
STREAMLIT_URL="${CHRONICCARE_STREAMLIT_URL:-http://127.0.0.1:18501}"

echo "Checking ${TOOL_SERVER_URL}/health"
if command -v curl >/dev/null 2>&1; then
  curl -fsS "${TOOL_SERVER_URL}/health" || true
  echo
  echo "Checking ${STREAMLIT_URL}"
  curl -I -fsS "${STREAMLIT_URL}" || true
else
  echo "curl not found. Please check these URLs manually:"
  echo "- ${TOOL_SERVER_URL}/health"
  echo "- ${STREAMLIT_URL}"
fi

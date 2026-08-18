#!/usr/bin/env bash
set -euo pipefail

TOOL_SERVER_URL="${CHRONICCARE_TOOL_SERVER_URL:-http://127.0.0.1:18088}"
MCP_BASE_URL="${CHRONICCARE_MCP_BASE_URL:-http://127.0.0.1:18188}"
MCP_ENDPOINT="${MCP_BASE_URL}/mcp"

echo "Checking Tool Server: ${TOOL_SERVER_URL}/health"
curl --noproxy '*' -fsS "${TOOL_SERVER_URL}/health" || true
echo

echo "Checking MCP Adapter root: ${MCP_BASE_URL}"
curl --noproxy '*' -fsS "${MCP_BASE_URL}" || true
echo

echo "Checking MCP tools list: ${MCP_BASE_URL}/tools"
curl --noproxy '*' -fsS "${MCP_BASE_URL}/tools" || true
echo

echo "Checking MCP initialize"
curl --noproxy '*' -fsS "${MCP_ENDPOINT}" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"init-1","method":"initialize","params":{"clientInfo":{"name":"shell-check","version":"0.1.0"}}}' || true
echo

echo "Checking chroniccare_health_check"
curl --noproxy '*' -fsS "${MCP_ENDPOINT}" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"call-1","method":"tools/call","params":{"name":"chroniccare_health_check","arguments":{}}}' || true
echo

echo "Checking chroniccare_kg_summary"
curl --noproxy '*' -fsS "${MCP_ENDPOINT}" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"call-2","method":"tools/call","params":{"name":"chroniccare_kg_summary","arguments":{}}}' || true
echo

echo "Checking chroniccare_agent_run"
curl --noproxy '*' -fsS "${MCP_ENDPOINT}" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"call-3","method":"tools/call","params":{"name":"chroniccare_agent_run","arguments":{"user_goal":"请总结当前慢病系统的图谱质量、NL2SQL 准确率和可视化产物。"}}}' || true
echo

echo "Checking nexent-runtime -> MCP Adapter"
docker exec nexent-runtime python -c "import urllib.request; opener=urllib.request.build_opener(urllib.request.ProxyHandler({})); print(opener.open('${MCP_BASE_URL}', timeout=5).read().decode()[:500])" || true
echo

echo "Checking nexent-mcp -> MCP Adapter"
docker exec nexent-mcp python -c "import urllib.request; opener=urllib.request.build_opener(urllib.request.ProxyHandler({})); print(opener.open('${MCP_BASE_URL}', timeout=5).read().decode()[:500])" || true
echo

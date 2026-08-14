#!/usr/bin/env bash
set -euo pipefail

CHRONICCARE_DEPLOY_ROOT="${CHRONICCARE_DEPLOY_ROOT:-/opt/chroniccare}"

mkdir -p "${CHRONICCARE_DEPLOY_ROOT}/project"
mkdir -p "${CHRONICCARE_DEPLOY_ROOT}/data"
mkdir -p "${CHRONICCARE_DEPLOY_ROOT}/outputs"
mkdir -p "${CHRONICCARE_DEPLOY_ROOT}/logs"
mkdir -p "${CHRONICCARE_DEPLOY_ROOT}/models/qwen2_5_vl_7b"
mkdir -p "${CHRONICCARE_DEPLOY_ROOT}/env"

if [ ! -f "${CHRONICCARE_DEPLOY_ROOT}/env/.env" ]; then
  echo "CHRONICCARE_DEPLOY_ROOT=${CHRONICCARE_DEPLOY_ROOT}" > "${CHRONICCARE_DEPLOY_ROOT}/env/.env"
  echo "CHRONICCARE_TOOL_SERVER_PORT=18088" >> "${CHRONICCARE_DEPLOY_ROOT}/env/.env"
  echo "CHRONICCARE_STREAMLIT_PORT=18501" >> "${CHRONICCARE_DEPLOY_ROOT}/env/.env"
  echo "CHRONICCARE_TOOL_SERVER_URL=http://chroniccare-tool-server:18088" >> "${CHRONICCARE_DEPLOY_ROOT}/env/.env"
fi

touch "${CHRONICCARE_DEPLOY_ROOT}/models/qwen2_5_vl_7b/.gitkeep"

echo "Prepared server directories under ${CHRONICCARE_DEPLOY_ROOT}"
echo "Do not copy real model weights automatically. Place them manually under models/qwen2_5_vl_7b when needed."

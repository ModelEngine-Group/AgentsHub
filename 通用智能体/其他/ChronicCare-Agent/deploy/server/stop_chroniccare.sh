#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHRONICCARE_DEPLOY_ROOT="${CHRONICCARE_DEPLOY_ROOT:-/opt/chroniccare}"
ENV_FILE="${CHRONICCARE_DEPLOY_ROOT}/env/.env"

if [ -f "${ENV_FILE}" ]; then
  set -a
  . "${ENV_FILE}"
  set +a
fi

cd "${PROJECT_ROOT}"
docker compose -f docker-compose.server.yml down
echo "ChronicCare-Agent server stack stopped."

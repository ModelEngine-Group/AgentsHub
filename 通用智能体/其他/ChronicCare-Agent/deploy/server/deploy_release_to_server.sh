#!/usr/bin/env bash
set -euo pipefail

CHRONICCARE_DEPLOY_ROOT="${CHRONICCARE_DEPLOY_ROOT:-/opt/chroniccare}"
TARGET_DIR="${CHRONICCARE_DEPLOY_ROOT}/project/ChronicCare-Agent"
CURRENT_DIR="$(pwd)"

mkdir -p "${CHRONICCARE_DEPLOY_ROOT}/project"
rm -rf "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude '.env' "${CURRENT_DIR}/" "${TARGET_DIR}/"
else
  cp -a "${CURRENT_DIR}/." "${TARGET_DIR}/"
  if [ -f "${TARGET_DIR}/.env" ]; then
    rm -f "${TARGET_DIR}/.env"
  fi
fi

echo "Release copied to ${TARGET_DIR}"
echo "Model weights are not copied by this script."

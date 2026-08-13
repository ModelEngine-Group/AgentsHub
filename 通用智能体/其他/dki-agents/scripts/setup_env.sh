#!/usr/bin/env bash
# Reproducible Python 3.11 environment bootstrap for Linux and macOS.
#
# Nexent and the DataMate executor are installed with --no-deps so their
# broad upstream dependency declarations cannot replace a target machine's
# fixed torch/torch_npu stack.  Their required lightweight dependencies are
# installed explicitly.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/4] Creating the Python 3.11 virtual environment"
uv venv .venv --python 3.11
PY=".venv/bin/python"
if [ -f ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
fi

echo "[2/4] Installing Nexent 0.1.2 without its broad dependency extras"
uv pip install -p "$PY" \
  "smolagents==1.12.0" \
  "huggingface-hub==0.28.1" \
  "aiofiles==25.1.0" \
  "elasticsearch==8.17.2" \
  "exa_py==2.16.2" \
  "httpx==0.28.1" \
  "openai==2.50.0" \
  "openpyxl==3.1.5" \
  "pydantic==2.13.4" \
  "websockets==16.1.1" \
  "aiohttp==3.14.3" \
  "jieba==0.42.1" \
  "boto3==1.43.57" \
  "botocore==1.43.57" \
  "python-multipart==0.0.32" \
  "markdownify==1.2.3" \
  "duckduckgo-search==8.1.1"
if [ -d "../nexent/sdk" ]; then
  uv pip install -p "$PY" --no-deps ../nexent/sdk
else
  uv pip install -p "$PY" --no-deps "nexent==0.1.2"
fi

echo "[3/4] Installing the DataMate executor"
DATAMATE_RUNTIME="${DKI_DATAMATE_RUNTIME:-../datamate/runtime}"
REQUIRE_REAL_DATAMATE=0
case "${DKI_REQUIRE_REAL_DATAMATE:-${DKI_USE_DATAMATE_RUNTIME:-0}}" in
  1|true|TRUE|yes|YES|on|ON) REQUIRE_REAL_DATAMATE=1 ;;
esac

if [ -d "$DATAMATE_RUNTIME/ops" ] &&
   [ -d "$DATAMATE_RUNTIME/python-executor" ]; then
  DATAMATE_RUNTIME="$(cd "$DATAMATE_RUNTIME" && pwd -P)"
  export DKI_DATAMATE_RUNTIME="$DATAMATE_RUNTIME"
  uv pip install -p "$PY" --no-deps "$DATAMATE_RUNTIME/python-executor"
  uv pip install -p "$PY" \
    opencv-python-headless \
    numpy \
    loguru \
    pytz \
    "unstructured==0.18.15" \
    sqlalchemy \
    requests \
    emoji \
    zhconv \
    beautifulsoup4 \
    xmltodict \
    datasketch
elif [ "$REQUIRE_REAL_DATAMATE" -eq 1 ]; then
  echo \
    "Required DataMate runtime must contain ops/ and python-executor/: $DATAMATE_RUNTIME" \
    >&2
  exit 3
else
  echo \
    "DataMate runtime not found at $DATAMATE_RUNTIME; shim mode remains available." \
    >&2
fi

echo "[4/4] Installing project dependencies"
uv pip install -p "$PY" \
  pyyaml \
  jinja2 \
  pandas \
  python-dotenv \
  pytest \
  sqlglot \
  "streamlit==1.54.0"

echo "Environment ready. Run: ./run.sh demo-all"

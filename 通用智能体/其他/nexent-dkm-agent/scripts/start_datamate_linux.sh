#!/usr/bin/env bash
set -euo pipefail

compose_dir=""
compose_file="docker-compose.local.yml"
mode="up"
registry="ghcr.io/modelengine-group/"
datamate_url="http://localhost:18000"
wait_seconds=90
probe_interval_seconds=10
evidence_dir=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: bash scripts/start_datamate_linux.sh [options]

Linux-first DataMate launcher for review/demo runs. It starts the DataMate
Docker Compose stack, waits for health plus the database-backed core APIs, and
stores the probe log under an ignored outputs/ folder.

Options:
  --compose-dir PATH        DataMate compose directory. Defaults to ../DataMate/deployment/docker/datamate.
  --compose-file FILE       Compose file name. Default: docker-compose.local.yml
  --mode up|start           up runs "docker compose up -d"; start starts existing containers. Default: up
  --registry URL            REGISTRY value passed to compose. Default: ghcr.io/modelengine-group/
  --datamate-url URL        DataMate backend URL. Default: http://localhost:18000
  --wait-seconds N          Max seconds to wait for health. Default: 90
  --probe-interval N        Seconds between probes. Default: 10
  --evidence-dir PATH       Log directory. Default: outputs/competition_evidence/datamate-linux-start
  --dry-run                 Print resolved commands without starting services.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compose-dir)
      compose_dir="${2:?missing value for --compose-dir}"
      shift 2
      ;;
    --compose-file)
      compose_file="${2:?missing value for --compose-file}"
      shift 2
      ;;
    --mode)
      mode="${2:?missing value for --mode}"
      shift 2
      ;;
    --registry)
      registry="${2:?missing value for --registry}"
      shift 2
      ;;
    --datamate-url)
      datamate_url="${2:?missing value for --datamate-url}"
      shift 2
      ;;
    --wait-seconds)
      wait_seconds="${2:?missing value for --wait-seconds}"
      shift 2
      ;;
    --probe-interval)
      probe_interval_seconds="${2:?missing value for --probe-interval}"
      shift 2
      ;;
    --evidence-dir)
      evidence_dir="${2:?missing value for --evidence-dir}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$mode" != "up" && "$mode" != "start" ]]; then
  echo "--mode must be 'up' or 'start'." >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
workspace_root="$(cd "$project_root/.." && pwd)"

if [[ -z "$compose_dir" ]]; then
  if [[ -f "$PWD/$compose_file" ]]; then
    compose_dir="$PWD"
  else
    compose_dir="$workspace_root/DataMate/deployment/docker/datamate"
  fi
fi

if [[ -z "$evidence_dir" ]]; then
  evidence_dir="$project_root/outputs/competition_evidence/datamate-linux-start"
fi

if [[ "$mode" == "start" ]]; then
  compose_args=(
    start
    datamate-database
    datamate-runtime
    datamate-frontend
    datamate-backend-python
    datamate-backend
    datamate-gateway
  )
else
  compose_args=(up -d)
fi

if [[ "$dry_run" -eq 1 ]]; then
  echo "DataMate compose dir: $compose_dir"
  echo "Compose file: $compose_file"
  echo "Mode: $mode"
  echo "REGISTRY=$registry docker compose -f '$compose_file' ${compose_args[*]}"
  echo "DataMate URL: $datamate_url"
  exit 0
fi

if [[ ! -d "$compose_dir" ]]; then
  echo "DataMate compose directory does not exist: $compose_dir" >&2
  exit 1
fi

mkdir -p "$evidence_dir"
stamp="$(date +%Y%m%d-%H%M%S)"
log_path="$evidence_dir/start-datamate-linux-$stamp.log"

log() {
  printf '%s\n' "$*" | tee -a "$log_path"
}

probe_readiness() {
  if command -v python3 >/dev/null 2>&1; then
    python3 "$project_root/scripts/datamate_readiness.py" \
      --url "$datamate_url" \
      --timeout 8
    return
  fi
  if command -v python >/dev/null 2>&1; then
    python "$project_root/scripts/datamate_readiness.py" \
      --url "$datamate_url" \
      --timeout 8
    return
  fi
  echo "Python is required for DataMate readiness probing." >&2
  return 127
}

log "timestamp=$(date -Is)"
log "compose_dir=$compose_dir"
log "compose_file=$compose_file"
log "mode=$mode"
log "datamate_url=$datamate_url"

(
  cd "$compose_dir"
  REGISTRY="$registry" docker compose -f "$compose_file" "${compose_args[@]}"
) 2>&1 | tee -a "$log_path"

deadline=$((SECONDS + wait_seconds))
probe_index=0
while (( SECONDS <= deadline )); do
  log ""
  log "probe=$probe_index timestamp=$(date -Is)"
  if readiness_body="$(probe_readiness 2>&1)"; then
    log "readiness_exit_code=0 report=$readiness_body"
    log "status=ready"
    echo "DataMate is ready. Evidence log: $log_path"
    exit 0
  else
    readiness_exit_code=$?
    log "readiness_exit_code=$readiness_exit_code report=$readiness_body"
  fi

  docker ps -a --filter name=datamate --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>&1 | tee -a "$log_path"

  probe_index=$((probe_index + 1))
  sleep "$probe_interval_seconds"
done

log "status=not_ready_after_${wait_seconds}s"
echo "DataMate did not become ready within $wait_seconds seconds. Evidence log: $log_path"
exit 1

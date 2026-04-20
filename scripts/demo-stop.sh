#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/demo}"
STOP_DOCKER_COMPOSE="${STOP_DOCKER_COMPOSE:-false}"
DELETE_K8S_CLUSTER="${DELETE_K8S_CLUSTER:-false}"

stop_from_pid_file() {
  local pid_file="$1"
  local label="$2"
  local pid

  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi

  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    printf 'Stopped %s (pid %s).\n' "${label}" "${pid}"
  fi
  rm -f "${pid_file}"
}

stop_from_pid_file "${LOG_DIR}/streamlit.pid" "Streamlit"
stop_from_pid_file "${LOG_DIR}/api.pid" "FastAPI"
stop_from_pid_file "${LOG_DIR}/mlflow.pid" "MLflow"

if [[ "${STOP_DOCKER_COMPOSE}" == "true" ]]; then
  docker compose down
fi

if [[ "${DELETE_K8S_CLUSTER}" == "true" ]]; then
  export PATH="${ROOT_DIR}/.tools/bin:${PATH}"
  if command -v kind >/dev/null 2>&1; then
    kind delete cluster --name mlops-cluster
  fi
fi

printf 'Demo cleanup complete.\n'

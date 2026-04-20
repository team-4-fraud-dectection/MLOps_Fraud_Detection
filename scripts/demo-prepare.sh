#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/demo}"
mkdir -p "${LOG_DIR}"

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
ENABLE_K8S="${ENABLE_K8S:-false}"
RECREATE_CLUSTER="${RECREATE_CLUSTER:-false}"

wait_for_http() {
  local url="$1"
  local timeout_seconds="${2:-60}"
  local start_ts
  start_ts="$(date +%s)"
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if (( "$(date +%s)" - start_ts >= timeout_seconds )); then
      echo "Timed out waiting for ${url}" >&2
      return 1
    fi
    sleep 2
  done
}

pick_mlflow_port() {
  local preferred_port="$1"
  local candidate_port

  for candidate_port in "${preferred_port}" 5001; do
    if curl -fsS "http://${MLFLOW_HOST}:${candidate_port}" >/dev/null 2>&1; then
      echo "${candidate_port}"
      return 0
    fi

    if ! lsof -iTCP:"${candidate_port}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
      echo "${candidate_port}"
      return 0
    fi
  done

  echo "Could not find a usable port for MLflow." >&2
  return 1
}

start_if_missing() {
  local port="$1"
  local healthcheck_url="$2"
  local command="$3"
  local pid_file="$4"
  local log_file="$5"

  if curl -fsS "${healthcheck_url}" >/dev/null 2>&1; then
    echo "Service already responding on ${healthcheck_url}; skipping startup."
    return 0
  fi

  if lsof -iTCP:"${port}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    echo "Port ${port} already has a listener; waiting for it to respond."
    return 0
  fi

  nohup bash -lc "${command}" >"${log_file}" 2>&1 &
  echo $! >"${pid_file}"
}

MLFLOW_PORT="$(pick_mlflow_port "${MLFLOW_PORT}")"
start_if_missing \
  "${MLFLOW_PORT}" \
  "http://${MLFLOW_HOST}:${MLFLOW_PORT}" \
  "mlflow ui --host ${MLFLOW_HOST} --port ${MLFLOW_PORT}" \
  "${LOG_DIR}/mlflow.pid" \
  "${LOG_DIR}/mlflow.log"
wait_for_http "http://${MLFLOW_HOST}:${MLFLOW_PORT}"

start_if_missing \
  "${API_PORT}" \
  "http://${API_HOST}:${API_PORT}/health" \
  "uvicorn src.api:app --host ${API_HOST} --port ${API_PORT}" \
  "${LOG_DIR}/api.pid" \
  "${LOG_DIR}/api.log"
wait_for_http "http://${API_HOST}:${API_PORT}/health"

start_if_missing \
  "${STREAMLIT_PORT}" \
  "http://${API_HOST}:${STREAMLIT_PORT}" \
  "streamlit run src/streamlit.py --server.headless true --server.port ${STREAMLIT_PORT}" \
  "${LOG_DIR}/streamlit.pid" \
  "${LOG_DIR}/streamlit.log"
wait_for_http "http://${API_HOST}:${STREAMLIT_PORT}"

if [[ "${ENABLE_K8S}" == "true" ]]; then
  export PATH="${ROOT_DIR}/.tools/bin:${PATH}"
  RECREATE_CLUSTER="${RECREATE_CLUSTER}" bash "${ROOT_DIR}/scripts/run-k8s-e2e.sh"
fi

printf '\nDemo environment is ready.\n'
printf 'MLflow: http://%s:%s\n' "${MLFLOW_HOST}" "${MLFLOW_PORT}"
printf 'FastAPI: http://%s:%s/docs\n' "${API_HOST}" "${API_PORT}"
printf 'Streamlit: http://%s:%s\n' "${API_HOST}" "${STREAMLIT_PORT}"
if [[ "${ENABLE_K8S}" == "true" ]]; then
  printf 'Kubernetes API: http://127.0.0.1:30007/docs\n'
  printf 'Prometheus: http://127.0.0.1:30300\n'
  printf 'Grafana: http://127.0.0.1:30200\n'
fi

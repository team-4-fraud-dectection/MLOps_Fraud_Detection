#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-30300}"
GRAFANA_PORT="${GRAFANA_PORT:-30200}"
OPEN_EDITOR="${OPEN_EDITOR:-true}"

open_target() {
  local target="$1"

  if command -v open >/dev/null 2>&1; then
    open "${target}"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${target}" >/dev/null 2>&1 &
  else
    echo "No supported opener found for ${target}" >&2
    return 1
  fi
}

open_editor_file() {
  local target="$1"

  if command -v code >/dev/null 2>&1; then
    code -g "${target}"
  elif command -v open >/dev/null 2>&1; then
    open -a "Visual Studio Code" "${target}"
  else
    echo "Skipping editor open for ${target}; VS Code launcher not available."
  fi
}

for url in \
  "http://127.0.0.1:${MLFLOW_PORT}" \
  "http://127.0.0.1:${API_PORT}/docs" \
  "http://127.0.0.1:${STREAMLIT_PORT}" \
  "http://127.0.0.1:${PROMETHEUS_PORT}/targets" \
  "http://127.0.0.1:${PROMETHEUS_PORT}/graph" \
  "http://127.0.0.1:${GRAFANA_PORT}/explore"; do
  open_target "${url}"
done

if [[ "${OPEN_EDITOR}" == "true" ]]; then
  for file in \
    "README.md" \
    "docs/system_architecture.md" \
    "docs/demo_runbook.md" \
    "dvc.yaml" \
    "metrics/train_metrics.json" \
    "deployment/kubernetes/deployment.yaml" \
    ".github/workflows/quality-ci.yml" \
    ".github/workflows/continuous-training.yml"; do
    open_editor_file "${ROOT_DIR}/${file}"
  done
fi

if command -v pbcopy >/dev/null 2>&1; then
  pbcopy < "${ROOT_DIR}/sample_request.json"
  printf 'Copied sample_request.json to clipboard.\n'
fi

printf 'Opened demo browser tabs and editor files.\n'

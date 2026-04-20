#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-mlops-cluster}"
IMAGE_NAME="${IMAGE_NAME:-ghcr.io/team-5-fraud-dectection/mlops-fraud-detection:latest}"
SCRAPE_WAIT_SECONDS="${SCRAPE_WAIT_SECONDS:-30}"
RECREATE_CLUSTER="${RECREATE_CLUSTER:-false}"
IMAGE_LOAD_RETRIES="${IMAGE_LOAD_RETRIES:-5}"
IMAGE_LOAD_RETRY_DELAY_SECONDS="${IMAGE_LOAD_RETRY_DELAY_SECONDS:-8}"
HELM_RETRIES="${HELM_RETRIES:-5}"
HELM_RETRY_DELAY_SECONDS="${HELM_RETRY_DELAY_SECONDS:-15}"

"${ROOT_DIR}/scripts/install-local-k8s-tools.sh" >/dev/null
export PATH="${ROOT_DIR}/.tools/bin:${PATH}"

if kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  if [[ "${RECREATE_CLUSTER}" == "true" ]]; then
    kind delete cluster --name "${CLUSTER_NAME}"
  fi
fi

if ! kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  kind create cluster --name "${CLUSTER_NAME}" --config "${ROOT_DIR}/deployment/kubernetes/kind-three-node-cluster.yaml"
fi

kubectl config use-context "kind-${CLUSTER_NAME}" >/dev/null
kubectl wait --for=condition=Ready nodes --all --timeout=300s
sleep 5

docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"

load_image_into_kind() {
  local attempt
  for attempt in $(seq 1 "${IMAGE_LOAD_RETRIES}"); do
    if kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"; then
      return 0
    fi

    if [[ "${attempt}" -lt "${IMAGE_LOAD_RETRIES}" ]]; then
      echo "kind image load failed on attempt ${attempt}/${IMAGE_LOAD_RETRIES}; retrying in ${IMAGE_LOAD_RETRY_DELAY_SECONDS}s..." >&2
      sleep "${IMAGE_LOAD_RETRY_DELAY_SECONDS}"
    fi
  done

  echo "Failed to load image into kind after ${IMAGE_LOAD_RETRIES} attempts." >&2
  return 1
}

run_with_retries() {
  local description="$1"
  shift

  local attempt
  for attempt in $(seq 1 "${HELM_RETRIES}"); do
    if "$@"; then
      return 0
    fi

    if [[ "${attempt}" -lt "${HELM_RETRIES}" ]]; then
      echo "${description} failed on attempt ${attempt}/${HELM_RETRIES}; retrying in ${HELM_RETRY_DELAY_SECONDS}s..." >&2
      sleep "${HELM_RETRY_DELAY_SECONDS}"
    fi
  done

  echo "${description} failed after ${HELM_RETRIES} attempts." >&2
  return 1
}

load_image_into_kind

kubectl apply -k "${ROOT_DIR}/deployment/kubernetes"
kubectl rollout status deployment/ml-api --timeout=300s

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
run_with_retries "helm repo update" helm repo update >/dev/null
run_with_retries "helm upgrade/install kube-prometheus-stack" \
  helm upgrade --install prom \
    -n monitoring \
    --create-namespace \
    prometheus-community/kube-prometheus-stack \
    -f "${ROOT_DIR}/deployment/monitoring/kube-prometheus-stack-values.yaml"

kubectl rollout status deployment/prom-grafana -n monitoring --timeout=600s
kubectl rollout status deployment/prom-kube-prometheus-stack-operator -n monitoring --timeout=600s
kubectl apply -k "${ROOT_DIR}/deployment/monitoring"

kubectl wait --for=condition=Ready pod -n monitoring -l app.kubernetes.io/instance=prom --timeout=600s

curl -fsS http://127.0.0.1:30007/health >/dev/null
curl -fsS -X POST http://127.0.0.1:30007/predict \
  -H "Content-Type: application/json" \
  --data @"${ROOT_DIR}/sample_request_predict.json" >/dev/null

sleep "${SCRAPE_WAIT_SECONDS}"

python - <<'PY'
import json
import time
import urllib.request


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.load(response)


deadline = time.time() + 180
last_error = "Prometheus target for ml-api-service not ready yet."
while time.time() < deadline:
    try:
        data = fetch_json("http://127.0.0.1:30300/api/v1/targets")
        active_targets = data.get("data", {}).get("activeTargets", [])
        matches = [
            target for target in active_targets
            if "ml-api-service" in target.get("scrapeUrl", "")
            or target.get("labels", {}).get("service") == "ml-api-service"
        ]
        if matches and any(target.get("health") == "up" for target in matches):
            print(json.dumps({"prometheus_target_health": "up", "matches": len(matches)}, indent=2))
            break
    except Exception as exc:  # pragma: no cover - operational check
        last_error = str(exc)
    time.sleep(5)
else:
    raise SystemExit(last_error)
PY

printf '\nAPI health:\n'
curl -fsS http://127.0.0.1:30007/health

printf '\n\nAPI metrics sample:\n'
curl -fsS http://127.0.0.1:30007/metrics | head -n 10

printf '\n\nPrometheus query sample:\n'
python - <<'PY'
import json
import urllib.parse
import urllib.request

query = urllib.parse.quote("http_requests_total")
with urllib.request.urlopen(f"http://127.0.0.1:30300/api/v1/query?query={query}", timeout=20) as response:
    data = json.load(response)
print(json.dumps(data, indent=2)[:2000])
PY

printf '\n\nFraud prediction metric sample:\n'
python - <<'PY'
import json
import urllib.parse
import urllib.request

query = urllib.parse.quote('sum(increase(fraud_predictions_total{prediction="fraud"}[5m]))')
with urllib.request.urlopen(f"http://127.0.0.1:30300/api/v1/query?query={query}", timeout=20) as response:
    data = json.load(response)
print(json.dumps(data, indent=2)[:1200])
PY

printf '\n\nGrafana: http://127.0.0.1:30200\n'
printf 'Prometheus: http://127.0.0.1:30300\n'
printf 'API docs: http://127.0.0.1:30007/docs\n'
printf 'Grafana dashboard: Fraud Detection API Overview\n'

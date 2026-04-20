#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOCAL_API_URL="${LOCAL_API_URL:-http://127.0.0.1:8000}"
K8S_API_URL="${K8S_API_URL:-http://127.0.0.1:30007}"
SKIP_K8S="${SKIP_K8S:-false}"

printf 'Warming up local /predict_raw ...\n'
curl -fsS -X POST "${LOCAL_API_URL}/predict_raw" \
  -H "Content-Type: application/json" \
  --data @"${ROOT_DIR}/sample_request.json" >/tmp/demo_predict_raw.json
python - <<'PY'
import json
payload = json.load(open("/tmp/demo_predict_raw.json", encoding="utf-8"))
print({"local_predict_raw_n_records": payload.get("n_records")})
PY

printf '\nWarming up local /predict ...\n'
curl -fsS -X POST "${LOCAL_API_URL}/predict" \
  -H "Content-Type: application/json" \
  --data @"${ROOT_DIR}/sample_request_predict.json" >/tmp/demo_predict.json
python - <<'PY'
import json
payload = json.load(open("/tmp/demo_predict.json", encoding="utf-8"))
print({"local_predict_n_records": payload.get("n_records")})
PY

if [[ "${SKIP_K8S}" != "true" ]]; then
  printf '\nWarming up Kubernetes /predict ...\n'
  curl -fsS -X POST "${K8S_API_URL}/predict" \
    -H "Content-Type: application/json" \
    --data @"${ROOT_DIR}/sample_request_predict.json" >/tmp/demo_k8s_predict.json
  python - <<'PY'
import json
payload = json.load(open("/tmp/demo_k8s_predict.json", encoding="utf-8"))
print({"k8s_predict_n_records": payload.get("n_records")})
PY
fi

if command -v pbcopy >/dev/null 2>&1; then
  pbcopy < "${ROOT_DIR}/sample_request.json"
  printf '\nsample_request.json copied to clipboard.\n'
fi

printf '\nWarm-up complete.\n'

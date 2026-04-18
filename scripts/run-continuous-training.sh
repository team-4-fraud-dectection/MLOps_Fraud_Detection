#!/usr/bin/env bash
set -euo pipefail

eval "$(python - <<'PY'
from pathlib import Path
import shlex
import yaml

payload = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8")) or {}
ct = payload.get("continuous_training", {}) or {}

defaults = {
    "DEFAULT_STATUS_SUMMARY_PATH": ct.get("status_summary_path", "reports/monitoring/status_summary.json"),
    "DEFAULT_DECISION_REPORT_PATH": ct.get("decision_report_path", "reports/monitoring/ct_decision.json"),
    "DEFAULT_PIPELINE_SCOPE": ct.get("pipeline_scope", "train"),
}

for key, value in defaults.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

STATUS_SUMMARY_PATH="${STATUS_SUMMARY_PATH:-$DEFAULT_STATUS_SUMMARY_PATH}"
DECISION_REPORT_PATH="${DECISION_REPORT_PATH:-$DEFAULT_DECISION_REPORT_PATH}"
PIPELINE_SCOPE="${PIPELINE_SCOPE:-$DEFAULT_PIPELINE_SCOPE}"
FORCE_RETRAIN="${FORCE_RETRAIN:-false}"
FORCE_STAGE_REBUILD="${FORCE_STAGE_REBUILD:-true}"
DRY_RUN="${DRY_RUN:-false}"

export DECISION_REPORT_PATH

decision_cmd=(
  python src/evaluate_ct_trigger.py
  --status-summary-path "$STATUS_SUMMARY_PATH"
  --output-path "$DECISION_REPORT_PATH"
)

if [[ "$FORCE_RETRAIN" == "true" ]]; then
  decision_cmd+=(--force-retrain)
fi

"${decision_cmd[@]}"

SHOULD_RETRAIN="$(python - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["DECISION_REPORT_PATH"]).read_text(encoding="utf-8"))
print(str(payload.get("should_retrain", False)).lower())
PY
)"

if [[ "$SHOULD_RETRAIN" != "true" ]]; then
  echo "Continuous training skipped because monitoring is within thresholds."
  exit 0
fi

ct_cmd=(dvc repro)
if [[ "$FORCE_STAGE_REBUILD" == "true" ]]; then
  ct_cmd+=(--force)
fi
if [[ "$PIPELINE_SCOPE" != "full" ]]; then
  ct_cmd+=(--single-item)
  ct_cmd+=(train)
fi

if [[ "$DRY_RUN" == "true" ]]; then
  printf 'Continuous training command:'
  printf ' %q' "${ct_cmd[@]}"
  printf '\n'
  exit 0
fi

"${ct_cmd[@]}"

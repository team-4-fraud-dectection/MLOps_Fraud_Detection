#!/usr/bin/env bash
set -euo pipefail

REFERENCE_PATH="${REFERENCE_PATH:-data/featured/X_val.parquet}"
CURRENT_PATH="${CURRENT_PATH:-}"
PREDICTION_LOG_PATH="${PREDICTION_LOG_PATH:-logs/predictions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/drift}"
DRIFT_SHARE_THRESHOLD="${DRIFT_SHARE_THRESHOLD:-0.5}"
MIN_CURRENT_RECORDS="${MIN_CURRENT_RECORDS:-30}"
MAX_CURRENT_RECORDS="${MAX_CURRENT_RECORDS:-0}"
ENDPOINT="${ENDPOINT:-/predict}"

docker run --rm \
  -e REFERENCE_PATH="$REFERENCE_PATH" \
  -e CURRENT_PATH="$CURRENT_PATH" \
  -e PREDICTION_LOG_PATH="$PREDICTION_LOG_PATH" \
  -e OUTPUT_DIR="$OUTPUT_DIR" \
  -e DRIFT_SHARE_THRESHOLD="$DRIFT_SHARE_THRESHOLD" \
  -e MIN_CURRENT_RECORDS="$MIN_CURRENT_RECORDS" \
  -e MAX_CURRENT_RECORDS="$MAX_CURRENT_RECORDS" \
  -e ENDPOINT="$ENDPOINT" \
  -v "$PWD:/app" \
  -w /app \
  python:3.11-slim \
  sh -lc '
    pip install -q numpy pandas pyarrow scikit-learn evidently &&
    python src/monitor_drift.py \
      --reference_path "$REFERENCE_PATH" \
      ${CURRENT_PATH:+--current_path "$CURRENT_PATH"} \
      --prediction_log_path "$PREDICTION_LOG_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --drift_share_threshold "$DRIFT_SHARE_THRESHOLD" \
      --min_current_records "$MIN_CURRENT_RECORDS" \
      --max_current_records "$MAX_CURRENT_RECORDS" \
      --endpoint "$ENDPOINT"
  '

#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
ENDPOINT="${ENDPOINT:-/predict}"
REFERENCE_PATH="${REFERENCE_PATH:-data/featured/X_train.parquet}"
CURRENT_X_PATH="${CURRENT_X_PATH:-data/featured/X_val.parquet}"
CURRENT_Y_PATH="${CURRENT_Y_PATH:-data/featured/y_val.parquet}"
PREDICTION_LOG_PATH="${PREDICTION_LOG_PATH:-logs/predictions.jsonl}"
FEEDBACK_LOG_PATH="${FEEDBACK_LOG_PATH:-logs/prediction_feedback.jsonl}"
PERFORMANCE_OUTPUT_PATH="${PERFORMANCE_OUTPUT_PATH:-reports/monitoring/performance_summary.json}"
DRIFT_OUTPUT_DIR="${DRIFT_OUTPUT_DIR:-reports/drift}"
STATUS_OUTPUT_PATH="${STATUS_OUTPUT_PATH:-reports/monitoring/status_summary.json}"
MAX_RECORDS="${MAX_RECORDS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
DRIFT_SHARE_THRESHOLD="${DRIFT_SHARE_THRESHOLD:-0.5}"
PERFORMANCE_F1_THRESHOLD="${PERFORMANCE_F1_THRESHOLD:-0.7}"
MIN_CURRENT_RECORDS="${MIN_CURRENT_RECORDS:-30}"

python src/replay_monitoring_window.py \
  --base_url "$BASE_URL" \
  --endpoint "$ENDPOINT" \
  --x_path "$CURRENT_X_PATH" \
  --y_path "$CURRENT_Y_PATH" \
  --prediction_log_path "$PREDICTION_LOG_PATH" \
  --feedback_log_path "$FEEDBACK_LOG_PATH" \
  --batch_size "$BATCH_SIZE" \
  --max_records "$MAX_RECORDS" \
  --sample_seed "$SAMPLE_SEED" \
  --reset_logs

python src/monitor_performance.py \
  --prediction_log_path "$PREDICTION_LOG_PATH" \
  --feedback_log_path "$FEEDBACK_LOG_PATH" \
  --output_path "$PERFORMANCE_OUTPUT_PATH" \
  --endpoint "$ENDPOINT"

REFERENCE_PATH="$REFERENCE_PATH" \
PREDICTION_LOG_PATH="$PREDICTION_LOG_PATH" \
OUTPUT_DIR="$DRIFT_OUTPUT_DIR" \
DRIFT_SHARE_THRESHOLD="$DRIFT_SHARE_THRESHOLD" \
MIN_CURRENT_RECORDS="$MIN_CURRENT_RECORDS" \
ENDPOINT="$ENDPOINT" \
bash scripts/run-drift-report.sh

python src/monitor_status.py \
  --performance_summary_path "$PERFORMANCE_OUTPUT_PATH" \
  --drift_summary_path "$DRIFT_OUTPUT_DIR/data_drift_summary.json" \
  --output_path "$STATUS_OUTPUT_PATH" \
  --performance_f1_threshold "$PERFORMANCE_F1_THRESHOLD" \
  --drift_share_threshold "$DRIFT_SHARE_THRESHOLD"

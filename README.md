## Setup

### 1. Clone repository

```bash
git clone https://github.com/team-5-fraud-dectection/MLOps_Fraud_Detection.git
cd MLOps_Fraud_Detection
git checkout ldtesting
````

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data (DVC)

### Configure DVC remote (DagsHub)

```bash
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/rizerize-1/DVC_Fraud_Detection.s3
```

PowerShell:

```powershell
$DAGSHUB_TOKEN="YOUR_TOKEN"
dvc remote modify --local origin access_key_id $DAGSHUB_TOKEN
dvc remote modify --local origin secret_access_key $DAGSHUB_TOKEN
```

---

### Pull data and artifacts

```bash
dvc pull
```

---

## Run full pipeline

```bash
dvc repro
```

This will:

* preprocess data
* generate features
* balance dataset
* train models
* save best model + metrics

---

## MLflow (Experiment Tracking)

Start MLflow server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Open UI:

```
http://127.0.0.1:5000
```

### Model Registry Promotion

The training pipeline now registers the champion model to MLflow using
`mlflow.register_model_name` from `params.yaml`.

`Candidate` is represented in MLflow by:

* registered model alias: `candidate`
* model version tag: `deployment_status=Candidate`

Promote the latest registered version manually:

```bash
python src/promote_model.py --model-name fraud_detection_model --tracking-uri "$MLFLOW_TRACKING_URI"
```

GitHub Actions workflow `.github/workflows/model-registry-promotion.yml`
will promote the model automatically after the CI job passes on
`main`, `master`, or `ldtesting`.

Required GitHub secret:

```bash
MLFLOW_TRACKING_URI=<your-remote-mlflow-uri>
```

If the remote registry is hosted on DagsHub, add one of these secret pairs:

```bash
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-access-token>
```

or:

```bash
DAGSHUB_USERNAME=<your-dagshub-username>
DAGSHUB_TOKEN=<your-dagshub-access-token>
```

---

## API (Live Inference)

Run API:

```bash
uvicorn src.api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### Test prediction

Use `sample_request_predict.json`:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data @sample_request_predict.json
```

---

## Docker

### Build image

```bash
docker build -t ieee-fraud-api .
```

### Run container

```bash
docker run -p 8000:8000 ieee-fraud-api
```

---

## Docker Compose

```bash
docker compose up --build
```

## Blue-Green Deployment

Local blue-green deployment files are in [deploy/bluegreen](deploy/bluegreen/README.md).

Quick start:

```bash
docker compose -f deploy/bluegreen/compose.bluegreen.yml up -d fraud-api-blue fraud-api-proxy
curl http://127.0.0.1:8080/health
```

Deploy a new version to green:

```bash
docker build -t ieee-fraud-api:green .
bash scripts/bluegreen-deploy.sh green ieee-fraud-api:green
```

## GCP Cloud Run Deployment

GCP deployment instructions are in [deploy/gcp/README.md](deploy/gcp/README.md).

Quick start:

```bash
gcloud auth login
PROJECT_ID=<your-project-id> REGION=asia-southeast1 SERVICE_NAME=fraud-api TRAFFIC_TAG=green bash scripts/gcp-cloudrun-deploy.sh
SERVICE_NAME=fraud-api REGION=asia-southeast1 TRAFFIC_TAG=green bash scripts/gcp-cloudrun-switch.sh
```

## Monitoring & Drift Detection

Prediction monitoring is built into the FastAPI service:

- `/predict` and `/predict_raw` append per-record prediction events to `logs/predictions.jsonl`
- `/feedback` appends realized labels to `logs/prediction_feedback.jsonl`
- `/health` exposes monitoring paths and model readiness

Generate a more production-like demo window by replaying held-out validation features through the live API:

```bash
bash scripts/run-monitoring-demo.sh
```

This helper:

- uses `data/featured/X_train.parquet` as the reference dataset
- samples `data/featured/X_val.parquet` + `data/featured/y_val.parquet` as the current window
- refreshes `logs/predictions.jsonl` and `logs/prediction_feedback.jsonl`
- rebuilds the performance, drift, and status reports end-to-end

Override the replay size if you want a larger or smaller demo batch:

```bash
MAX_RECORDS=2000 BATCH_SIZE=256 bash scripts/run-monitoring-demo.sh
```

Generate a realized performance summary from prediction + feedback logs:

```bash
python src/monitor_performance.py \
  --prediction_log_path logs/predictions.jsonl \
  --feedback_log_path logs/prediction_feedback.jsonl \
  --output_path reports/monitoring/performance_summary.json
```

Generate an Evidently data drift report:

```bash
python src/monitor_drift.py \
  --reference_path data/featured/X_train.parquet \
  --prediction_log_path logs/predictions.jsonl \
  --output_dir reports/drift \
  --drift_share_threshold 0.5 \
  --min_current_records 30
```

If your local Python version has issues importing Evidently, run the drift report through Docker:

```bash
bash scripts/run-drift-report.sh
```

Combine performance + drift outputs into a single retraining status summary:

```bash
python src/monitor_status.py
```

Notes:

- The drift script compares the reference feature dataset against recent inference features extracted from prediction logs.
- For a quick monitoring demo, use `X_train` as reference and replay a sampled slice of `X_val/y_val` as the current window.
- Evidently works best in Python 3.11. If your local Python is incompatible, run drift reporting through the project's CI or Docker environment.
- `monitor_status.py` is the bridge to CT: it emits `should_retrain=true` when F1 drops below threshold or drift becomes too large.

## Continuous Training (CT)

Continuous training now uses the monitoring output as a gate instead of retraining blindly on every schedule.

Evaluate the latest monitoring status locally:

```bash
python src/evaluate_ct_trigger.py \
  --status-summary-path reports/monitoring/status_summary.json \
  --output-path reports/monitoring/ct_decision.json
```

Run the local CT helper:

```bash
bash scripts/run-continuous-training.sh
```

Useful CT options:

```bash
DRY_RUN=true bash scripts/run-continuous-training.sh
FORCE_RETRAIN=true bash scripts/run-continuous-training.sh
PIPELINE_SCOPE=full bash scripts/run-continuous-training.sh
```

How CT works in this repo:

- `monitor_status.py` writes `reports/monitoring/status_summary.json`
- `evaluate_ct_trigger.py` reads that file and decides whether retraining is required
- if `should_retrain=true`, the local helper reruns the DVC pipeline (`train` stage by default)
- the GitHub Actions workflow `.github/workflows/continuous-training.yml` follows the same logic and only retrains when monitoring requests it, unless `force_retrain` is set on manual dispatch

---

## Model Information

* Best model: XGBoost (based on AUPRC)
* Threshold tuned for optimal F1 score
* Model stored at:

```
models/model.pkl
```

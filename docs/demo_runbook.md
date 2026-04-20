# Demo Runbook

This runbook turns the repository into a repeatable live demo setup for video recording, report evidence, and dry runs before presentation day.

## Ports

- `5000` / `5001`: MLflow UI (the prepare script will fall back to `5001` if `5000` is busy)
- `8000`: local FastAPI service
- `8501`: Streamlit dashboard
- `30007`: Kubernetes NodePort API
- `30300`: Prometheus
- `30200`: Grafana

## Recommended Sequence

1. Prepare the local services:

   ```bash
   bash scripts/demo-prepare.sh
   ```

2. Warm up the monitoring stack so the graphs are not empty:

   ```bash
   bash scripts/demo-warmup.sh
   ```

3. Open the browser tabs and editor files used during the recording:

   ```bash
   bash scripts/demo-open-tabs.sh
   ```

4. If you want the Kubernetes + monitoring part online as well:

   ```bash
   ENABLE_K8S=true RECREATE_CLUSTER=false bash scripts/demo-prepare.sh
   ```

5. After the recording is done, clean up the local services:

   ```bash
   bash scripts/demo-stop.sh
   ```

## Files to Show During the Demo

- `docs/system_architecture.md`
- `README.md`
- `dvc.yaml`
- `metrics/train_metrics.json`
- `reports/monitoring/ct_summary.json`
- `deployment/kubernetes/deployment.yaml`
- `.github/workflows/quality-ci.yml`
- `.github/workflows/continuous-training.yml`

## Suggested Live Flow

1. Show the architecture diagram and explain the end-to-end path.
2. Show the DVC DAG / pipeline files.
3. Show local inference with FastAPI and Streamlit.
4. Show CI/CD workflows in GitHub Actions.
5. Show Kubernetes deployment, Prometheus targets, Grafana Explore, and alerts.
6. Close with monitoring outputs and CT trigger evidence.

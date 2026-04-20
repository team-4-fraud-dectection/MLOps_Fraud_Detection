# Prometheus & Grafana Monitoring

This folder contains the Kubernetes monitoring assets that match the lecture flow:

- expose FastAPI metrics on `/metrics`
- install Prometheus and Grafana with Helm
- scrape the FastAPI service via `ServiceMonitor`
- provision a project-specific Grafana dashboard
- define Prometheus alert rules for API health and fraud spikes

## 1. Install kube-prometheus-stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install prom \
  -n monitoring \
  --create-namespace \
  prometheus-community/kube-prometheus-stack \
  -f deployment/monitoring/kube-prometheus-stack-values.yaml
```

## 2. Apply the ServiceMonitor

```bash
kubectl apply -k deployment/monitoring
kubectl get servicemonitor -n monitoring
```

## 3. Access the dashboards

- Prometheus: `http://localhost:30300`
- Grafana: `http://localhost:30200`
- Built-in project dashboard: `Fraud Detection API Overview`

## 4. Project dashboard contents

The repo now provisions a Grafana dashboard automatically through a labeled ConfigMap.
It highlights both API-level and model-level signals:

- requests over the last 5 minutes
- error rate
- p95 latency
- fraud predictions over the last 5 minutes
- traffic split by handler
- prediction outcomes over time
- last batch fraud rate
- mean fraud probability over the last 5 minutes

## 5. Alert rules shipped with the repo

The Prometheus rule bundle provisions these alerts:

- `MlApiDown`
- `MlApiHighErrorRate`
- `MlApiHighLatencyP95`
- `MlApiFraudSpike`

## 6. Suggested validation checks

Open the Prometheus targets page:

```text
http://localhost:30300/targets
```

Useful queries:

```text
http_requests_total
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
sum(increase(fraud_predictions_total{prediction="fraud"}[5m]))
sum(rate(fraud_prediction_probability_sum[5m])) / sum(rate(fraud_prediction_probability_count[5m]))
```

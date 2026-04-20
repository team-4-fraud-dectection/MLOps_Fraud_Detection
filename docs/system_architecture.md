# System Architecture

This document captures the final end-to-end architecture used in the project demo and report.

## End-to-End MLOps Flow

```mermaid
flowchart LR
    A[Kaggle IEEE-CIS Fraud Detection Data] --> B[DVC Pipeline]
    B --> B1[download_data]
    B1 --> B2[preprocess]
    B2 --> B3[feature_engineering]
    B3 --> B4[balance]
    B4 --> B5[train]

    B5 --> C[Model Artifacts<br/>models + artifacts + metrics]
    B5 --> D[MLflow Tracking & Registry]

    C --> E[FastAPI Inference Service]
    D --> E

    E --> F[Docker Image]
    F --> G[GitHub Actions CI/CD]
    G --> H[GHCR]
    H --> I[Kubernetes Deployment]
    I --> J[Service / NodePort]

    E --> K[Prediction Logs]
    E --> L[Feedback Logs]
    E --> M[/metrics endpoint]

    M --> N[ServiceMonitor]
    N --> O[Prometheus]
    O --> P[Grafana Dashboard]
    O --> Q[Prometheus Alerts]

    K --> R[Performance Monitoring]
    K --> S[Data Drift Monitoring]
    L --> R

    R --> T[Status Summary]
    S --> T
    T --> U[CT Trigger]
    U --> B5
```

## Demo-Oriented View

```mermaid
flowchart TD
    A[Video Demo Operator] --> B[Local FastAPI :8000]
    A --> C[Streamlit UI :8501]
    A --> D[MLflow UI]
    A --> E[Kubernetes API :30007]
    A --> F[Prometheus :30300]
    A --> G[Grafana :30200]

    C --> B
    B --> H[Prediction + Feedback Logs]
    B --> I[/metrics]
    I --> F
    F --> G
    E --> F
```

## Component Notes

- `FastAPI` serves both `/predict` and `/predict_raw`, logs inference events, and exposes Prometheus metrics.
- `MLflow` stores experiment runs and model versions used for promotion / retraining workflows.
- `Prometheus + Grafana` monitor both HTTP traffic and fraud-specific model metrics.
- `Continuous Training` is gated by monitoring outputs, so retraining happens only when the system requests it.
- `Kubernetes` demonstrates production-style deployment, scraping, and observability on a local cluster.

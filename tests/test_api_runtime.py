import json

import numpy as np
from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


class DummyModel:
    def predict_proba(self, X):
        positive = np.full(len(X), 0.9, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class DummyRawPipeline:
    model_name = "raw_demo_model"
    threshold = 0.5

    def prepare_raw_features(self, records, context=None):
        return api.pd.DataFrame([{"feature_a": float(record["TransactionAmt"])} for record in records])

    def predict_feature_matrix(self, prepared_features):
        return [
            {
                "index": index,
                "fraud_probability": 0.91,
                "prediction": 1,
                "threshold": self.threshold,
                "model_name": self.model_name,
            }
            for index in range(len(prepared_features))
        ]


def test_health_reports_raw_pipeline_status():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "model_ready" in payload
    assert "raw_pipeline_ready" in payload
    assert "prediction_log_path" in payload
    assert "feedback_log_path" in payload
    assert payload["metrics_path"] == "/metrics"


def test_metrics_endpoint_is_exposed():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text or "http_request_duration_seconds" in response.text


def test_predict_returns_503_when_model_artifact_is_missing(monkeypatch):
    monkeypatch.setattr(api, "artifact", None)
    monkeypatch.setattr(api, "model", None)
    monkeypatch.setattr(api, "artifact_error", "mock missing model")

    response = client.post(
        "/predict",
        json={"records": [{"TransactionAmt": 100.0}]},
    )

    assert response.status_code == 503
    assert "Model artifact is unavailable" in response.json()["detail"]


def test_predict_raw_returns_503_when_pipeline_artifact_is_missing(monkeypatch):
    monkeypatch.setattr(api, "raw_pipeline", None)
    monkeypatch.setattr(api, "raw_pipeline_error", "mock missing raw pipeline")

    response = client.post(
        "/predict_raw",
        json={"records": [{"TransactionAmt": 100.0}]},
    )

    assert response.status_code == 503
    assert "Raw prediction pipeline is unavailable" in response.json()["detail"]


def test_predict_logs_events_and_returns_ids(monkeypatch, tmp_path):
    prediction_log = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(api, "PREDICTION_LOG_PATH", prediction_log)
    monkeypatch.setattr(
        api,
        "artifact",
        {
            "model_name": "demo_model",
            "model": DummyModel(),
            "threshold": 0.5,
            "feature_name_mapping": None,
            "feature_names": ["feature_a"],
        },
    )
    monkeypatch.setattr(api, "model", api.artifact["model"])
    monkeypatch.setattr(api, "model_name", "demo_model")
    monkeypatch.setattr(api, "threshold", 0.5)
    monkeypatch.setattr(api, "artifact_error", None)

    response = client.post("/predict", json={"records": [{"feature_a": 1.23}]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["n_records"] == 1
    result = payload["results"][0]
    assert result["request_id"]
    assert result["prediction_id"]
    assert prediction_log.exists()

    logged = [json.loads(line) for line in prediction_log.read_text(encoding="utf-8").splitlines()]
    assert logged[0]["event_type"] == "prediction"
    assert logged[0]["endpoint"] == "/predict"
    assert abs(logged[0]["feature__feature_a"] - 1.23) < 1e-6


def test_feedback_endpoint_appends_feedback_log(monkeypatch, tmp_path):
    feedback_log = tmp_path / "prediction_feedback.jsonl"
    monkeypatch.setattr(api, "FEEDBACK_LOG_PATH", feedback_log)

    response = client.post(
        "/feedback",
        json={
            "items": [
                {
                    "prediction_id": "req-1:0",
                    "request_id": "req-1",
                    "actual_label": 1,
                    "feedback_source": "test",
                }
            ]
        },
    )

    assert response.status_code == 200
    assert feedback_log.exists()

    logged = [json.loads(line) for line in feedback_log.read_text(encoding="utf-8").splitlines()]
    assert logged[0]["event_type"] == "feedback"
    assert logged[0]["actual_label"] == 1
    assert logged[0]["prediction_id"] == "req-1:0"


def test_predict_raw_returns_risk_fields(monkeypatch, tmp_path):
    prediction_log = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(api, "PREDICTION_LOG_PATH", prediction_log)
    monkeypatch.setattr(api, "raw_pipeline", DummyRawPipeline())
    monkeypatch.setattr(api, "raw_pipeline_error", None)

    response = client.post(
        "/predict_raw",
        json={"records": [{"TransactionAmt": 125.0}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["n_records"] == 1

    result = payload["results"][0]
    assert result["prediction"] == 1
    assert result["risk_score"] == 91.0
    assert result["risk_level"] == "VERY_HIGH"
    assert result["verification_required"] == "Blocked"
    assert "recommended_action" in result
    assert prediction_log.exists()

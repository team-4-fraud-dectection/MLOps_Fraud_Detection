import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.inference_pipeline import RawInferencePipeline
from src.monitoring import (
    FEEDBACK_LOG_PATH,
    PREDICTION_LOG_PATH,
    append_jsonl,
    build_feedback_events,
    build_prediction_events,
)
from src.download_data import DATA_RAW_DIR, download_kaggle_dataset
from src.risk_score import RiskScoringEngine
from src.validation import validate_feature_matrix, validate_model_artifact


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "models/preprocessor_v1.pkl"))
FEATURE_ARTIFACT_PATH = Path(os.getenv("FEATURE_ARTIFACT_PATH", "artifacts/fe_artifact.pkl"))
INFERENCE_LOG_FILE = Path(os.getenv("INFERENCE_LOG_FILE", "logs/inference_history.csv"))
INFERENCE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="List of feature dictionaries")


class RawPredictionRequest(BaseModel):
    records: list[dict[str, Any]]
    context: list[dict[str, Any]] | None = None


class FeedbackRecord(BaseModel):
    prediction_id: str
    actual_label: int = Field(..., ge=0, le=1)
    request_id: str | None = None
    observed_at: str | None = None
    feedback_source: str | None = None
    notes: str | None = None


class FeedbackRequest(BaseModel):
    items: list[FeedbackRecord]


class DownloadDataRequest(BaseModel):
    competition: str = Field(default="ieee-fraud-detection")
    force: bool = Field(default=False)


def log_inference_data(
    input_records: list[dict[str, Any]],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    endpoint: str,
) -> None:
    df_log = pd.DataFrame(input_records)
    df_log["fraud_probability"] = probabilities
    df_log["prediction"] = predictions
    df_log["endpoint"] = endpoint
    df_log["timestamp"] = datetime.now().isoformat()

    header = not INFERENCE_LOG_FILE.exists()
    df_log.to_csv(INFERENCE_LOG_FILE, mode="a", header=header, index=False)


def sanitize_feature_name(name: str) -> str:
    import re

    name = str(name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def load_artifact(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)
    return validate_model_artifact(artifact)


def apply_feature_mapping(
    X: pd.DataFrame,
    feature_name_mapping: dict[str, str] | None,
    expected_feature_names: list[str] | None,
) -> pd.DataFrame:
    X = X.copy()

    if feature_name_mapping:
        renamed_cols = {}
        for col in X.columns:
            if col in feature_name_mapping:
                renamed_cols[col] = feature_name_mapping[col]
            else:
                renamed_cols[col] = sanitize_feature_name(col)
        X = X.rename(columns=renamed_cols)
    else:
        X.columns = [sanitize_feature_name(c) for c in X.columns]

    if expected_feature_names:
        for col in expected_feature_names:
            if col not in X.columns:
                X[col] = 0.0

        extra_cols = [c for c in X.columns if c not in expected_feature_names]
        if extra_cols:
            logging.warning("Dropping %d unexpected columns.", len(extra_cols))
            X = X.drop(columns=extra_cols)

        X = X[expected_feature_names]

    return X


def prepare_features(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    X = df.copy()

    if "isFraud" in X.columns:
        X = X.drop(columns=["isFraud"])

    expected_feature_names = artifact.get("feature_names")
    feature_name_mapping = artifact.get("feature_name_mapping")

    X = apply_feature_mapping(X, feature_name_mapping, expected_feature_names)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return validate_feature_matrix(X, dataset_name="api feature matrix")


def get_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    raise ValueError(f"Model {type(model).__name__} does not support probability-like outputs.")


def init_model_artifact() -> tuple[dict[str, Any] | None, str | None]:
    try:
        loaded_artifact = load_artifact(MODEL_PATH)
        return loaded_artifact, None
    except FileNotFoundError as exc:
        logging.warning("Primary model artifact is unavailable: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logging.exception("Primary model artifact failed to initialize")
        return None, str(exc)


def init_raw_pipeline() -> tuple[RawInferencePipeline | None, str | None]:
    try:
        pipeline = RawInferencePipeline(
            preprocessor_path=str(PREPROCESSOR_PATH),
            feature_artifact_path=str(FEATURE_ARTIFACT_PATH),
            model_artifact_path=str(MODEL_PATH),
        )
        return pipeline, None
    except FileNotFoundError as exc:
        logging.warning("Raw inference pipeline is unavailable: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logging.exception("Raw inference pipeline failed to initialize")
        return None, str(exc)


artifact: dict[str, Any] | None = None
artifact_error: str | None = None
model: Any | None = None
model_name = "unavailable"
threshold: float | None = None
raw_pipeline: RawInferencePipeline | None = None
raw_pipeline_error: str | None = None
risk_engine = RiskScoringEngine()


def get_or_create_metric(metric_cls, name: str, documentation: str, *args, **kwargs):
    try:
        return metric_cls(name, documentation, *args, registry=REGISTRY, **kwargs)
    except ValueError:
        collector = (
            REGISTRY._names_to_collectors.get(name)
            or REGISTRY._names_to_collectors.get(f"{name}_total")
            or REGISTRY._names_to_collectors.get(f"{name}_bucket")
        )
        if collector is None:
            raise
        return collector


FRAUD_PREDICTIONS = get_or_create_metric(
    Counter,
    "fraud_predictions",
    "Total fraud prediction outcomes served by the API.",
    labelnames=("endpoint", "prediction"),
)
FRAUD_PREDICTION_PROBABILITY = get_or_create_metric(
    Histogram,
    "fraud_prediction_probability",
    "Observed fraud probabilities emitted by the API.",
    labelnames=("endpoint",),
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
)
FRAUD_PREDICTION_BATCH_RATE = get_or_create_metric(
    Gauge,
    "fraud_prediction_batch_rate",
    "Share of fraud predictions in the most recent batch.",
    labelnames=("endpoint",),
)


def record_prediction_metrics(
    endpoint: str,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    if len(predictions) == 0:
        return

    FRAUD_PREDICTION_BATCH_RATE.labels(endpoint=endpoint).set(float(np.mean(predictions)))
    for probability, prediction in zip(probabilities, predictions):
        prediction_label = "fraud" if int(prediction) == 1 else "legit"
        FRAUD_PREDICTIONS.labels(
            endpoint=endpoint,
            prediction=prediction_label,
        ).inc()
        FRAUD_PREDICTION_PROBABILITY.labels(endpoint=endpoint).observe(float(probability))


def refresh_runtime_state() -> None:
    global artifact, artifact_error, model, model_name, threshold, raw_pipeline, raw_pipeline_error

    artifact, artifact_error = init_model_artifact()
    model = artifact["model"] if artifact else None
    model_name = artifact["model_name"] if artifact else "unavailable"
    threshold = float(artifact["threshold"]) if artifact else None
    raw_pipeline, raw_pipeline_error = init_raw_pipeline()


app = FastAPI(
    title="IEEE Fraud Detection API",
    version="1.0.0",
    description="Live inference API for fraud probability prediction.",
)

refresh_runtime_state()


@app.get("/health")
def health():
    return {
        "status": "ok" if artifact is not None else "degraded",
        "model_ready": artifact is not None,
        "model_name": model_name,
        "threshold": threshold,
        "model_path": str(MODEL_PATH),
        "raw_pipeline_ready": raw_pipeline is not None,
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "feature_artifact_path": str(FEATURE_ARTIFACT_PATH),
        "prediction_log_path": str(PREDICTION_LOG_PATH),
        "feedback_log_path": str(FEEDBACK_LOG_PATH),
        "inference_log_path": str(INFERENCE_LOG_FILE),
        "data_raw_dir": str(DATA_RAW_DIR),
        "metrics_path": "/metrics",
    }


@app.post("/download-data")
def download_data(request: DownloadDataRequest):
    try:
        destination = download_kaggle_dataset(
            competition=request.competition,
            dest_dir=DATA_RAW_DIR,
            force=request.force,
        )
        n_csv_files = len(list(destination.rglob("*.csv")))
        return {
            "status": "downloaded",
            "competition": request.competition,
            "destination": str(destination),
            "csv_files": n_csv_files,
            "force": request.force,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("Kaggle data download failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        if not request.records:
            raise HTTPException(status_code=400, detail="records must not be empty")

        if artifact is None or model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model artifact is unavailable. "
                    f"Missing or invalid artifact: {artifact_error}"
                ),
            )

        df_input = pd.DataFrame(request.records)
        X = prepare_features(df_input, artifact)
        probabilities = get_probabilities(model, X)
        predictions = (probabilities >= threshold).astype(int)
        record_prediction_metrics("/predict", probabilities, predictions)
        logged_records = X.to_dict(orient="records")
        logged_events = build_prediction_events(
            logged_records,
            probabilities,
            predictions,
            endpoint="/predict",
            model_name=model_name,
            threshold=threshold,
            model_ready=True,
        )

        try:
            append_jsonl(PREDICTION_LOG_PATH, logged_events)
        except Exception:
            logging.exception("Failed to write prediction logs")

        try:
            log_inference_data(
                request.records,
                probabilities,
                predictions,
                endpoint="/predict",
            )
        except Exception:
            logging.exception("Failed to write inference CSV logs")

        results = []
        for index, _row in enumerate(request.records):
            results.append(
                {
                    "index": index,
                    "request_id": logged_events[index]["request_id"],
                    "prediction_id": logged_events[index]["prediction_id"],
                    "fraud_probability": float(probabilities[index]),
                    "prediction": int(predictions[index]),
                    "threshold": threshold,
                    "model_name": model_name,
                }
            )

        return {
            "n_records": len(results),
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict_raw")
def predict_raw(request: RawPredictionRequest):
    try:
        if not request.records:
            raise HTTPException(status_code=400, detail="records must not be empty")

        if raw_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Raw prediction pipeline is unavailable. "
                    f"Missing or invalid artifact: {raw_pipeline_error}"
                ),
            )

        prepared_features = raw_pipeline.prepare_raw_features(
            request.records,
            context=request.context,
        )
        results = raw_pipeline.predict_feature_matrix(prepared_features)

        probabilities = np.array([row["fraud_probability"] for row in results], dtype=float)
        predictions = np.array([row["prediction"] for row in results], dtype=int)
        record_prediction_metrics("/predict_raw", probabilities, predictions)
        logged_events = build_prediction_events(
            prepared_features.to_dict(orient="records"),
            probabilities,
            predictions,
            endpoint="/predict_raw",
            model_name=raw_pipeline.model_name,
            threshold=raw_pipeline.threshold,
            model_ready=True,
        )

        try:
            append_jsonl(PREDICTION_LOG_PATH, logged_events)
        except Exception:
            logging.exception("Failed to write raw prediction logs")

        try:
            log_inference_data(request.records, probabilities, predictions, endpoint="/predict_raw")
        except Exception:
            logging.exception("Failed to write raw inference CSV logs")

        enriched_results = []
        for index, result in enumerate(results):
            risk_info = risk_engine.generate(result["fraud_probability"])
            enriched_results.append(
                {
                    **result,
                    "risk_score": risk_info["risk_score"],
                    "risk_level": risk_info["risk_level"],
                    "verification_required": risk_info["verification_required"],
                    "recommended_action": risk_info["recommended_action"],
                    "request_id": logged_events[index]["request_id"],
                    "prediction_id": logged_events[index]["prediction_id"],
                }
            )

        return {
            "n_records": len(enriched_results),
            "results": enriched_results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Raw prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    try:
        if not request.items:
            raise HTTPException(status_code=400, detail="items must not be empty")

        events = build_feedback_events([item.model_dump() for item in request.items])
        append_jsonl(FEEDBACK_LOG_PATH, events)

        return {
            "n_records": len(events),
            "feedback_log_path": str(FEEDBACK_LOG_PATH),
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Feedback logging failed")
        raise HTTPException(status_code=500, detail=str(e))


Instrumentator(
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, include_in_schema=False)

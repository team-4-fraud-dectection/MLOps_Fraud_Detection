import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
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
from src.validation import validate_feature_matrix, validate_model_artifact
from src.risk_score import RiskScoringEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "models/preprocessor_v1.pkl"))
FEATURE_ARTIFACT_PATH = Path(os.getenv("FEATURE_ARTIFACT_PATH", "artifacts/fe_artifact.pkl"))


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

risk_engine = RiskScoringEngine()

artifact, artifact_error = init_model_artifact()
model = artifact["model"] if artifact else None
model_name = artifact["model_name"] if artifact else "unavailable"
threshold = float(artifact["threshold"]) if artifact else None

app = FastAPI(
    title="IEEE Fraud Detection API",
    version="1.0.0",
    description="Live inference API for fraud probability prediction."
)


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


raw_pipeline, raw_pipeline_error = init_raw_pipeline()


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
        "metrics_path": "/metrics",
    }


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

        results = []
        for i, row in enumerate(request.records):
            results.append({
                "index": i,
                "request_id": logged_events[i]["request_id"],
                "prediction_id": logged_events[i]["prediction_id"],
                "fraud_probability": float(probabilities[i]),
                "prediction": int(predictions[i]),
                "threshold": threshold,
                "model_name": model_name,
            })

        return {
            "n_records": len(results),
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


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


# Expose Prometheus-compatible service metrics for Kubernetes monitoring.
Instrumentator(
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, include_in_schema=False)

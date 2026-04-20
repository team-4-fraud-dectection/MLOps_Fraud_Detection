import logging
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
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
from src.risk_score import RiskScoringEngine
from src.validation import validate_feature_matrix, validate_model_artifact

try:
    import kagglehub
except ImportError:  # pragma: no cover - optional dependency in some environments
    kagglehub = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "models/preprocessor_v1.pkl"))
FEATURE_ARTIFACT_PATH = Path(os.getenv("FEATURE_ARTIFACT_PATH", "artifacts/fe_artifact.pkl"))
KAGGLE_COMPETITION = os.getenv("KAGGLE_COMPETITION", "ieee-fraud-detection")
DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
INFERENCE_LOG_FILE = Path(os.getenv("INFERENCE_LOG_FILE", "logs/inference_history.csv"))
INFERENCE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
AUTO_DOWNLOAD_KAGGLE_DATA = os.getenv("AUTO_DOWNLOAD_KAGGLE_DATA", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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
    competition: str = Field(default=KAGGLE_COMPETITION)
    force: bool = False


def download_kaggle_dataset(
    competition: str = KAGGLE_COMPETITION,
    dest_dir: Path = DATA_RAW_DIR,
    force: bool = False,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing_csv_files = list(dest_dir.rglob("*.csv"))
    if existing_csv_files and not force:
        logger.info(
            "Kaggle raw data already exists at %s (%d csv files).",
            dest_dir,
            len(existing_csv_files),
        )
        return dest_dir

    if kagglehub is None:
        raise RuntimeError("kagglehub is not installed in the current environment.")

    logger.info("Downloading Kaggle competition data for '%s' ...", competition)
    kaggle_cache_path = Path(kagglehub.competition_download(competition))

    for src_file in kaggle_cache_path.rglob("*"):
        if src_file.is_file():
            relative = src_file.relative_to(kaggle_cache_path)
            dst_file = dest_dir / relative
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

    logger.info("Kaggle data synced to %s", dest_dir)
    return dest_dir


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


def refresh_runtime_state() -> None:
    global artifact, artifact_error, model, model_name, threshold, raw_pipeline, raw_pipeline_error

    artifact, artifact_error = init_model_artifact()
    model = artifact["model"] if artifact else None
    model_name = artifact["model_name"] if artifact else "unavailable"
    threshold = float(artifact["threshold"]) if artifact else None
    raw_pipeline, raw_pipeline_error = init_raw_pipeline()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if AUTO_DOWNLOAD_KAGGLE_DATA:
        try:
            download_kaggle_dataset(force=False)
        except Exception as exc:  # pragma: no cover - startup network/auth dependent
            logger.warning("Automatic Kaggle download skipped: %s", exc)

    refresh_runtime_state()
    yield


app = FastAPI(
    title="IEEE Fraud Detection API",
    version="1.0.0",
    description="Live inference API for fraud probability prediction.",
    lifespan=lifespan,
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
        "kaggle_competition": KAGGLE_COMPETITION,
        "metrics_path": "/metrics",
    }


@app.post("/download-data")
def download_data(request: DownloadDataRequest):
    try:
        destination = download_kaggle_dataset(
            competition=request.competition,
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


# Expose Prometheus-compatible service metrics for Kubernetes monitoring.
Instrumentator(
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, include_in_schema=False)

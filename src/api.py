import logging
from pathlib import Path
from typing import Any
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

MODEL_PATH = Path("models/model.pkl")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True) 
INFERENCE_LOG_FILE = LOG_DIR / "inference_history.csv"

class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="List of feature dictionaries")


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

    required_keys = {"model_name", "model", "threshold"}
    missing_keys = required_keys - set(artifact.keys())
    if missing_keys:
        raise ValueError(f"Model artifact missing keys: {missing_keys}")

    return artifact


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

    return X


def get_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    raise ValueError(f"Model {type(model).__name__} does not support probability-like outputs.")

def log_inference_data(input_records: list, probabilities: np.ndarray, predictions: np.ndarray):
    df_log = pd.DataFrame(input_records)
    df_log['fraud_probability'] = probabilities
    df_log['prediction'] = predictions
    df_log['timestamp'] = datetime.now()
    
    header = not INFERENCE_LOG_FILE.exists()
    df_log.to_csv(INFERENCE_LOG_FILE, mode='a', header=header, index=False)


artifact = load_artifact(MODEL_PATH)
model = artifact["model"]
model_name = artifact["model_name"]
threshold = float(artifact["threshold"])

app = FastAPI(
    title="IEEE Fraud Detection API",
    version="1.0.0",
    description="Live inference API for fraud probability prediction."
)

Instrumentator().instrument(app).expose(app)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": model_name,
        "threshold": threshold,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        if not request.records:
            raise HTTPException(status_code=400, detail="records must not be empty")

        df_input = pd.DataFrame(request.records)
        X = prepare_features(df_input, artifact)
        probabilities = get_probabilities(model, X)
        predictions = (probabilities >= threshold).astype(int)

        log_inference_data(request.records, probabilities, predictions)

        results = []
        for i, row in enumerate(request.records):
            results.append({
                "index": i,
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
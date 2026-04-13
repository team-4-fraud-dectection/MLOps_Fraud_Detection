import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run fraud inference with champion model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model.pkl",
        help="Path to saved champion model artifact.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_path",
        type=str,
        help="Path to input CSV or Parquet file containing features only.",
    )
    input_group.add_argument(
        "--input_json",
        type=str,
        help="JSON string for a single record or a list of records.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save predictions as CSV or JSON.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of rows to print in console preview.",
    )
    return parser.parse_args()


def load_artifact(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)

    required_keys = {"model_name", "model", "threshold"}
    missing_keys = required_keys - set(artifact.keys())
    if missing_keys:
        raise ValueError(f"Model artifact missing keys: {missing_keys}")

    return artifact


def load_input_data(input_path: str | None, input_json: str | None) -> pd.DataFrame:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(path)
        else:
            raise ValueError("Unsupported input file format. Use CSV or Parquet.")
        return df

    parsed = json.loads(input_json)
    if isinstance(parsed, dict):
        return pd.DataFrame([parsed])
    if isinstance(parsed, list):
        return pd.DataFrame(parsed)

    raise ValueError("--input_json must be a JSON object or a list of JSON objects.")


def sanitize_feature_name(name: str) -> str:
    import re

    name = str(name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


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


def build_output(df_input: pd.DataFrame, probabilities: np.ndarray, threshold: float, model_name: str) -> pd.DataFrame:
    results = df_input.copy()
    results["fraud_probability"] = probabilities
    results["prediction"] = (probabilities >= threshold).astype(int)
    results["threshold"] = threshold
    results["model_name"] = model_name
    return results


def save_output(df_output: pd.DataFrame, output_path: str | None) -> None:
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df_output.to_csv(path, index=False)
    elif path.suffix.lower() == ".json":
        path.write_text(df_output.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        raise ValueError("Unsupported output format. Use .csv or .json")

    logging.info("Saved predictions to %s", path)


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    artifact = load_artifact(model_path)

    model = artifact["model"]
    model_name = artifact["model_name"]
    threshold = float(artifact["threshold"])

    logging.info("Loaded model artifact from %s", model_path)
    logging.info("Model name: %s", model_name)
    logging.info("Decision threshold: %.4f", threshold)

    df_input = load_input_data(args.input_path, args.input_json)
    logging.info("Input shape: %s", df_input.shape)

    X = prepare_features(df_input, artifact)
    logging.info("Prepared feature shape: %s", X.shape)

    probabilities = get_probabilities(model, X)
    df_output = build_output(df_input, probabilities, threshold, model_name)

    preview = df_output.head(args.top_k)
    print(preview[["fraud_probability", "prediction", "threshold", "model_name"]])

    save_output(df_output, args.output_path)


if __name__ == "__main__":
    main()
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a held-out feature dataset through the live API to generate "
            "larger prediction and feedback logs for monitoring demos."
        )
    )
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", type=str, default="/predict")
    parser.add_argument("--x_path", type=str, default="data/featured/X_val.parquet")
    parser.add_argument("--y_path", type=str, default="data/featured/y_val.parquet")
    parser.add_argument("--prediction_log_path", type=str, default="logs/predictions.jsonl")
    parser.add_argument("--feedback_log_path", type=str, default="logs/prediction_feedback.jsonl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_records", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--feedback_source", type=str, default="replay_x_val")
    parser.add_argument("--timeout_seconds", type=float, default=60.0)
    parser.add_argument("--reset_logs", action="store_true")
    return parser.parse_args()


def load_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)

    raise ValueError(f"Unsupported replay input format: {source}")


def ensure_label_series(y_frame: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(y_frame, pd.Series):
        return y_frame.reset_index(drop=True)

    if y_frame.shape[1] != 1:
        raise ValueError("Label input must contain exactly one column.")

    return y_frame.iloc[:, 0].reset_index(drop=True)


def select_replay_rows(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    max_records: int,
    sample_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(features) != len(labels):
        raise ValueError("Feature and label inputs must have the same number of rows.")

    if max_records <= 0 or max_records >= len(features):
        return features.reset_index(drop=True), labels.reset_index(drop=True)

    sampled_index = features.sample(n=max_records, random_state=sample_seed).index
    sampled_features = features.loc[sampled_index].reset_index(drop=True)
    sampled_labels = labels.loc[sampled_index].reset_index(drop=True)
    return sampled_features, sampled_labels


def json_ready_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None

    if pd.isna(value):
        return None

    return value


def dataframe_to_request_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({str(key): json_ready_value(value) for key, value in row.items()})
    return records


def build_feedback_items(
    prediction_results: list[dict[str, Any]],
    labels: pd.Series | list[Any],
    *,
    feedback_source: str,
) -> list[dict[str, Any]]:
    if len(prediction_results) != len(labels):
        raise ValueError("Prediction results and labels must have the same batch size.")

    items = []
    for result, label in zip(prediction_results, labels):
        items.append(
            {
                "prediction_id": result["prediction_id"],
                "request_id": result.get("request_id"),
                "actual_label": int(label),
                "feedback_source": feedback_source,
            }
        )
    return items


def reset_log_file(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()


def join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def replay_batches(
    client: httpx.Client,
    *,
    base_url: str,
    endpoint: str,
    records: list[dict[str, Any]],
    labels: pd.Series,
    batch_size: int,
    feedback_source: str,
) -> dict[str, Any]:
    total_predictions = 0
    total_feedback = 0
    predicted_positive = 0

    predict_url = join_url(base_url, endpoint)
    feedback_url = join_url(base_url, "/feedback")

    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        response = client.post(predict_url, json={"records": records[start:end]})
        response.raise_for_status()

        payload = response.json()
        results = payload.get("results", [])
        if len(results) != (end - start):
            raise RuntimeError(
                f"API returned {len(results)} prediction results for a batch of {end - start} rows."
            )

        feedback_items = build_feedback_items(
            results,
            labels.iloc[start:end].tolist(),
            feedback_source=feedback_source,
        )
        feedback_response = client.post(feedback_url, json={"items": feedback_items})
        feedback_response.raise_for_status()

        total_predictions += len(results)
        total_feedback += len(feedback_items)
        predicted_positive += sum(int(result["prediction"]) for result in results)

        print(f"Replayed rows {start + 1}-{end} of {len(records)}")

    return {
        "replayed_records": total_predictions,
        "feedback_records": total_feedback,
        "predicted_positive_rate": (predicted_positive / total_predictions) if total_predictions else None,
    }


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")

    features = load_frame(args.x_path)
    labels = ensure_label_series(load_frame(args.y_path))
    features, labels = select_replay_rows(
        features,
        labels,
        max_records=args.max_records,
        sample_seed=args.sample_seed,
    )
    records = dataframe_to_request_records(features)

    if args.reset_logs:
        reset_log_file(args.prediction_log_path)
        reset_log_file(args.feedback_log_path)

    with httpx.Client(timeout=args.timeout_seconds) as client:
        health_response = client.get(join_url(args.base_url, "/health"))
        health_response.raise_for_status()
        health_payload = health_response.json()
        if not health_payload.get("model_ready"):
            raise RuntimeError("API health check succeeded but the model is not ready for /predict.")

        summary = replay_batches(
            client,
            base_url=args.base_url,
            endpoint=args.endpoint,
            records=records,
            labels=labels,
            batch_size=args.batch_size,
            feedback_source=args.feedback_source,
        )

    summary.update(
        {
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "x_path": args.x_path,
            "y_path": args.y_path,
            "prediction_log_path": args.prediction_log_path,
            "feedback_log_path": args.feedback_log_path,
            "sampled_records": int(len(records)),
            "observed_positive_rate": float(labels.mean()) if len(labels) else None,
            "sample_seed": int(args.sample_seed),
        }
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

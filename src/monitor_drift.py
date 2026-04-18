import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import extract_feature_frame, load_prediction_dataframe  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an Evidently data drift report.")
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--current_path", type=str, default="")
    parser.add_argument("--prediction_log_path", type=str, default="logs/predictions.jsonl")
    parser.add_argument("--endpoint", type=str, default="/predict")
    parser.add_argument("--output_dir", type=str, default="reports/drift")
    parser.add_argument("--drift_share_threshold", type=float, default=0.5)
    parser.add_argument("--min_current_records", type=int, default=30)
    parser.add_argument("--max_current_records", type=int, default=0)
    return parser.parse_args()


def load_tabular_data(path: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(source, lines=(suffix == ".jsonl"))

    raise ValueError(f"Unsupported input format for drift report: {source}")


def import_evidently():
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        return Report, DataDriftPreset
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Evidently could not be imported in this environment. "
            "Use Python 3.11 or run this script via the project's CI/Docker environment."
        ) from exc


def main():
    args = parse_args()
    Report, DataDriftPreset = import_evidently()

    reference_df = load_tabular_data(args.reference_path)
    if args.current_path:
        current_df = load_tabular_data(args.current_path)
    else:
        prediction_df = load_prediction_dataframe(args.prediction_log_path, endpoint=args.endpoint)
        current_df = extract_feature_frame(prediction_df)

    if current_df.empty:
        raise ValueError("Current dataset is empty. Generate predictions first or provide --current_path.")

    if args.max_current_records > 0:
        current_df = current_df.tail(args.max_current_records).reset_index(drop=True)

    if len(current_df) < args.min_current_records:
        raise ValueError(
            f"Current dataset has only {len(current_df)} rows; "
            f"need at least {args.min_current_records} rows "
            "for a stable drift report."
        )

    reference_df = reference_df.drop(columns=["isFraud"], errors="ignore")
    current_df = current_df.drop(columns=["isFraud"], errors="ignore")

    common_columns = [column for column in reference_df.columns if column in current_df.columns]
    if not common_columns:
        raise ValueError("No overlapping columns found between reference and current datasets.")

    numeric_reference = {}
    numeric_current = {}
    dropped_non_numeric = []
    for column in common_columns:
        ref_series = pd.to_numeric(reference_df[column], errors="coerce")
        cur_series = pd.to_numeric(current_df[column], errors="coerce")
        if ref_series.notna().sum() == 0 or cur_series.notna().sum() == 0:
            dropped_non_numeric.append(column)
            continue
        numeric_reference[column] = ref_series.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
        numeric_current[column] = cur_series.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

    if not numeric_reference:
        raise ValueError(
            "No numeric-compatible columns remain after aligning reference and current datasets."
        )

    reference_df = pd.DataFrame(numeric_reference).astype("float32")
    current_df = pd.DataFrame(numeric_current).astype("float32")

    report = Report(
        [DataDriftPreset(drift_share=args.drift_share_threshold)],
        include_tests=True,
    )
    snapshot = report.run(current_df, reference_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "data_drift_report.html"
    json_path = output_dir / "data_drift_report.json"
    summary_path = output_dir / "data_drift_summary.json"

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    report_dict = snapshot.dict()
    drift_metric = next(
        (
            metric
            for metric in report_dict.get("metrics", [])
            if metric.get("metric_name", "").startswith("DriftedColumnsCount")
        ),
        {},
    )
    drift_value = drift_metric.get("value", {})

    summary = {
        "reference_path": args.reference_path,
        "current_path": args.current_path or args.prediction_log_path,
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "monitored_columns": list(reference_df.columns),
        "dropped_non_numeric_columns": dropped_non_numeric,
        "drift_share_threshold": float(args.drift_share_threshold),
        "drifted_columns_count": int(drift_value.get("count", 0) or 0),
        "drifted_columns_share": float(drift_value.get("share", 0.0) or 0.0),
        "dataset_drift_detected": bool((drift_value.get("share", 0.0) or 0.0) >= args.drift_share_threshold),
        "html_report_path": str(html_path),
        "json_report_path": str(json_path),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

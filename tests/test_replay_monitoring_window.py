import numpy as np
import pandas as pd

from src.replay_monitoring_window import (
    build_feedback_items,
    dataframe_to_request_records,
    select_replay_rows,
)


def test_dataframe_to_request_records_converts_numpy_scalars_and_missing_values():
    frame = pd.DataFrame(
        {
            "feature_a": np.array([np.int64(7)]),
            "feature_b": [np.float32(0.25)],
            "feature_c": [np.nan],
        }
    )

    records = dataframe_to_request_records(frame)

    assert records == [{"feature_a": 7, "feature_b": 0.25, "feature_c": None}]


def test_select_replay_rows_keeps_feature_label_alignment():
    features = pd.DataFrame({"feature_a": [10, 20, 30, 40, 50]})
    labels = pd.Series([100, 200, 300, 400, 500], name="isFraud")

    sampled_features, sampled_labels = select_replay_rows(
        features,
        labels,
        max_records=3,
        sample_seed=42,
    )

    assert len(sampled_features) == 3
    assert len(sampled_labels) == 3
    assert sampled_labels.tolist() == [200, 500, 300]
    assert sampled_features["feature_a"].tolist() == [20, 50, 30]


def test_build_feedback_items_uses_prediction_ids_and_labels():
    prediction_results = [
        {"prediction_id": "req-1:0", "request_id": "req-1"},
        {"prediction_id": "req-1:1", "request_id": "req-1"},
    ]

    items = build_feedback_items(
        prediction_results,
        [0, 1],
        feedback_source="replay_x_val",
    )

    assert items == [
        {
            "prediction_id": "req-1:0",
            "request_id": "req-1",
            "actual_label": 0,
            "feedback_source": "replay_x_val",
        },
        {
            "prediction_id": "req-1:1",
            "request_id": "req-1",
            "actual_label": 1,
            "feedback_source": "replay_x_val",
        },
    ]

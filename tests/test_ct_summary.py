from src.ct_summary import (
    compare_snapshots,
    finalize_ct_summary,
    initialize_ct_summary,
    mark_ct_summary,
)


def sample_train_metrics(model_name="CatBoost", model_version="1", tuned_f1=0.49, tuned_auprc=0.50):
    return {
        "best_model_name": model_name,
        "best_model_default_metrics": {
            "f1": 0.45,
            "auprc": 0.48,
            "precision": 0.40,
            "recall": 0.52,
            "threshold": 0.5,
        },
        "best_model_tuned_metrics": {
            "f1": tuned_f1,
            "auprc": tuned_auprc,
            "precision": 0.63,
            "recall": 0.40,
            "threshold": 0.74,
        },
        "training_setup": {
            "selection_metric": "auprc",
            "threshold_tuning_metric": "f1",
            "train_samples": 1000,
            "val_samples": 200,
            "feature_count": 32,
            "n_trials": 10,
            "tuned_models": ["CatBoost", "XGBoost"],
        },
        "registered_model": {
            "model_name": "fraud_detection_model",
            "model_version": model_version,
            "run_id": f"run-{model_version}",
            "model_uri": f"models:/fraud_detection_model/{model_version}",
            "logged_model_id": f"logged-{model_version}",
        },
    }


def test_initialize_ct_summary_marks_pending_when_retrain_is_requested():
    summary = initialize_ct_summary(
        ct_decision={"should_retrain": True, "reasons": ["performance_below_threshold"]},
        status_summary={"current_f1": 0.41},
        train_metrics=sample_train_metrics(),
        decision_path="reports/monitoring/ct_decision.json",
        status_summary_path="reports/monitoring/status_summary.json",
        train_metrics_path="metrics/train_metrics.json",
    )

    assert summary["run_status"] == "pending_retrain"
    assert summary["decision"]["should_retrain"] is True
    assert summary["before_retrain"]["registered_model"]["model_version"] == "1"
    assert summary["model_version_trace"][0]["stage"] == "before_retrain"


def test_finalize_ct_summary_records_before_after_comparison():
    initial = initialize_ct_summary(
        ct_decision={"should_retrain": True, "reasons": ["performance_below_threshold"]},
        status_summary={"current_f1": 0.41},
        train_metrics=sample_train_metrics(model_version="1", tuned_f1=0.49, tuned_auprc=0.50),
        decision_path="reports/monitoring/ct_decision.json",
        status_summary_path="reports/monitoring/status_summary.json",
        train_metrics_path="metrics/train_metrics.json",
    )

    finalized = finalize_ct_summary(
        initial,
        train_metrics=sample_train_metrics(model_version="2", tuned_f1=0.53, tuned_auprc=0.54),
        train_metrics_path="metrics/train_metrics.json",
    )

    assert finalized["run_status"] == "completed"
    assert finalized["after_retrain"]["registered_model"]["model_version"] == "2"
    assert finalized["comparison"]["registered_model_version_before"] == "1"
    assert finalized["comparison"]["registered_model_version_after"] == "2"
    assert finalized["comparison"]["model_version_incremented"] is True
    assert round(finalized["comparison"]["tuned_f1_delta"], 6) == 0.04
    assert len(finalized["model_version_trace"]) == 2


def test_compare_snapshots_detects_model_name_changes():
    before = initialize_ct_summary(
        ct_decision={"should_retrain": True},
        status_summary={},
        train_metrics=sample_train_metrics(model_name="CatBoost", model_version="1"),
        decision_path="reports/monitoring/ct_decision.json",
        status_summary_path="reports/monitoring/status_summary.json",
        train_metrics_path="metrics/train_metrics.json",
    )["before_retrain"]

    after = initialize_ct_summary(
        ct_decision={"should_retrain": True},
        status_summary={},
        train_metrics=sample_train_metrics(model_name="LightGBM", model_version="2"),
        decision_path="reports/monitoring/ct_decision.json",
        status_summary_path="reports/monitoring/status_summary.json",
        train_metrics_path="metrics/train_metrics.json",
    )["before_retrain"]

    comparison = compare_snapshots(before, after)

    assert comparison["best_model_changed"] is True
    assert comparison["model_version_incremented"] is True


def test_mark_ct_summary_records_terminal_status_and_notes():
    summary = initialize_ct_summary(
        ct_decision={"should_retrain": True},
        status_summary={},
        train_metrics=sample_train_metrics(),
        decision_path="reports/monitoring/ct_decision.json",
        status_summary_path="reports/monitoring/status_summary.json",
        train_metrics_path="metrics/train_metrics.json",
    )

    updated = mark_ct_summary(
        summary,
        run_status="dry_run",
        note="Command generated without executing training.",
    )

    assert updated["run_status"] == "dry_run"
    assert updated["notes"] == ["Command generated without executing training."]
    assert "completed_at" in updated

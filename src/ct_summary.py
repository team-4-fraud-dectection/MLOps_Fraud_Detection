import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPARISON_METRIC_PATHS = {
    "tuned_f1_delta": ("tuned_metrics", "f1"),
    "tuned_auprc_delta": ("tuned_metrics", "auprc"),
    "tuned_precision_delta": ("tuned_metrics", "precision"),
    "tuned_recall_delta": ("tuned_metrics", "recall"),
    "default_f1_delta": ("default_metrics", "f1"),
    "default_auprc_delta": ("default_metrics", "auprc"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build before/after continuous-training summaries with model version trace."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--decision-path", required=True)
    init_parser.add_argument("--status-summary-path", required=True)
    init_parser.add_argument("--train-metrics-path", required=True)
    init_parser.add_argument("--output-path", required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--summary-path", required=True)
    finalize_parser.add_argument("--train-metrics-path", required=True)
    finalize_parser.add_argument("--output-path", required=True)

    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("--summary-path", required=True)
    mark_parser.add_argument("--output-path", required=True)
    mark_parser.add_argument("--run-status", required=True)
    mark_parser.add_argument("--note", default="")

    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_training_snapshot(train_metrics: dict[str, Any], *, source_path: str | Path) -> dict[str, Any]:
    default_metrics = train_metrics.get("best_model_default_metrics") or {}
    tuned_metrics = train_metrics.get("best_model_tuned_metrics") or {}
    registered_model = train_metrics.get("registered_model") or {}
    training_setup = train_metrics.get("training_setup") or {}

    return {
        "snapshot_taken_at": utc_now(),
        "source_path": str(source_path),
        "available": bool(train_metrics),
        "best_model_name": train_metrics.get("best_model_name"),
        "training_setup": {
            "selection_metric": training_setup.get("selection_metric"),
            "threshold_tuning_metric": training_setup.get("threshold_tuning_metric"),
            "train_samples": training_setup.get("train_samples"),
            "val_samples": training_setup.get("val_samples"),
            "feature_count": training_setup.get("feature_count"),
            "n_trials": training_setup.get("n_trials"),
            "tuned_models": training_setup.get("tuned_models"),
        },
        "default_metrics": {
            "auprc": default_metrics.get("auprc"),
            "recall": default_metrics.get("recall"),
            "precision": default_metrics.get("precision"),
            "f1": default_metrics.get("f1"),
            "threshold": default_metrics.get("threshold"),
        },
        "tuned_metrics": {
            "auprc": tuned_metrics.get("auprc"),
            "recall": tuned_metrics.get("recall"),
            "precision": tuned_metrics.get("precision"),
            "f1": tuned_metrics.get("f1"),
            "threshold": tuned_metrics.get("threshold"),
        },
        "registered_model": {
            "model_name": registered_model.get("model_name"),
            "model_version": registered_model.get("model_version"),
            "run_id": registered_model.get("run_id"),
            "model_uri": registered_model.get("model_uri"),
            "logged_model_id": registered_model.get("logged_model_id"),
        },
    }


def _value_at_path(payload: dict[str, Any], path: tuple[str, str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _safe_delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_model_version_trace(
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []

    def append_entry(stage: str, snapshot: dict[str, Any] | None) -> None:
        if not snapshot:
            return
        registered_model = snapshot.get("registered_model") or {}
        trace.append(
            {
                "stage": stage,
                "best_model_name": snapshot.get("best_model_name"),
                "model_name": registered_model.get("model_name"),
                "model_version": registered_model.get("model_version"),
                "run_id": registered_model.get("run_id"),
                "model_uri": registered_model.get("model_uri"),
                "logged_model_id": registered_model.get("logged_model_id"),
            }
        )

    append_entry("before_retrain", before_snapshot)
    append_entry("after_retrain", after_snapshot)
    return trace


def compare_snapshots(before_snapshot: dict[str, Any], after_snapshot: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "best_model_before": before_snapshot.get("best_model_name"),
        "best_model_after": after_snapshot.get("best_model_name"),
        "best_model_changed": before_snapshot.get("best_model_name") != after_snapshot.get("best_model_name"),
    }

    before_registered = before_snapshot.get("registered_model") or {}
    after_registered = after_snapshot.get("registered_model") or {}
    before_version = before_registered.get("model_version")
    after_version = after_registered.get("model_version")

    comparison["registered_model_version_before"] = before_version
    comparison["registered_model_version_after"] = after_version

    before_version_int = _safe_int(before_version)
    after_version_int = _safe_int(after_version)
    comparison["model_version_incremented"] = (
        after_version_int > before_version_int
        if before_version_int is not None and after_version_int is not None
        else None
    )

    for output_name, path in COMPARISON_METRIC_PATHS.items():
        before_value = _value_at_path(before_snapshot, path)
        after_value = _value_at_path(after_snapshot, path)
        comparison[output_name] = _safe_delta(before_value, after_value)

    return comparison


def initialize_ct_summary(
    *,
    ct_decision: dict[str, Any],
    status_summary: dict[str, Any],
    train_metrics: dict[str, Any],
    decision_path: str | Path,
    status_summary_path: str | Path,
    train_metrics_path: str | Path,
) -> dict[str, Any]:
    before_snapshot = extract_training_snapshot(train_metrics, source_path=train_metrics_path)
    should_retrain = bool(ct_decision.get("should_retrain", False))

    summary = {
        "generated_at": utc_now(),
        "run_status": "pending_retrain" if should_retrain else "skipped",
        "decision": ct_decision,
        "decision_path": str(decision_path),
        "monitoring_snapshot": status_summary,
        "status_summary_path": str(status_summary_path),
        "before_retrain": before_snapshot,
        "after_retrain": None,
        "comparison": None,
        "model_version_trace": build_model_version_trace(before_snapshot),
    }

    if not should_retrain:
        summary["completed_at"] = utc_now()

    return summary


def finalize_ct_summary(
    summary: dict[str, Any],
    *,
    train_metrics: dict[str, Any],
    train_metrics_path: str | Path,
) -> dict[str, Any]:
    updated = dict(summary)
    before_snapshot = updated.get("before_retrain") or {}
    after_snapshot = extract_training_snapshot(train_metrics, source_path=train_metrics_path)
    updated["after_retrain"] = after_snapshot
    updated["comparison"] = compare_snapshots(before_snapshot, after_snapshot)
    updated["model_version_trace"] = build_model_version_trace(before_snapshot, after_snapshot)
    updated["run_status"] = "completed"
    updated["completed_at"] = utc_now()
    return updated


def mark_ct_summary(
    summary: dict[str, Any],
    *,
    run_status: str,
    note: str = "",
) -> dict[str, Any]:
    updated = dict(summary)
    updated["run_status"] = run_status
    if note:
        notes = list(updated.get("notes") or [])
        notes.append(note)
        updated["notes"] = notes
    updated["completed_at"] = utc_now()
    return updated


def main():
    args = parse_args()

    if args.command == "init":
        ct_decision = load_json(args.decision_path)
        status_summary = load_json(args.status_summary_path)
        train_metrics = load_json(args.train_metrics_path)
        payload = initialize_ct_summary(
            ct_decision=ct_decision,
            status_summary=status_summary,
            train_metrics=train_metrics,
            decision_path=args.decision_path,
            status_summary_path=args.status_summary_path,
            train_metrics_path=args.train_metrics_path,
        )
        write_json(args.output_path, payload)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "finalize":
        summary = load_json(args.summary_path)
        train_metrics = load_json(args.train_metrics_path)
        payload = finalize_ct_summary(
            summary,
            train_metrics=train_metrics,
            train_metrics_path=args.train_metrics_path,
        )
        write_json(args.output_path, payload)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "mark":
        summary = load_json(args.summary_path)
        payload = mark_ct_summary(
            summary,
            run_status=args.run_status,
            note=args.note,
        )
        write_json(args.output_path, payload)
        print(json.dumps(payload, indent=2))
        return


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether continuous training should run based on the "
            "latest monitoring status summary."
        )
    )
    parser.add_argument(
        "--status-summary-path",
        type=str,
        default="reports/monitoring/status_summary.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="reports/monitoring/ct_decision.json",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Override monitoring status and force a retraining decision.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def normalize_reasons(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def evaluate_ct_trigger(
    status_summary: dict[str, Any] | None = None,
    *,
    force_retrain: bool = False,
) -> dict[str, Any]:
    status_summary = status_summary or {}
    status_available = bool(status_summary)
    status_should_retrain = bool(status_summary.get("should_retrain", False))
    status_reasons = normalize_reasons(status_summary.get("reasons"))

    if force_retrain:
        reasons = ["manual_override"]
        for reason in status_reasons:
            if reason not in reasons:
                reasons.append(reason)
        should_retrain = True
        trigger_source = "manual_override"
    elif not status_available:
        reasons = ["missing_status_summary"]
        should_retrain = False
        trigger_source = "missing_status_summary"
    elif status_should_retrain:
        reasons = status_reasons or ["status_summary_requested_retraining"]
        should_retrain = True
        trigger_source = "status_summary"
    else:
        reasons = ["monitoring_within_thresholds"]
        should_retrain = False
        trigger_source = "status_summary"

    return {
        "status_available": status_available,
        "status_should_retrain": status_should_retrain,
        "force_retrain": force_retrain,
        "should_retrain": should_retrain,
        "trigger_source": trigger_source,
        "reasons": reasons,
        "current_f1": status_summary.get("current_f1"),
        "current_drift_share": status_summary.get("current_drift_share"),
        "performance_f1_threshold": status_summary.get("performance_f1_threshold"),
        "drift_share_threshold": status_summary.get("drift_share_threshold"),
    }


def main():
    args = parse_args()
    status_summary = load_json(args.status_summary_path)
    decision = evaluate_ct_trigger(
        status_summary,
        force_retrain=args.force_retrain,
    )
    decision["status_summary_path"] = str(args.status_summary_path)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

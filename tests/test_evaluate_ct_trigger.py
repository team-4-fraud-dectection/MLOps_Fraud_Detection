from src.evaluate_ct_trigger import evaluate_ct_trigger


def test_evaluate_ct_trigger_uses_status_summary_signal():
    decision = evaluate_ct_trigger(
        {
            "should_retrain": True,
            "reasons": ["performance_below_threshold"],
            "current_f1": 0.4,
        }
    )

    assert decision["should_retrain"] is True
    assert decision["trigger_source"] == "status_summary"
    assert decision["reasons"] == ["performance_below_threshold"]
    assert decision["current_f1"] == 0.4


def test_evaluate_ct_trigger_skips_when_monitoring_is_healthy():
    decision = evaluate_ct_trigger(
        {
            "should_retrain": False,
            "reasons": [],
            "current_f1": 0.85,
            "current_drift_share": 0.1,
        }
    )

    assert decision["should_retrain"] is False
    assert decision["trigger_source"] == "status_summary"
    assert decision["reasons"] == ["monitoring_within_thresholds"]


def test_evaluate_ct_trigger_supports_manual_override():
    decision = evaluate_ct_trigger(
        {
            "should_retrain": False,
            "reasons": [],
        },
        force_retrain=True,
    )

    assert decision["should_retrain"] is True
    assert decision["trigger_source"] == "manual_override"
    assert "manual_override" in decision["reasons"]

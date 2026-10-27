import pandas as pd

from churn_project.data import FEATURES, make_churn_dataset
from churn_project.monitor import build_drift_report, calculate_numeric_psi


def test_identical_distributions_have_low_psi():
    series = pd.Series([1, 2, 3, 4, 5] * 50)
    assert calculate_numeric_psi(series, series.copy()) < 1e-6


def test_shift_monitor_flags_at_least_one_feature():
    baseline = make_churn_dataset(n_samples=1200, random_state=42)[FEATURES]
    current = make_churn_dataset(n_samples=1200, random_state=2026, shift=True)[FEATURES]
    report = build_drift_report(baseline, current, threshold=0.2)
    assert report["status"] == "DRIFT_DETECTED"
    assert len(report["drifted_features"]) >= 1


def test_monitor_requires_expected_columns():
    try:
        build_drift_report(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}))
    except ValueError as exc:
        assert "Required feature columns missing" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing features")

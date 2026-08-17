from __future__ import annotations

import pandas as pd

from home_credit_mlops.monitoring.drift import (
    DriftConfig,
    compute_categorical_drift,
    compute_numeric_drift,
    summarize_drift,
)


def test_compute_numeric_drift_detects_shifted_distribution() -> None:
    reference = pd.DataFrame({"AMT_INCOME_TOTAL": [100, 120, 130, 140, 150, 160]})
    current = pd.DataFrame({"AMT_INCOME_TOTAL": [500, 520, 530, 540]})

    drift = compute_numeric_drift(
        reference,
        current,
        config=DriftConfig(bins=3, min_current_rows=1),
    )

    assert drift.loc[0, "feature"] == "AMT_INCOME_TOTAL"
    assert drift.loc[0, "psi"] > 0
    assert drift.loc[0, "drift_level"] in {"moderate", "high"}


def test_compute_categorical_drift_handles_new_categories() -> None:
    reference = pd.DataFrame({"CODE_GENDER": ["F", "F", "M", "M"]})
    current = pd.DataFrame({"CODE_GENDER": ["XNA", "XNA", "F"]})

    drift = compute_categorical_drift(
        reference,
        current,
        config=DriftConfig(min_current_rows=1),
    )

    assert drift.loc[0, "feature"] == "CODE_GENDER"
    assert drift.loc[0, "psi"] > 0


def test_summarize_drift_counts_alert_levels() -> None:
    drift = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "psi": [0.30, 0.05],
            "drift_level": ["high", "low"],
        }
    )

    summary = summarize_drift(drift, pd.DataFrame())

    assert summary.loc[0, "feature_count"] == 2
    assert summary.loc[0, "high_drift_count"] == 1

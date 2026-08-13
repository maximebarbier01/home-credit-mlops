from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit_mlops.fairness.metrics import (
    build_age_bands,
    compute_disparity_summary,
    compute_fairness_table,
    compute_group_metrics,
)
from home_credit_mlops.modeling.metrics import business_cost


def test_build_age_bands_assigns_expected_labels() -> None:
    ages = pd.Series([25, 35, 65])

    bands = build_age_bands(ages)

    assert bands.astype(str).tolist() == ["20-29", "30-39", "60+"]


def test_compute_group_metrics_matches_business_cost_formula() -> None:
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    metrics = compute_group_metrics(y_true, y_pred, fn_cost=10.0, fp_cost=1.0)

    assert metrics["business_cost"] == business_cost(y_true, y_pred, fn_cost=10.0, fp_cost=1.0)
    assert metrics["n"] == 5
    assert metrics["n_positive_target"] == 3
    assert metrics["selection_rate"] == 0.6
    assert metrics["recall"] == 2 / 3
    assert metrics["fpr"] == 0.5


def test_compute_fairness_table_excludes_nan_group_without_imputing() -> None:
    frame = pd.DataFrame(
        {
            "CODE_GENDER": ["M", "F", "M", "F", None],
            "TARGET": [0, 1, 0, 1, 1],
            "prediction": [0, 1, 0, 0, 1],
        }
    )

    table = compute_fairness_table(frame, group_column="CODE_GENDER")

    assert table["group"].tolist() == ["F", "M"]
    assert table.loc[table["group"] == "F", "n"].iloc[0] == 2


def test_disparate_impact_ratio_flags_known_unfair_case() -> None:
    # Groupe A : 80% selectionnes ; groupe B : 20% -> ratio 0.25, tres < 0.8
    fairness_table = pd.DataFrame(
        {
            "group": ["A", "B"],
            "selection_rate": [0.8, 0.2],
            "recall": [0.7, 0.5],
        }
    )

    summary = compute_disparity_summary(fairness_table)

    assert summary["disparate_impact_ratio"] == 0.25
    assert summary["flag_disparate_impact"] is True
    assert summary["equal_opportunity_difference"] == pytest.approx(0.2)


def test_disparate_impact_ratio_does_not_flag_balanced_case() -> None:
    fairness_table = pd.DataFrame(
        {
            "group": ["A", "B"],
            "selection_rate": [0.5, 0.48],
            "recall": [0.6, 0.61],
        }
    )

    summary = compute_disparity_summary(fairness_table)

    assert summary["flag_disparate_impact"] is False

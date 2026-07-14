import numpy as np
import pytest

from home_credit_mlops.modeling.metrics import (
    build_threshold_sweep,
    business_cost,
    find_best_threshold,
)


def test_business_cost_penalizes_false_negatives_more_heavily() -> None:
    y_true = np.array([1, 1, 0, 0])
    false_negative_predictions = np.array([0, 1, 0, 0])
    false_positive_predictions = np.array([1, 1, 1, 0])

    fn_cost = business_cost(y_true, false_negative_predictions, fn_cost=10.0, fp_cost=1.0)
    fp_cost = business_cost(y_true, false_positive_predictions, fn_cost=10.0, fp_cost=1.0)

    assert fn_cost > fp_cost


def test_find_best_threshold_returns_valid_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.65, 0.95])

    result = find_best_threshold(y_true, y_score, fn_cost=10.0, fp_cost=1.0, grid_size=11)

    assert 0.0 <= result.threshold <= 1.0
    assert result.business_cost >= 0.0


def test_build_threshold_sweep_includes_selected_threshold_from_extra_values() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.65, 0.95])
    best_result = find_best_threshold(y_true, y_score, fn_cost=10.0, fp_cost=1.0, grid_size=11)

    sweep = build_threshold_sweep(
        y_true,
        y_score,
        fn_cost=10.0,
        fp_cost=1.0,
        grid_size=11,
        extra_thresholds=[best_result.threshold],
    )

    selected_rows = sweep.loc[np.isclose(sweep['threshold'], best_result.threshold)]
    assert len(selected_rows) == 1
    assert selected_rows.iloc[0]['business_cost'] == pytest.approx(best_result.business_cost)
    assert sweep['threshold'].is_monotonic_increasing

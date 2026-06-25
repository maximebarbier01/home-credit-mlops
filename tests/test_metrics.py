import numpy as np

from home_credit_mlops.modeling.metrics import business_cost, find_best_threshold


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

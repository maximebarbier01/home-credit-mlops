from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    business_cost: float
    business_score: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    average_precision: float
    brier_score: float
    ks_statistic: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    return true_negative, false_positive, false_negative, true_positive


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    except ValueError:
        return float("nan")
    return float(np.max(true_positive_rate - false_positive_rate))


def business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    normalize: bool = True,
) -> float:
    _, false_positive, false_negative, _ = _confusion_counts(y_true, y_pred)
    cost = false_negative * fn_cost + false_positive * fp_cost
    if normalize:
        return float(cost / len(y_true))
    return float(cost)


def evaluate_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> ThresholdResult:
    y_pred = (y_score >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = _confusion_counts(y_true, y_pred)
    cost = business_cost(y_true, y_pred, fn_cost=fn_cost, fp_cost=fp_cost, normalize=True)
    return ThresholdResult(
        threshold=float(threshold),
        business_cost=cost,
        business_score=-cost,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        roc_auc=_safe_roc_auc(y_true, y_score),
        average_precision=_safe_average_precision(y_true, y_score),
        brier_score=float(brier_score_loss(y_true, y_score)),
        ks_statistic=_ks_statistic(y_true, y_score),
        true_negatives=true_negative,
        false_positives=false_positive,
        false_negatives=false_negative,
        true_positives=true_positive,
    )


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    grid_size: int = 401,
) -> ThresholdResult:
    candidate_thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, grid_size), y_score]))

    best_result: ThresholdResult | None = None
    for threshold in candidate_thresholds:
        result = evaluate_threshold(
            y_true,
            y_score,
            threshold=float(threshold),
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        if best_result is None:
            best_result = result
            continue

        if result.business_cost < best_result.business_cost:
            best_result = result
            continue

        if result.business_cost == best_result.business_cost and result.recall > best_result.recall:
            best_result = result

    if best_result is None:
        raise ValueError("Unable to compute an optimal threshold.")
    return best_result


def business_scorer(
    estimator,
    features,
    target,
    *,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    grid_size: int = 401,
) -> float:
    probabilities = estimator.predict_proba(features)[:, 1]
    threshold_result = find_best_threshold(
        np.asarray(target),
        np.asarray(probabilities),
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        grid_size=grid_size,
    )
    return threshold_result.business_score

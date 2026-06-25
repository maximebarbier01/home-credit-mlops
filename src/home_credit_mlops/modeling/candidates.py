from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


EstimatorFactory = Callable[[], Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_factory: EstimatorFactory
    param_grid: list[dict[str, list[Any]]]


def get_model_specs() -> dict[str, ModelSpec]:
    return {
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            estimator_factory=lambda: LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
            ),
            param_grid=[
                {
                    "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                }
            ],
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            estimator_factory=lambda: RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=1,
            ),
            param_grid=[
                {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 12, 20],
                    "model__min_samples_leaf": [1, 5],
                }
            ],
        ),
        "extra_trees": ModelSpec(
            name="extra_trees",
            estimator_factory=lambda: ExtraTreesClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            param_grid=[
                {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 12, 20],
                    "model__min_samples_leaf": [1, 5],
                }
            ],
        ),
    }

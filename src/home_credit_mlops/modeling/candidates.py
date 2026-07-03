from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from lightgbm import LGBMClassifier
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
                max_iter=4000,
                random_state=42,
                solver="saga",
            ),
            param_grid=[
                {
                    "model__C": [0.1, 0.5, 1.0, 2.0],
                    "model__penalty": ["l1", "l2"],
                }
            ],
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            estimator_factory=lambda: RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=42,
            ),
            param_grid=[
                {
                    "model__n_estimators": [300],
                    "model__max_depth": [None, 16],
                    "model__min_samples_leaf": [1, 5],
                }
            ],
        ),
        "extra_trees": ModelSpec(
            name="extra_trees",
            estimator_factory=lambda: ExtraTreesClassifier(
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            ),
            param_grid=[
                {
                    "model__n_estimators": [300],
                    "model__max_depth": [None, 16],
                    "model__min_samples_leaf": [1, 5],
                }
            ],
        ),
        "lightgbm": ModelSpec(
            name="lightgbm",
            estimator_factory=lambda: LGBMClassifier(
                class_weight="balanced",
                colsample_bytree=0.8,
                learning_rate=0.05,
                n_estimators=400,
                n_jobs=1,
                objective="binary",
                random_state=42,
                subsample=0.8,
                verbosity=-1,
            ),
            param_grid=[
                {
                    "model__n_estimators": [300, 500],
                    "model__learning_rate": [0.03, 0.05],
                    "model__num_leaves": [31, 63],
                }
            ],
        ),
    }

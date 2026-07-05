from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


EstimatorFactory = Callable[[], Any]
VALID_SAMPLING_STRATEGIES = (
    "baseline",
    "smote",
    "borderline_smote",
    "adasyn",
    "smote_under",
)
DEFAULT_SAMPLING_STRATEGIES = ("baseline",)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_factory: EstimatorFactory
    param_grid: list[dict[str, list[Any]]]
    base_model_name: str
    sampling_strategy: str = "baseline"


def get_model_specs() -> dict[str, ModelSpec]:
    return {
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            base_model_name="logistic_regression",
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
            base_model_name="random_forest",
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
            base_model_name="extra_trees",
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
            base_model_name="lightgbm",
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


def build_candidate_model_specs(
    model_names: Sequence[str] | None = None,
    sampling_strategies: Sequence[str] | None = None,
) -> dict[str, ModelSpec]:
    base_specs = get_model_specs()
    selected_model_names = list(model_names) if model_names else list(base_specs.keys())
    selected_sampling_strategies = list(dict.fromkeys(sampling_strategies or DEFAULT_SAMPLING_STRATEGIES))

    unknown_models = [name for name in selected_model_names if name not in base_specs]
    if unknown_models:
        raise ValueError(f"Unknown model names: {unknown_models}")

    unknown_sampling = [
        sampling for sampling in selected_sampling_strategies if sampling not in VALID_SAMPLING_STRATEGIES
    ]
    if unknown_sampling:
        raise ValueError(f"Unknown sampling strategies: {unknown_sampling}")

    candidates: dict[str, ModelSpec] = {}
    for base_model_name in selected_model_names:
        base_spec = base_specs[base_model_name]
        for sampling_strategy in selected_sampling_strategies:
            candidate_name = (
                base_model_name
                if sampling_strategy == "baseline"
                else f"{base_model_name}__{sampling_strategy}"
            )
            candidates[candidate_name] = ModelSpec(
                name=candidate_name,
                base_model_name=base_model_name,
                estimator_factory=base_spec.estimator_factory,
                param_grid=base_spec.param_grid,
                sampling_strategy=sampling_strategy,
            )
    return candidates

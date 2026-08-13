from __future__ import annotations

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from home_credit_mlops.modeling.benchmark import _build_pipeline
from home_credit_mlops.modeling.candidates import (
    VALID_SAMPLING_STRATEGIES,
    build_candidate_model_specs,
    get_model_specs,
)
from home_credit_mlops.settings import load_settings


EXPECTED_SAMPLING_STRATEGIES = {
    "baseline",
    "smote",
    "borderline_smote",
    "adasyn",
    "smote_under",
}


def test_valid_sampling_strategies_cover_supported_variants() -> None:
    assert set(VALID_SAMPLING_STRATEGIES) == EXPECTED_SAMPLING_STRATEGIES


def test_build_candidate_model_specs_expands_sampling_variants() -> None:
    specs = build_candidate_model_specs(
        ["lightgbm"],
        ["baseline", "smote", "borderline_smote", "adasyn", "smote_under"],
    )

    assert list(specs) == [
        "lightgbm",
        "lightgbm__smote",
        "lightgbm__borderline_smote",
        "lightgbm__adasyn",
        "lightgbm__smote_under",
    ]
    assert specs["lightgbm"].sampling_strategy == "baseline"
    assert specs["lightgbm__borderline_smote"].base_model_name == "lightgbm"
    assert specs["lightgbm__smote_under"].sampling_strategy == "smote_under"


def test_mlp_is_available_and_requires_scaling() -> None:
    spec = get_model_specs()["mlp"]
    estimator = spec.estimator_factory()

    assert estimator.__class__.__name__ == "MLPClassifier"
    assert spec.requires_scaling is True
    assert spec.param_grid == [
        {
            "model__hidden_layer_sizes": [(64,), (64, 32)],
            "model__alpha": [1e-4, 1e-3],
        }
    ]


def test_tree_based_models_do_not_require_scaling() -> None:
    specs = get_model_specs()
    for model_name in ["logistic_regression", "random_forest", "extra_trees", "lightgbm", "xgboost"]:
        assert specs[model_name].requires_scaling is False


def test_build_pipeline_inserts_scaler_only_for_models_that_require_it() -> None:
    features = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "b", "a", "b", "a", "c"],
        }
    )
    settings = load_settings()
    specs = build_candidate_model_specs(["lightgbm", "mlp"], ["baseline", "smote"])

    lightgbm_pipeline = _build_pipeline(specs["lightgbm"], features, settings)
    mlp_pipeline = _build_pipeline(specs["mlp"], features, settings)
    mlp_smote_pipeline = _build_pipeline(specs["mlp__smote"], features, settings)

    assert "scaler" not in lightgbm_pipeline.named_steps
    assert "scaler" in mlp_pipeline.named_steps
    assert isinstance(mlp_pipeline, Pipeline)
    assert isinstance(mlp_smote_pipeline, ImbPipeline)
    assert "scaler" in mlp_smote_pipeline.named_steps
    # Le scaler doit precede le sampler (SMOTE est sensible a l'echelle),
    # lui-meme avant le modele.
    assert list(mlp_smote_pipeline.named_steps.keys()) == [
        "preprocessor",
        "scaler",
        "sampler",
        "model",
    ]


def test_xgboost_is_available_with_safe_parallelism() -> None:
    spec = get_model_specs()["xgboost"]
    estimator = spec.estimator_factory()

    assert estimator.__class__.__name__ == "XGBClassifier"
    assert estimator.n_jobs == 1
    assert estimator.tree_method == "hist"
    assert spec.param_grid == [
        {
            "model__n_estimators": [300, 500],
            "model__learning_rate": [0.03, 0.05],
            "model__max_depth": [4, 6],
        }
    ]


def test_build_pipeline_uses_expected_sampling_steps() -> None:
    features = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "b", "a", "b", "a", "c"],
        }
    )
    settings = load_settings()
    specs = build_candidate_model_specs(
        ["lightgbm"],
        ["baseline", "smote", "borderline_smote", "adasyn", "smote_under"],
    )

    baseline_pipeline = _build_pipeline(specs["lightgbm"], features, settings)
    smote_pipeline = _build_pipeline(specs["lightgbm__smote"], features, settings)
    borderline_pipeline = _build_pipeline(specs["lightgbm__borderline_smote"], features, settings)
    adasyn_pipeline = _build_pipeline(specs["lightgbm__adasyn"], features, settings)
    smote_under_pipeline = _build_pipeline(specs["lightgbm__smote_under"], features, settings)

    assert isinstance(baseline_pipeline, Pipeline)
    assert isinstance(smote_pipeline, ImbPipeline)
    assert isinstance(borderline_pipeline, ImbPipeline)
    assert isinstance(adasyn_pipeline, ImbPipeline)
    assert isinstance(smote_under_pipeline, ImbPipeline)
    assert smote_pipeline.named_steps["sampler"].__class__.__name__ == "SMOTE"
    assert borderline_pipeline.named_steps["sampler"].__class__.__name__ == "BorderlineSMOTE"
    assert adasyn_pipeline.named_steps["sampler"].__class__.__name__ == "ADASYN"
    assert smote_under_pipeline.named_steps["over"].__class__.__name__ == "SMOTE"
    assert smote_under_pipeline.named_steps["under"].__class__.__name__ == "RandomUnderSampler"

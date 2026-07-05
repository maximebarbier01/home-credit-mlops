from __future__ import annotations

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from home_credit_mlops.modeling.benchmark import _build_pipeline
from home_credit_mlops.modeling.candidates import (
    VALID_SAMPLING_STRATEGIES,
    build_candidate_model_specs,
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

from __future__ import annotations

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from home_credit_mlops.modeling.benchmark import _build_pipeline
from home_credit_mlops.modeling.candidates import build_candidate_model_specs
from home_credit_mlops.settings import load_settings


def test_build_candidate_model_specs_expands_sampling_variants() -> None:
    specs = build_candidate_model_specs(["lightgbm"], ["baseline", "smote"])

    assert list(specs) == ["lightgbm", "lightgbm__smote"]
    assert specs["lightgbm"].sampling_strategy == "baseline"
    assert specs["lightgbm__smote"].base_model_name == "lightgbm"


def test_build_pipeline_uses_smote_variant_when_requested() -> None:
    features = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "b", "a", "b", "a", "c"],
        }
    )
    settings = load_settings()
    specs = build_candidate_model_specs(["lightgbm"], ["baseline", "smote"])

    baseline_pipeline = _build_pipeline(specs["lightgbm"], features, settings)
    smote_pipeline = _build_pipeline(specs["lightgbm__smote"], features, settings)

    assert isinstance(baseline_pipeline, Pipeline)
    assert isinstance(smote_pipeline, ImbPipeline)
    assert "sampler" in smote_pipeline.named_steps

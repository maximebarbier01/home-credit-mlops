from __future__ import annotations

import pandas as pd

from home_credit_mlops.modeling.benchmark import (
    _build_default_campaign_name,
    _build_mlflow_runs_summary,
    _slugify_campaign_name,
)


def test_slugify_campaign_name_normalizes_text() -> None:
    assert _slugify_campaign_name("Final Benchmark LightGBM CV5") == "final_benchmark_lightgbm_cv5"


def test_build_default_campaign_name_reflects_scope() -> None:
    name = _build_default_campaign_name(["lightgbm", "extra_trees"], 10000, 3)
    assert name == "benchmark_2_models_10000_rows_cv3"


def test_build_mlflow_runs_summary_contains_campaign_and_models() -> None:
    results = pd.DataFrame([
        {"model_name": "lightgbm", "run_id": "run-1", "threshold": 0.12, "holdout_business_cost": 0.45, "holdout_roc_auc": 0.78},
        {"model_name": "extra_trees", "run_id": "run-2", "threshold": 0.19, "holdout_business_cost": 0.51, "holdout_roc_auc": 0.74},
    ])

    summary = _build_mlflow_runs_summary(
        results,
        campaign_name="benchmark_2_models_10000_rows_cv3",
        best_model_name="lightgbm",
        root_run_id="root-1",
    )

    assert summary.iloc[0]["scope"] == "campaign"
    assert summary.iloc[0]["run_id"] == "root-1"
    assert set(summary["scope"]) == {"campaign", "model"}
    assert summary.loc[summary["model"] == "lightgbm", "selected_as_best"].item() is True

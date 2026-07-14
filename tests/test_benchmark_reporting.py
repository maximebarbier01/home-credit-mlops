from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from home_credit_mlops.modeling.benchmark import (
    _build_default_campaign_name,
    _build_mlflow_runs_summary,
    _export_threshold_optimization_artifacts,
    _slugify_campaign_name,
)
from home_credit_mlops.settings import (
    BusinessConfig,
    DatasetConfig,
    MlflowConfig,
    PathsConfig,
    Settings,
    TrainingConfig,
)


def _build_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        paths=PathsConfig(
            raw_dir=tmp_path / 'raw',
            interim_dir=tmp_path / 'interim',
            processed_dir=tmp_path / 'processed',
            reports_dir=tmp_path / 'reports',
            artifacts_dir=tmp_path / 'artifacts',
        ),
        dataset=DatasetConfig(
            default_train_path=tmp_path / 'processed' / 'train.parquet',
            target_column='TARGET',
            id_column='SK_ID_CURR',
            test_size=0.2,
            random_state=42,
        ),
        business=BusinessConfig(
            fn_cost=10.0,
            fp_cost=1.0,
            threshold_grid_size=11,
        ),
        training=TrainingConfig(
            cv_folds=3,
            n_jobs=1,
        ),
        mlflow=MlflowConfig(
            experiment_name='test-experiment',
            backend_store_path=tmp_path / 'mlflow.db',
            artifact_root=tmp_path / 'mlartifacts',
        ),
    )


def test_slugify_campaign_name_normalizes_text() -> None:
    assert _slugify_campaign_name('Final Benchmark LightGBM CV5') == 'final_benchmark_lightgbm_cv5'


def test_build_default_campaign_name_reflects_scope() -> None:
    name = _build_default_campaign_name(['lightgbm', 'extra_trees'], ['baseline', 'smote'], 10000, 3)
    assert name == 'benchmark_2_models_2_sampling_modes_10000_rows_cv3'


def test_build_mlflow_runs_summary_contains_campaign_models_and_sampling() -> None:
    results = pd.DataFrame([
        {
            'model_name': 'lightgbm',
            'base_model_name': 'lightgbm',
            'sampling_strategy': 'baseline',
            'run_id': 'run-1',
            'threshold': 0.12,
            'holdout_business_cost': 0.45,
            'holdout_roc_auc': 0.78,
        },
        {
            'model_name': 'extra_trees__smote',
            'base_model_name': 'extra_trees',
            'sampling_strategy': 'smote',
            'run_id': 'run-2',
            'threshold': 0.19,
            'holdout_business_cost': 0.51,
            'holdout_roc_auc': 0.74,
        },
    ])

    summary = _build_mlflow_runs_summary(
        results,
        campaign_name='benchmark_2_models_2_sampling_modes_10000_rows_cv3',
        best_model_name='lightgbm',
        root_run_id='root-1',
    )

    assert summary.iloc[0]['scope'] == 'campaign'
    assert summary.iloc[0]['run_id'] == 'root-1'
    assert set(summary['scope']) == {'campaign', 'model'}
    assert summary.loc[summary['model'] == 'lightgbm', 'selected_as_best'].item() is True
    assert summary.loc[summary['model'] == 'extra_trees__smote', 'sampling'].item() == 'smote'


def test_export_threshold_optimization_artifacts_writes_expected_files(tmp_path: Path) -> None:
    settings = _build_test_settings(tmp_path)
    output_dir = tmp_path / 'run'
    selected_threshold = 0.42
    oof_predictions = pd.DataFrame(
        {
            'TARGET': [0, 0, 1, 1],
            'probability': [0.05, 0.35, 0.62, 0.91],
        }
    )
    holdout_predictions = pd.DataFrame(
        {
            'TARGET': [0, 0, 1, 1],
            'probability': [0.08, 0.45, 0.58, 0.88],
        }
    )

    summary = _export_threshold_optimization_artifacts(
        output_dir,
        model_name='lightgbm__smote',
        oof_predictions=oof_predictions,
        holdout_predictions=holdout_predictions,
        selected_threshold=selected_threshold,
        settings=settings,
    )

    threshold_dir = output_dir / 'threshold_optimization'
    assert (threshold_dir / 'lightgbm__smote_oof_threshold_metrics.csv').exists()
    assert (threshold_dir / 'lightgbm__smote_holdout_threshold_metrics.csv').exists()
    assert (threshold_dir / 'lightgbm__smote_business_cost_vs_threshold.png').exists()
    assert (threshold_dir / 'lightgbm__smote_classification_metrics_vs_threshold.png').exists()
    assert (threshold_dir / 'lightgbm__smote_threshold_selection_summary.json').exists()

    oof_metrics = pd.read_csv(threshold_dir / 'lightgbm__smote_oof_threshold_metrics.csv')
    holdout_metrics = pd.read_csv(threshold_dir / 'lightgbm__smote_holdout_threshold_metrics.csv')
    assert np.isclose(oof_metrics['threshold'], selected_threshold).any()
    assert np.isclose(holdout_metrics['threshold'], selected_threshold).any()
    assert oof_metrics['selected_threshold'].sum() == 1
    assert holdout_metrics['selected_threshold'].sum() == 1
    assert summary['selected_threshold'] == selected_threshold

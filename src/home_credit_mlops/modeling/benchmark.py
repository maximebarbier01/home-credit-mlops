from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import json
import logging
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from home_credit_mlops.data.io import read_table
from home_credit_mlops.features.preprocessing import build_preprocessor, split_features_target
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.mlflow_utils import configure_mlflow, register_logged_model
from home_credit_mlops.modeling.candidates import (
    DEFAULT_SAMPLING_STRATEGIES,
    VALID_SAMPLING_STRATEGIES,
    ModelSpec,
    build_candidate_model_specs,
)
from home_credit_mlops.modeling.interpretability import (
    export_feature_importance,
    export_shap_analysis,
)
from home_credit_mlops.modeling.metrics import (
    business_scorer,
    evaluate_threshold,
    find_best_threshold,
)
from home_credit_mlops.reporting.excel import build_experiment_workbooks, remove_files_by_suffix
from home_credit_mlops.settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)


PIPELINE_STEPS = [
    "model_preprocessing",
    "cross_validated_training",
    "performance_evaluation",
    "decision_threshold_optimization",
    "final_model_refit",
    "interpretability_export",
    "report_packaging",
]
SMOTE_SAMPLING_STRATEGY = 0.3
SMOTE_K_NEIGHBORS = 5


@dataclass(frozen=True)
class BenchmarkRunResult:
    model_name: str
    base_model_name: str
    sampling_strategy: str
    run_id: str | None
    best_params: dict[str, Any]
    threshold: float
    cv_business_cost: float
    cv_roc_auc: float
    cv_average_precision: float
    cv_accuracy: float
    cv_balanced_accuracy: float
    oof_roc_auc: float
    oof_average_precision: float
    oof_precision: float
    oof_recall: float
    oof_f1: float
    oof_accuracy: float
    oof_balanced_accuracy: float
    holdout_business_cost: float
    holdout_business_score: float
    holdout_roc_auc: float
    holdout_average_precision: float
    holdout_accuracy: float
    holdout_balanced_accuracy: float
    holdout_precision: float
    holdout_recall: float
    holdout_f1: float
    holdout_brier_score: float
    holdout_ks_statistic: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


@dataclass
class ModelBenchmarkArtifacts:
    result: BenchmarkRunResult
    search: GridSearchCV
    best_estimator: Pipeline
    oof_predictions: pd.DataFrame
    holdout_predictions: pd.DataFrame


def _jsonable(mapping: dict[str, Any]) -> dict[str, Any]:
    jsonable: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            jsonable[key] = {inner_key: inner_value for inner_key, inner_value in value.items()}
        elif hasattr(value, "item"):
            jsonable[key] = value.item()
        else:
            jsonable[key] = value
    return jsonable


def _build_smote_sampler(settings: Settings) -> SMOTE:
    return SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=settings.dataset.random_state,
    )


def _build_pipeline(
    model_spec: ModelSpec,
    features: pd.DataFrame,
    settings: Settings,
) -> Pipeline | ImbPipeline:
    preprocessor, _, _ = build_preprocessor(features)
    model = model_spec.estimator_factory()

    if model_spec.sampling_strategy == "smote":
        return ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("sampler", _build_smote_sampler(settings)),
                ("model", model),
            ]
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _sample_training_frame(
    dataframe: pd.DataFrame,
    *,
    sample_size: int | None,
    target_column: str,
    random_state: int,
) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(dataframe):
        return dataframe.copy()

    sampled, _ = train_test_split(
        dataframe,
        train_size=sample_size,
        stratify=dataframe[target_column],
        random_state=random_state,
    )
    return sampled.copy().sort_values("SK_ID_CURR").reset_index(drop=True)


def _save_cv_results(search: GridSearchCV, output_dir: Path, model_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_cv_results.csv"
    pd.DataFrame(search.cv_results_).to_csv(output_path, index=False)
    return output_path


def _save_prediction_tables(
    output_dir: Path,
    model_name: str,
    oof_predictions: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_predictions.to_parquet(output_dir / f"{model_name}_oof_predictions.parquet", index=False)
    holdout_predictions.to_parquet(
        output_dir / f"{model_name}_holdout_predictions.parquet",
        index=False,
    )


def _resolve_pre_dispatch(n_jobs: int) -> int | str:
    if n_jobs == -1:
        return "2*n_jobs"
    return max(1, n_jobs)


def _plot_holdout_diagnostics(
    output_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_score >= threshold).astype(int)

    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_roc_curve.png", dpi=150)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_precision_recall_curve.png", dpi=150)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close()


def _export_all_model_diagnostics(
    output_dir: Path,
    *,
    artifacts_by_model: dict[str, ModelBenchmarkArtifacts],
    y_holdout: pd.Series,
) -> None:
    diagnostics_root = output_dir / "diagnostics"
    for model_name, artifacts in artifacts_by_model.items():
        _plot_holdout_diagnostics(
            diagnostics_root / model_name,
            model_name,
            y_holdout.to_numpy(),
            artifacts.holdout_predictions["probability"].to_numpy(),
            artifacts.result.threshold,
        )


def _build_scoring(settings: Settings) -> dict[str, Any]:
    return {
        "business_score": partial(
            business_scorer,
            fn_cost=settings.business.fn_cost,
            fp_cost=settings.business.fp_cost,
            grid_size=settings.business.threshold_grid_size,
        ),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }


def _metrics_from_result(result: BenchmarkRunResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in asdict(result).items():
        if key in {"model_name", "run_id", "best_params"} or value is None:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            metrics[key] = float(value)
    return metrics


def _f_beta_from_precision_recall(
    precision: float,
    recall: float,
    *,
    beta: float = 2.0,
) -> float:
    beta_sq = beta**2
    denominator = beta_sq * precision + recall
    if denominator == 0:
        return 0.0
    return float((1 + beta_sq) * precision * recall / denominator)


def _model_family_from_estimator_class(estimator_class: str) -> str:
    if "LogisticRegression" in estimator_class:
        return "linear"
    if "Forest" in estimator_class or "Trees" in estimator_class:
        return "bagging_tree"
    if "LGBM" in estimator_class or "Boost" in estimator_class:
        return "boosting_tree"
    return "other"


def _slugify_campaign_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    slug = normalized.strip("_")
    return slug or "benchmark"


def _build_default_campaign_name(
    model_names: list[str] | None,
    sampling_strategies: list[str] | None,
    sample_size: int | None,
    cv_folds: int,
) -> str:
    if not model_names:
        model_token = "all_models"
    elif len(model_names) == 1:
        model_token = model_names[0]
    else:
        model_token = f"{len(model_names)}_models"

    if not sampling_strategies or sampling_strategies == ["baseline"]:
        sampling_token = "baseline"
    elif len(sampling_strategies) == 1:
        sampling_token = sampling_strategies[0]
    else:
        sampling_token = f"{len(sampling_strategies)}_sampling_modes"

    sample_token = "full_dataset" if sample_size is None else f"{sample_size}_rows"
    return _slugify_campaign_name(
        f"benchmark_{model_token}_{sampling_token}_{sample_token}_cv{cv_folds}"
    )


def _build_campaign_overview(
    *,
    campaign_name: str,
    created_at: str,
    dataset_label: str,
    dataset_path: str,
    target_column: str,
    id_column: str,
    drop_columns: list[str],
    selected_model_names: list[str],
    sampling_strategies: list[str],
    sample_size: int | None,
    experiment_frame: pd.DataFrame,
    train_rows: int,
    holdout_rows: int,
    cv_folds: int,
    n_jobs: int,
    fn_cost: float,
    fp_cost: float,
    enable_mlflow: bool,
    root_run_id: str | None,
    registered_model_name: str | None,
    registered_model_version: str | None,
    best_model_name: str,
    output_dir: Path,
    test_dataset_available: bool,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "campaign_name": campaign_name,
        "campaign_slug": _slugify_campaign_name(campaign_name),
        "created_at": created_at,
        "dataset_label": dataset_label,
        "dataset_path": dataset_path,
        "output_dir": output_dir.as_posix(),
        "target_column": target_column,
        "id_column": id_column,
        "drop_columns": json.dumps(drop_columns),
        "candidate_models": ",".join(selected_model_names),
        "candidate_model_count": len(selected_model_names),
        "sampling_strategies": ",".join(sampling_strategies),
        "sample_size_requested": sample_size if sample_size is not None else "full_dataset",
        "sampled_rows": int(len(experiment_frame)),
        "sampled_columns": int(experiment_frame.shape[1]),
        "train_rows": int(train_rows),
        "holdout_rows": int(holdout_rows),
        "target_rate": float(experiment_frame[target_column].mean()),
        "cv_folds": int(cv_folds),
        "n_jobs": int(n_jobs),
        "fn_cost": float(fn_cost),
        "fp_cost": float(fp_cost),
        "threshold_policy": "oof_business_cost_minimization",
        "mlflow_enabled": bool(enable_mlflow),
        "mlflow_root_run_id": root_run_id or "",
        "registered_model_name": registered_model_name or "",
        "registered_model_version": registered_model_version or "",
        "best_model": best_model_name,
        "test_dataset_available": bool(test_dataset_available),
    }])


def _build_cv_summary(results_frame: pd.DataFrame, *, best_model_name: str) -> pd.DataFrame:
    summary = results_frame.copy()
    summary.insert(0, "selected_as_best", summary["model_name"] == best_model_name)
    summary["model"] = summary["model_name"]
    summary["base_model"] = summary["base_model_name"]
    summary["sampling"] = summary["sampling_strategy"]
    ordered_cols = [
        "selected_as_best", "model", "base_model", "sampling", "threshold", "cv_business_cost", "cv_roc_auc",
        "cv_average_precision", "cv_accuracy", "cv_balanced_accuracy", "oof_roc_auc",
        "oof_average_precision", "oof_precision", "oof_recall", "oof_f1",
        "oof_accuracy", "oof_balanced_accuracy", "best_params", "run_id",
    ]
    return summary[ordered_cols].copy()


def _build_holdout_summary(results_frame: pd.DataFrame, *, best_model_name: str) -> pd.DataFrame:
    summary = results_frame.copy()
    summary.insert(0, "selected_as_best", summary["model_name"] == best_model_name)
    summary["model"] = summary["model_name"]
    summary["base_model"] = summary["base_model_name"]
    summary["sampling"] = summary["sampling_strategy"]
    ordered_cols = [
        "selected_as_best", "model", "base_model", "sampling", "threshold", "holdout_business_cost",
        "holdout_business_score", "holdout_roc_auc", "holdout_average_precision",
        "holdout_accuracy", "holdout_balanced_accuracy", "holdout_precision",
        "holdout_recall", "holdout_f1", "holdout_brier_score", "holdout_ks_statistic",
        "true_negatives", "false_positives", "false_negatives", "true_positives",
        "best_params", "run_id",
    ]
    return summary[ordered_cols].copy()


def _build_decision_threshold_summary(results_frame: pd.DataFrame, *, best_model_name: str) -> pd.DataFrame:
    summary = results_frame.copy()
    summary.insert(0, "selected_as_best", summary["model_name"] == best_model_name)
    summary["model"] = summary["model_name"]
    summary["base_model"] = summary["base_model_name"]
    summary["sampling"] = summary["sampling_strategy"]
    summary["selection_basis"] = "out_of_fold_business_cost_minimization"
    ordered_cols = [
        "selected_as_best", "model", "base_model", "sampling", "selection_basis", "threshold",
        "holdout_business_cost", "holdout_business_score", "holdout_precision",
        "holdout_recall", "holdout_f1", "true_negatives", "false_positives",
        "false_negatives", "true_positives", "run_id",
    ]
    return summary[ordered_cols].copy()


def _build_mlflow_runs_summary(results_frame: pd.DataFrame, *, campaign_name: str, best_model_name: str, root_run_id: str | None) -> pd.DataFrame:
    rows = [{
        "scope": "campaign",
        "campaign_name": campaign_name,
        "model": "",
        "base_model": "",
        "sampling": "",
        "run_id": root_run_id or "",
        "selected_as_best": False,
        "threshold": np.nan,
        "holdout_business_cost": np.nan,
        "holdout_roc_auc": np.nan,
    }]
    for row in results_frame.itertuples(index=False):
        rows.append({
            "scope": "model",
            "campaign_name": campaign_name,
            "model": row.model_name,
            "base_model": row.base_model_name,
            "sampling": row.sampling_strategy,
            "run_id": row.run_id or "",
            "selected_as_best": row.model_name == best_model_name,
            "threshold": float(row.threshold),
            "holdout_business_cost": float(row.holdout_business_cost),
            "holdout_roc_auc": float(row.holdout_roc_auc),
        })
    return pd.DataFrame(rows)


def _build_best_model_summary(performance_summary: pd.DataFrame, *, best_model_name: str) -> pd.DataFrame:
    return performance_summary.loc[performance_summary["model"] == best_model_name].copy()


def _build_model_performance_summary(
    results_frame: pd.DataFrame,
    available_models: dict[str, ModelSpec],
    *,
    best_model_name: str,
) -> pd.DataFrame:
    summary = results_frame.copy()
    estimator_class_lookup = {
        model_name: available_models[model_name].estimator_factory().__class__.__name__
        for model_name in summary["model_name"].tolist()
        if model_name in available_models
    }
    summary["model"] = summary["model_name"]
    summary["base_model"] = summary["base_model_name"]
    summary["sampling"] = summary["sampling_strategy"]
    summary["estimator_class"] = summary["model"].map(estimator_class_lookup)
    summary["family"] = summary["estimator_class"].map(_model_family_from_estimator_class)
    summary["strategie_seuil"] = "cv_business_cost_optimized"
    summary["selected_as_best"] = summary["model"] == best_model_name
    summary["precision_1"] = summary["holdout_precision"]
    summary["recall_1"] = summary["holdout_recall"]
    summary["f1_1"] = summary["holdout_f1"]
    summary["f2_1"] = summary.apply(
        lambda row: _f_beta_from_precision_recall(
            float(row["holdout_precision"]),
            float(row["holdout_recall"]),
            beta=2.0,
        ),
        axis=1,
    )
    summary["prc_auc"] = summary["holdout_average_precision"]
    summary["train_precision_1"] = summary["oof_precision"]
    summary["train_recall_1"] = summary["oof_recall"]
    summary["train_f1_1"] = summary["oof_f1"]
    summary["train_f2_1"] = summary.apply(
        lambda row: _f_beta_from_precision_recall(
            float(row["oof_precision"]),
            float(row["oof_recall"]),
            beta=2.0,
        ),
        axis=1,
    )
    summary["train_prc_auc"] = summary["oof_average_precision"]
    summary["tn"] = summary["true_negatives"]
    summary["fp"] = summary["false_positives"]
    summary["fn"] = summary["false_negatives"]
    summary["tp"] = summary["true_positives"]
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    ordered_cols = [
        "rank",
        "selected_as_best",
        "model",
        "base_model",
        "sampling",
        "family",
        "estimator_class",
        "strategie_seuil",
        "threshold",
        "precision_1",
        "recall_1",
        "f1_1",
        "f2_1",
        "prc_auc",
        "train_precision_1",
        "train_recall_1",
        "train_f1_1",
        "train_f2_1",
        "train_prc_auc",
        "holdout_business_cost",
        "holdout_business_score",
        "cv_business_cost",
        "cv_roc_auc",
        "cv_average_precision",
        "tn",
        "fp",
        "fn",
        "tp",
        "best_params",
        "run_id",
    ]
    return summary[[column for column in ordered_cols if column in summary.columns]].copy()


def _log_candidate_run(
    artifacts: ModelBenchmarkArtifacts,
    *,
    output_dir: Path,
    x_example: pd.DataFrame,
    campaign_name: str,
) -> None:
    result = artifacts.result
    mlflow.set_tags(
        {
            "stage": "candidate_benchmark",
            "pipeline": "home_credit_build",
            "campaign_name": campaign_name,
            "model_name": result.model_name,
            "base_model_name": result.base_model_name,
            "sampling_strategy": result.sampling_strategy,
        }
    )
    mlflow.log_param("model_name", result.model_name)
    mlflow.log_param("base_model_name", result.base_model_name)
    mlflow.log_param("sampling_strategy", result.sampling_strategy)
    mlflow.log_params(
        {
            key: (value if isinstance(value, (str, int, float, bool)) else json.dumps(value))
            for key, value in _jsonable(result.best_params).items()
        }
    )
    mlflow.log_metrics(_metrics_from_result(result))
    mlflow.log_dict(_jsonable(asdict(result)), "evaluation_summary.json")

    cv_path = output_dir / "cv_results" / f"{result.model_name}_cv_results.csv"
    if cv_path.exists():
        mlflow.log_artifact(cv_path.as_posix(), artifact_path="cv_results")

    prediction_dir = output_dir / "predictions"
    oof_path = prediction_dir / f"{result.model_name}_oof_predictions.parquet"
    holdout_path = prediction_dir / f"{result.model_name}_holdout_predictions.parquet"
    if oof_path.exists():
        mlflow.log_artifact(oof_path.as_posix(), artifact_path="predictions")
    if holdout_path.exists():
        mlflow.log_artifact(holdout_path.as_posix(), artifact_path="predictions")

    example = x_example.head(min(5, len(x_example))).copy()
    if example.empty:
        return

    signature = infer_signature(example, artifacts.best_estimator.predict_proba(example))
    mlflow.sklearn.log_model(
        sk_model=artifacts.best_estimator,
        artifact_path="candidate_model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature,
        input_example=example,
    )


def _log_experiment_artifacts(output_dir: Path) -> None:
    supported_suffixes = {".csv", ".json", ".xlsx"}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in supported_suffixes:
            continue
        artifact_path = "predictions" if path.name == "best_model_test_predictions.csv" else "experiment"
        mlflow.log_artifact(path.as_posix(), artifact_path=artifact_path)

    for directory_name in ["cv_results", "diagnostics", "interpretability", "predictions"]:
        directory = output_dir / directory_name
        if directory.exists():
            mlflow.log_artifacts(directory.as_posix(), artifact_path=directory_name)


def _cleanup_experiment_csv_files(output_dir: Path) -> None:
    remove_files_by_suffix(output_dir)
    for directory in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        remove_files_by_suffix(directory)


def _log_final_model(
    pipeline: Pipeline | ImbPipeline,
    *,
    features: pd.DataFrame,
    best_result: BenchmarkRunResult,
    register_model_name: str | None,
) -> str | None:
    example = features.head(min(5, len(features))).copy()
    if example.empty:
        return None

    signature = infer_signature(example, pipeline.predict_proba(example))
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="final_model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature,
        input_example=example,
    )
    mlflow.log_dict(
        {
            "model_name": best_result.model_name,
            "threshold": best_result.threshold,
            "best_params": _jsonable(best_result.best_params),
        },
        "best_model_summary.json",
    )

    if not register_model_name:
        return None

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/final_model"
    version = register_logged_model(model_uri, register_model_name)
    mlflow.log_param("registered_model_name", register_model_name)
    mlflow.log_param("registered_model_version", version)
    return version


def _benchmark_single_model(
    model_spec: ModelSpec,
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    id_train: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    id_holdout: pd.Series,
    settings: Settings,
    cv_folds: int,
    output_dir: Path,
) -> ModelBenchmarkArtifacts:
    pipeline = _build_pipeline(model_spec, x_train, settings)
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=settings.dataset.random_state,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_spec.param_grid,
        scoring=_build_scoring(settings),
        refit="business_score",
        cv=cv,
        n_jobs=settings.training.n_jobs,
        pre_dispatch=_resolve_pre_dispatch(settings.training.n_jobs),
        verbose=1,
        error_score="raise",
    )
    search.fit(x_train, y_train)

    oof_probabilities = cross_val_predict(
        search.best_estimator_,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=settings.training.n_jobs,
    )[:, 1]
    threshold_result = find_best_threshold(
        np.asarray(y_train),
        np.asarray(oof_probabilities),
        fn_cost=settings.business.fn_cost,
        fp_cost=settings.business.fp_cost,
        grid_size=settings.business.threshold_grid_size,
    )

    holdout_probabilities = search.best_estimator_.predict_proba(x_holdout)[:, 1]
    holdout_result = evaluate_threshold(
        np.asarray(y_holdout),
        np.asarray(holdout_probabilities),
        threshold=threshold_result.threshold,
        fn_cost=settings.business.fn_cost,
        fp_cost=settings.business.fp_cost,
    )

    result = BenchmarkRunResult(
        model_name=model_spec.name,
        base_model_name=model_spec.base_model_name,
        sampling_strategy=model_spec.sampling_strategy,
        run_id=None,
        best_params=search.best_params_,
        threshold=threshold_result.threshold,
        cv_business_cost=-float(search.best_score_),
        cv_roc_auc=float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
        cv_average_precision=float(
            search.cv_results_["mean_test_average_precision"][search.best_index_]
        ),
        cv_accuracy=float(search.cv_results_["mean_test_accuracy"][search.best_index_]),
        cv_balanced_accuracy=float(
            search.cv_results_["mean_test_balanced_accuracy"][search.best_index_]
        ),
        oof_roc_auc=threshold_result.roc_auc,
        oof_average_precision=threshold_result.average_precision,
        oof_precision=threshold_result.precision,
        oof_recall=threshold_result.recall,
        oof_f1=threshold_result.f1,
        oof_accuracy=threshold_result.accuracy,
        oof_balanced_accuracy=threshold_result.balanced_accuracy,
        holdout_business_cost=holdout_result.business_cost,
        holdout_business_score=holdout_result.business_score,
        holdout_roc_auc=holdout_result.roc_auc,
        holdout_average_precision=holdout_result.average_precision,
        holdout_accuracy=holdout_result.accuracy,
        holdout_balanced_accuracy=holdout_result.balanced_accuracy,
        holdout_precision=holdout_result.precision,
        holdout_recall=holdout_result.recall,
        holdout_f1=holdout_result.f1,
        holdout_brier_score=holdout_result.brier_score,
        holdout_ks_statistic=holdout_result.ks_statistic,
        true_negatives=holdout_result.true_negatives,
        false_positives=holdout_result.false_positives,
        false_negatives=holdout_result.false_negatives,
        true_positives=holdout_result.true_positives,
    )

    oof_predictions = pd.DataFrame(
        {
            "SK_ID_CURR": id_train.to_numpy(),
            "TARGET": y_train.to_numpy(),
            "probability": oof_probabilities,
            "prediction": (oof_probabilities >= threshold_result.threshold).astype(int),
        }
    )
    holdout_predictions = pd.DataFrame(
        {
            "SK_ID_CURR": id_holdout.to_numpy(),
            "TARGET": y_holdout.to_numpy(),
            "probability": holdout_probabilities,
            "prediction": (holdout_probabilities >= threshold_result.threshold).astype(int),
        }
    )

    _save_cv_results(search, output_dir / "cv_results", model_spec.name)
    _save_prediction_tables(
        output_dir / "predictions",
        model_spec.name,
        oof_predictions,
        holdout_predictions,
    )

    return ModelBenchmarkArtifacts(
        result=result,
        search=search,
        best_estimator=search.best_estimator_,
        oof_predictions=oof_predictions,
        holdout_predictions=holdout_predictions,
    )


def _run_benchmark_body(
    dataframe: pd.DataFrame,
    *,
    settings: Settings,
    destination: Path,
    campaign_name: str,
    dataset_label: str,
    dataset_path: str,
    target_column: str,
    id_column: str,
    drop_columns: list[str],
    model_names: list[str] | None,
    sampling_strategies: list[str] | None,
    test_dataframe: pd.DataFrame | None,
    sample_size: int | None,
    cv_folds: int,
    shap_sample_size: int,
    local_explanations: int,
    top_features: int,
    enable_mlflow: bool,
    register_model_name: str | None,
) -> pd.DataFrame:
    experiment_frame = _sample_training_frame(
        dataframe,
        sample_size=sample_size,
        target_column=target_column,
        random_state=settings.dataset.random_state,
    )

    features, target = split_features_target(
        experiment_frame,
        target_column=target_column,
        drop_columns=drop_columns,
    )
    identifiers = experiment_frame[id_column].copy()

    x_train, x_holdout, y_train, y_holdout, id_train, id_holdout = train_test_split(
        features,
        target,
        identifiers,
        test_size=settings.dataset.test_size,
        random_state=settings.dataset.random_state,
        stratify=target,
    )

    sampling_modes = list(dict.fromkeys(sampling_strategies or list(DEFAULT_SAMPLING_STRATEGIES)))
    available_models = build_candidate_model_specs(
        model_names=model_names,
        sampling_strategies=sampling_modes,
    )
    selected_model_names = list(available_models.keys())
    created_at = pd.Timestamp.now().isoformat()
    campaign_slug = _slugify_campaign_name(campaign_name)
    results: list[BenchmarkRunResult] = []
    artifacts_by_model: dict[str, ModelBenchmarkArtifacts] = {}

    if (
        sample_size is None
        and len(selected_model_names) > 1
        and cv_folds >= 5
        and len(experiment_frame) >= 100_000
    ):
        LOGGER.warning(
            "Large benchmark requested: %s rows, %s models, cv=%s, n_jobs=%s. "
            "This can destabilize WSL/VS Code. For development, prefer "
            "--model lightgbm --sample-size 5000 --cv-folds 3 --n-jobs 1.",
            len(experiment_frame),
            len(selected_model_names),
            cv_folds,
            settings.training.n_jobs,
        )

    if enable_mlflow:
        mlflow.set_tags(
            {
                "stage": "benchmark",
                "pipeline": "home_credit_build",
                "campaign_name": campaign_name,
                "campaign_slug": campaign_slug,
                "dataset_label": dataset_label,
                "target_column": target_column,
                "decision_policy": "oof_business_cost_minimization",
            }
        )
        mlflow.log_params(
            {
                "campaign_name": campaign_name,
                "dataset_label": dataset_label,
                "target_column": target_column,
                "id_column": id_column,
                "drop_columns": ",".join(drop_columns),
                "candidate_models": ",".join(selected_model_names),
                "sampling_strategies": ",".join(sampling_modes),
                "sample_size": int(sample_size) if sample_size is not None else int(len(experiment_frame)),
                "cv_folds": int(cv_folds),
                "n_jobs": int(settings.training.n_jobs),
                "test_size": float(settings.dataset.test_size),
                "fn_cost": float(settings.business.fn_cost),
                "fp_cost": float(settings.business.fp_cost),
                "shap_sample_size": int(shap_sample_size),
                "local_explanations": int(local_explanations),
                "top_features": int(top_features),
            }
        )
        mlflow.log_dict({"pipeline_steps": PIPELINE_STEPS}, "pipeline_overview.json")

    for model_name in selected_model_names:
        if model_name not in available_models:
            raise ValueError(f"Unknown model name: {model_name}")

        if enable_mlflow:
            with mlflow.start_run(run_name=model_name, nested=True) as candidate_run:
                artifacts = _benchmark_single_model(
                    available_models[model_name],
                    x_train=x_train,
                    y_train=y_train,
                    id_train=id_train,
                    x_holdout=x_holdout,
                    y_holdout=y_holdout,
                    id_holdout=id_holdout,
                    settings=settings,
                    cv_folds=cv_folds,
                    output_dir=destination,
                )
                artifacts.result = replace(artifacts.result, run_id=candidate_run.info.run_id)
                _log_candidate_run(
                    artifacts,
                    output_dir=destination,
                    x_example=x_holdout,
                    campaign_name=campaign_name,
                )
        else:
            artifacts = _benchmark_single_model(
                available_models[model_name],
                x_train=x_train,
                y_train=y_train,
                id_train=id_train,
                x_holdout=x_holdout,
                y_holdout=y_holdout,
                id_holdout=id_holdout,
                settings=settings,
                cv_folds=cv_folds,
                output_dir=destination,
            )

        results.append(artifacts.result)
        artifacts_by_model[model_name] = artifacts

    results_frame = pd.DataFrame([asdict(result) for result in results])
    results_frame["best_params"] = results_frame["best_params"].apply(
        lambda params: json.dumps(_jsonable(params))
    )
    results_frame = results_frame.sort_values(
        ["holdout_business_cost", "holdout_average_precision", "holdout_roc_auc"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    best_model_name = results_frame.iloc[0]["model_name"]
    performance_summary = _build_model_performance_summary(
        results_frame,
        available_models,
        best_model_name=best_model_name,
    )
    cv_summary = _build_cv_summary(results_frame, best_model_name=best_model_name)
    holdout_summary = _build_holdout_summary(results_frame, best_model_name=best_model_name)
    threshold_summary = _build_decision_threshold_summary(results_frame, best_model_name=best_model_name)
    best_model_summary = _build_best_model_summary(performance_summary, best_model_name=best_model_name)
    root_run_id = mlflow.active_run().info.run_id if enable_mlflow and mlflow.active_run() else None
    mlflow_runs = _build_mlflow_runs_summary(
        results_frame,
        campaign_name=campaign_name,
        best_model_name=best_model_name,
        root_run_id=root_run_id,
    )
    results_frame.to_csv(destination / "benchmark_results.csv", index=False)
    performance_summary.to_csv(destination / "model_performance_summary.csv", index=False)
    cv_summary.to_csv(destination / "cv_summary.csv", index=False)
    holdout_summary.to_csv(destination / "holdout_summary.csv", index=False)
    threshold_summary.to_csv(destination / "decision_threshold_summary.csv", index=False)
    best_model_summary.to_csv(destination / "best_model_summary.csv", index=False)
    mlflow_runs.to_csv(destination / "mlflow_runs.csv", index=False)

    best_artifacts = artifacts_by_model[best_model_name]
    best_result = best_artifacts.result
    best_model_spec = available_models[best_model_name]

    _export_all_model_diagnostics(
        destination,
        artifacts_by_model=artifacts_by_model,
        y_holdout=y_holdout,
    )

    final_pipeline = _build_pipeline(best_model_spec, features, settings)
    final_pipeline.set_params(**best_result.best_params)
    final_pipeline.fit(features, target)

    interpretability_dir = destination / "interpretability"
    export_feature_importance(
        final_pipeline,
        interpretability_dir,
        top_n=top_features,
    )
    export_shap_analysis(
        final_pipeline,
        x_holdout,
        id_holdout,
        interpretability_dir,
        sample_size=shap_sample_size,
        local_examples=local_explanations,
        max_display=top_features,
        random_state=settings.dataset.random_state,
    )

    if test_dataframe is not None:
        test_features = test_dataframe.drop(
            columns=[column for column in drop_columns if column in test_dataframe.columns],
            errors="ignore",
        )
        test_probabilities = final_pipeline.predict_proba(test_features)[:, 1]
        test_predictions = pd.DataFrame(
            {
                id_column: test_dataframe[id_column].to_numpy(),
                "default_probability": test_probabilities,
                "default_prediction": (test_probabilities >= best_result.threshold).astype(int),
            }
        )
        test_predictions.to_csv(destination / "best_model_test_predictions.csv", index=False)

    decision_threshold = {
        "model_name": best_model_name,
        "threshold": float(best_result.threshold),
        "selection_basis": "out_of_fold_business_cost_minimization",
        "fn_cost": float(settings.business.fn_cost),
        "fp_cost": float(settings.business.fp_cost),
    }
    (destination / "decision_threshold.json").write_text(
        json.dumps(decision_threshold, indent=2),
        encoding="utf-8",
    )

    registered_model_version = None
    if enable_mlflow:
        registered_model_version = _log_final_model(
            final_pipeline,
            features=features,
            best_result=best_result,
            register_model_name=register_model_name,
        )

    campaign_overview = _build_campaign_overview(
        campaign_name=campaign_name,
        created_at=created_at,
        dataset_label=dataset_label,
        dataset_path=dataset_path,
        target_column=target_column,
        id_column=id_column,
        drop_columns=drop_columns,
        selected_model_names=selected_model_names,
        sampling_strategies=sampling_modes,
        sample_size=sample_size,
        experiment_frame=experiment_frame,
        train_rows=len(x_train),
        holdout_rows=len(x_holdout),
        cv_folds=cv_folds,
        n_jobs=settings.training.n_jobs,
        fn_cost=settings.business.fn_cost,
        fp_cost=settings.business.fp_cost,
        enable_mlflow=enable_mlflow,
        root_run_id=root_run_id,
        registered_model_name=register_model_name,
        registered_model_version=registered_model_version,
        best_model_name=best_model_name,
        output_dir=destination,
        test_dataset_available=test_dataframe is not None,
    )
    campaign_overview.to_csv(destination / "campaign_overview.csv", index=False)

    metadata = {
        "campaign_name": campaign_name,
        "campaign_slug": campaign_slug,
        "created_at": created_at,
        "dataset_label": dataset_label,
        "dataset_path": dataset_path,
        "output_dir": destination.as_posix(),
        "pipeline_steps": PIPELINE_STEPS,
        "candidate_models": selected_model_names,
        "sampling_strategies": sampling_modes,
        "sample_size_requested": sample_size,
        "sampled_rows": int(len(experiment_frame)),
        "sampled_columns": int(experiment_frame.shape[1]),
        "train_rows": int(len(x_train)),
        "holdout_rows": int(len(x_holdout)),
        "target_rate": float(experiment_frame[target_column].mean()),
        "cv_folds": int(cv_folds),
        "n_jobs": int(settings.training.n_jobs),
        "mlflow_enabled": bool(enable_mlflow),
        "mlflow_root_run_id": root_run_id,
        "registered_model_name": register_model_name,
        "registered_model_version": registered_model_version,
        "best_model": _jsonable(asdict(best_result)),
    }
    (destination / "campaign_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    build_experiment_workbooks(destination, cleanup_csv=False)

    if enable_mlflow:
        mlflow.log_dict(metadata, "campaign_metadata.json")
        _log_experiment_artifacts(destination)

    _cleanup_experiment_csv_files(destination)
    return results_frame


def run_benchmark_experiment(
    dataframe: pd.DataFrame,
    *,
    settings: Settings,
    output_dir: str | Path,
    campaign_name: str,
    dataset_label: str,
    dataset_path: str,
    target_column: str,
    id_column: str,
    drop_columns: list[str],
    model_names: list[str] | None = None,
    sampling_strategies: list[str] | None = None,
    test_dataframe: pd.DataFrame | None = None,
    sample_size: int | None = None,
    cv_folds: int | None = None,
    shap_sample_size: int = 1_500,
    local_explanations: int = 3,
    top_features: int = 20,
    enable_mlflow: bool = True,
    mlflow_run_name: str | None = None,
    register_model_name: str | None = None,
) -> pd.DataFrame:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    effective_cv_folds = cv_folds or settings.training.cv_folds

    if enable_mlflow:
        configure_mlflow(settings)
        run_name = mlflow_run_name or campaign_name
        with mlflow.start_run(run_name=run_name):
            return _run_benchmark_body(
                dataframe,
                settings=settings,
                destination=destination,
                campaign_name=campaign_name,
                dataset_label=dataset_label,
                dataset_path=dataset_path,
                target_column=target_column,
                id_column=id_column,
                drop_columns=drop_columns,
                model_names=model_names,
                sampling_strategies=sampling_strategies,
                test_dataframe=test_dataframe,
                sample_size=sample_size,
                cv_folds=effective_cv_folds,
                shap_sample_size=shap_sample_size,
                local_explanations=local_explanations,
                top_features=top_features,
                enable_mlflow=True,
                register_model_name=register_model_name,
            )

    return _run_benchmark_body(
        dataframe,
        settings=settings,
        destination=destination,
        campaign_name=campaign_name,
        dataset_label=dataset_label,
        dataset_path=dataset_path,
        target_column=target_column,
        id_column=id_column,
        drop_columns=drop_columns,
        model_names=model_names,
        sampling_strategies=sampling_strategies,
        test_dataframe=test_dataframe,
        sample_size=sample_size,
        cv_folds=effective_cv_folds,
        shap_sample_size=shap_sample_size,
        local_explanations=local_explanations,
        top_features=top_features,
        enable_mlflow=False,
        register_model_name=register_model_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Home Credit model experiment: training, model comparison, "
            "threshold optimization, interpretability exports, and optional MLflow."
        )
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--data", default=None, help="Training dataset path.")
    parser.add_argument("--test-data", default=None, help="Optional test dataset path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports and exports.",
    )
    parser.add_argument(
        "--campaign-name",
        default=None,
        help="Optional human-readable campaign name used in reports and MLflow.",
    )
    parser.add_argument("--target", default=None, help="Target column name.")
    parser.add_argument("--id-column", default=None, help="Identifier column name.")
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Additional feature column to drop. Can be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Base model name to evaluate. Can be repeated. Defaults to all base models.",
    )
    parser.add_argument(
        "--sampling",
        action="append",
        choices=sorted(VALID_SAMPLING_STRATEGIES),
        default=[],
        help="Sampling strategy to evaluate. Can be repeated. Defaults to baseline only.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for faster experimentation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override the number of cross-validation folds.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Override joblib parallel workers. Use 1 for WSL-safe runs.",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=1_500,
        help="Maximum number of rows used for SHAP calculations.",
    )
    parser.add_argument(
        "--local-explanations",
        type=int,
        default=3,
        help="Number of highest-risk and lowest-risk clients exported locally.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="Maximum number of features displayed in plots and exports.",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Run locally without MLflow tracking.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional root MLflow run name.",
    )
    parser.add_argument(
        "--register-model-name",
        default=None,
        help="Optional MLflow Model Registry name for the final best model.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = load_settings(args.config)

    if args.n_jobs == 0 or (args.n_jobs is not None and args.n_jobs < -1):
        raise ValueError("`--n-jobs` must be -1 or a positive integer.")
    if args.n_jobs is not None:
        settings = replace(
            settings,
            training=replace(settings.training, n_jobs=args.n_jobs),
        )

    effective_model_names = args.model or None
    effective_sampling_strategies = list(dict.fromkeys(args.sampling or list(DEFAULT_SAMPLING_STRATEGIES)))
    effective_cv_folds = args.cv_folds or settings.training.cv_folds
    campaign_name = args.campaign_name or _build_default_campaign_name(
        effective_model_names,
        effective_sampling_strategies,
        args.sample_size,
        effective_cv_folds,
    )

    data_path = Path(args.data) if args.data else settings.dataset.default_train_path
    test_data_path = (
        Path(args.test_data)
        if args.test_data
        else settings.paths.processed_dir / "test_features.parquet"
    )
    target_column = args.target or settings.dataset.target_column
    id_column = args.id_column or settings.dataset.id_column
    drop_columns = [column for column in dict.fromkeys([id_column, *args.drop_column]) if column]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        date_prefix = pd.Timestamp.now().strftime("%Y%m%d")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        campaign_slug = _slugify_campaign_name(campaign_name)
        output_dir = (
            settings.paths.reports_dir
            / f"{date_prefix}_home_credit_experiments"
            / f"{timestamp}_{campaign_slug}"
        )

    dataframe = read_table(data_path)
    test_dataframe = read_table(test_data_path) if test_data_path.exists() else None

    results = run_benchmark_experiment(
        dataframe,
        settings=settings,
        output_dir=output_dir,
        campaign_name=campaign_name,
        dataset_label=data_path.name,
        dataset_path=data_path.as_posix(),
        target_column=target_column,
        id_column=id_column,
        drop_columns=drop_columns,
        model_names=effective_model_names,
        sampling_strategies=effective_sampling_strategies,
        test_dataframe=test_dataframe,
        sample_size=args.sample_size,
        cv_folds=args.cv_folds,
        shap_sample_size=args.shap_sample_size,
        local_explanations=args.local_explanations,
        top_features=args.top_features,
        enable_mlflow=not args.skip_mlflow,
        mlflow_run_name=args.mlflow_run_name,
        register_model_name=args.register_model_name,
    )

    print(results.to_string(index=False))
    print(f"Campaign: {campaign_name}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

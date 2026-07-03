from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from home_credit_mlops.data.io import read_table
from home_credit_mlops.eda.diagnostics import generate_home_credit_eda_artifacts
from home_credit_mlops.features.preprocessing import build_preprocessor, split_features_target
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.modeling.candidates import ModelSpec, get_model_specs
from home_credit_mlops.modeling.interpretability import export_feature_importance, export_shap_analysis
from home_credit_mlops.modeling.metrics import business_scorer, evaluate_threshold, find_best_threshold
from home_credit_mlops.reporting.excel import build_experiment_workbooks
from home_credit_mlops.settings import Settings, load_settings


@dataclass(frozen=True)
class BenchmarkRunResult:
    model_name: str
    best_params: dict[str, Any]
    threshold: float
    cv_business_cost: float
    cv_roc_auc: float
    cv_average_precision: float
    cv_accuracy: float
    cv_balanced_accuracy: float
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


def _build_pipeline(model_spec: ModelSpec, features: pd.DataFrame) -> Pipeline:
    preprocessor, _, _ = build_preprocessor(features)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_spec.estimator_factory()),
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
    pipeline = _build_pipeline(model_spec, x_train)
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
    _save_prediction_tables(output_dir / "predictions", model_spec.name, oof_predictions, holdout_predictions)

    return ModelBenchmarkArtifacts(
        result=result,
        best_estimator=search.best_estimator_,
        oof_predictions=oof_predictions,
        holdout_predictions=holdout_predictions,
    )


def run_benchmark_experiment(
    dataframe: pd.DataFrame,
    *,
    settings: Settings,
    output_dir: str | Path,
    target_column: str,
    id_column: str,
    drop_columns: list[str],
    model_names: list[str] | None = None,
    test_dataframe: pd.DataFrame | None = None,
    sample_size: int | None = None,
    cv_folds: int | None = None,
    association_sample_size: int = 100_000,
    shap_sample_size: int = 1_500,
    local_explanations: int = 3,
    top_features: int = 20,
    run_eda: bool = True,
) -> pd.DataFrame:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cv_folds = cv_folds or settings.training.cv_folds

    experiment_frame = _sample_training_frame(
        dataframe,
        sample_size=sample_size,
        target_column=target_column,
        random_state=settings.dataset.random_state,
    )
    if run_eda:
        generate_home_credit_eda_artifacts(
            experiment_frame,
            destination / "eda",
            target_column=target_column,
            association_sample_size=association_sample_size,
            top_associations=top_features,
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

    available_models = get_model_specs()
    selected_model_names = model_names or list(available_models.keys())
    results: list[BenchmarkRunResult] = []
    artifacts_by_model: dict[str, ModelBenchmarkArtifacts] = {}

    for model_name in selected_model_names:
        if model_name not in available_models:
            raise ValueError(f"Unknown model name: {model_name}")

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
    results_frame.to_csv(destination / "benchmark_results.csv", index=False)

    best_model_name = results_frame.iloc[0]["model_name"]
    best_artifacts = artifacts_by_model[best_model_name]
    best_result = best_artifacts.result
    best_model_spec = available_models[best_model_name]

    _plot_holdout_diagnostics(
        destination / "diagnostics",
        best_model_name,
        y_holdout.to_numpy(),
        best_artifacts.holdout_predictions["probability"].to_numpy(),
        best_result.threshold,
    )

    final_pipeline = _build_pipeline(best_model_spec, features)
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

    metadata = {
        "sampled_rows": int(len(experiment_frame)),
        "sampled_columns": int(experiment_frame.shape[1]),
        "train_rows": int(len(x_train)),
        "holdout_rows": int(len(x_holdout)),
        "target_rate": float(experiment_frame[target_column].mean()),
        "cv_folds": int(cv_folds),
        "best_model": _jsonable(asdict(best_result)),
    }
    (destination / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    build_experiment_workbooks(destination)

    return results_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a consolidated Home Credit experiment with EDA, benchmarking, and SHAP."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--data", default=None, help="Training dataset path.")
    parser.add_argument("--test-data", default=None, help="Optional test dataset path.")
    parser.add_argument("--output-dir", default=None, help="Output directory for reports and exports.")
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
        help="Model name to evaluate. Can be repeated. Defaults to all candidates.",
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
        "--association-sample-size",
        type=int,
        default=100_000,
        help="Maximum number of rows used for association plots.",
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
        "--skip-eda",
        action="store_true",
        help="Skip EDA artifact generation.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = load_settings(args.config)

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
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = settings.paths.reports_dir / "home_credit_experiments" / timestamp

    dataframe = read_table(data_path)
    test_dataframe = read_table(test_data_path) if test_data_path.exists() else None

    results = run_benchmark_experiment(
        dataframe,
        settings=settings,
        output_dir=output_dir,
        target_column=target_column,
        id_column=id_column,
        drop_columns=drop_columns,
        model_names=args.model or None,
        test_dataframe=test_dataframe,
        sample_size=args.sample_size,
        cv_folds=args.cv_folds,
        association_sample_size=args.association_sample_size,
        shap_sample_size=args.shap_sample_size,
        local_explanations=args.local_explanations,
        top_features=args.top_features,
        run_eda=not args.skip_eda,
    )

    print(results.to_string(index=False))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

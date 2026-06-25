from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import tempfile
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from home_credit_mlops.data.io import read_table
from home_credit_mlops.features.preprocessing import build_preprocessor, split_features_target
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.mlflow_utils import configure_mlflow, register_logged_model
from home_credit_mlops.modeling.candidates import ModelSpec, get_model_specs
from home_credit_mlops.modeling.metrics import business_scorer, evaluate_threshold, find_best_threshold
from home_credit_mlops.settings import Settings, load_settings


@dataclass(frozen=True)
class ModelRunResult:
    model_name: str
    run_id: str
    best_params: dict[str, Any]
    threshold: float
    cv_business_cost: float
    cv_roc_auc: float
    cv_accuracy: float
    test_business_cost: float
    test_business_score: float
    test_roc_auc: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    false_negatives: int
    false_positives: int


def _build_pipeline(model_spec: ModelSpec, features: pd.DataFrame) -> Pipeline:
    preprocessor, _, _ = build_preprocessor(features)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_spec.estimator_factory()),
        ]
    )


def _save_cv_results(search: GridSearchCV, model_name: str) -> Path:
    output_dir = Path(tempfile.mkdtemp(prefix="home-credit-cv-"))
    output_path = output_dir / f"{model_name}_cv_results.csv"
    pd.DataFrame(search.cv_results_).to_csv(output_path, index=False)
    return output_path


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


def _log_run_artifacts(search: GridSearchCV, result: ModelRunResult) -> None:
    mlflow.log_params(search.best_params_)
    mlflow.log_metrics(
        {
            "cv_business_cost": result.cv_business_cost,
            "cv_roc_auc": result.cv_roc_auc,
            "cv_accuracy": result.cv_accuracy,
            "threshold": result.threshold,
            "test_business_cost": result.test_business_cost,
            "test_business_score": result.test_business_score,
            "test_roc_auc": result.test_roc_auc,
            "test_accuracy": result.test_accuracy,
            "test_precision": result.test_precision,
            "test_recall": result.test_recall,
            "test_f1": result.test_f1,
            "test_false_negatives": result.false_negatives,
            "test_false_positives": result.false_positives,
        }
    )
    mlflow.log_dict(_jsonable(asdict(result)), "evaluation_summary.json")

    cv_results_path = _save_cv_results(search, result.model_name)
    mlflow.log_artifact(cv_results_path.as_posix(), artifact_path="cv")


def _train_single_model(
    model_spec: ModelSpec,
    dataframe: pd.DataFrame,
    *,
    settings: Settings,
    target_column: str,
    drop_columns: list[str],
) -> ModelRunResult:
    features, target = split_features_target(
        dataframe,
        target_column=target_column,
        drop_columns=drop_columns,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=settings.dataset.test_size,
        random_state=settings.dataset.random_state,
        stratify=target,
    )

    pipeline = _build_pipeline(model_spec, x_train)
    cv = StratifiedKFold(
        n_splits=settings.training.cv_folds,
        shuffle=True,
        random_state=settings.dataset.random_state,
    )
    scoring = {
        "business_score": partial(
            business_scorer,
            fn_cost=settings.business.fn_cost,
            fp_cost=settings.business.fp_cost,
            grid_size=settings.business.threshold_grid_size,
        ),
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_spec.param_grid,
        scoring=scoring,
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

    test_probabilities = search.best_estimator_.predict_proba(x_test)[:, 1]
    final_test_result = evaluate_threshold(
        np.asarray(y_test),
        np.asarray(test_probabilities),
        threshold=threshold_result.threshold,
        fn_cost=settings.business.fn_cost,
        fp_cost=settings.business.fp_cost,
    )

    signature = infer_signature(x_train.head(5), search.best_estimator_.predict_proba(x_train.head(5)))
    mlflow.sklearn.log_model(
        sk_model=search.best_estimator_,
        artifact_path="model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature,
        input_example=x_train.head(5),
    )

    result = ModelRunResult(
        model_name=model_spec.name,
        run_id=mlflow.active_run().info.run_id,
        best_params=search.best_params_,
        threshold=threshold_result.threshold,
        cv_business_cost=-float(search.best_score_),
        cv_roc_auc=float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
        cv_accuracy=float(search.cv_results_["mean_test_accuracy"][search.best_index_]),
        test_business_cost=final_test_result.business_cost,
        test_business_score=final_test_result.business_score,
        test_roc_auc=final_test_result.roc_auc,
        test_accuracy=final_test_result.accuracy,
        test_precision=final_test_result.precision,
        test_recall=final_test_result.recall,
        test_f1=final_test_result.f1,
        false_negatives=final_test_result.false_negatives,
        false_positives=final_test_result.false_positives,
    )
    _log_run_artifacts(search, result)
    return result


def _register_result(result: ModelRunResult, model_name: str) -> str:
    model_uri = f"runs:/{result.run_id}/model"
    return register_logged_model(model_uri, model_name)


def _build_results_frame(results: list[ModelRunResult]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(result) for result in results])
    frame["best_params"] = frame["best_params"].apply(json.dumps)
    return frame.sort_values(["test_business_cost", "test_roc_auc"], ascending=[True, False])


def compare_models(
    dataframe: pd.DataFrame,
    *,
    settings: Settings,
    target_column: str,
    drop_columns: list[str],
    model_names: list[str] | None = None,
    register_model_name: str | None = None,
) -> pd.DataFrame:
    configure_mlflow(settings)
    available_models = get_model_specs()
    selected_model_names = model_names or list(available_models.keys())
    results: list[ModelRunResult] = []

    with mlflow.start_run(run_name="model-comparison"):
        mlflow.set_tags({"stage": "comparison", "target_column": target_column})
        mlflow.log_param("candidate_models", ",".join(selected_model_names))

        for model_name in selected_model_names:
            if model_name not in available_models:
                raise ValueError(f"Unknown model name: {model_name}")

            model_spec = available_models[model_name]
            with mlflow.start_run(run_name=model_name, nested=True):
                result = _train_single_model(
                    model_spec,
                    dataframe,
                    settings=settings,
                    target_column=target_column,
                    drop_columns=drop_columns,
                )
                results.append(result)

        results_frame = _build_results_frame(results)
        output_dir = settings.paths.reports_dir / "model_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "comparison_results.csv"
        results_frame.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path.as_posix(), artifact_path="comparison")

        best_result = next(result for result in results if result.run_id == results_frame.iloc[0]["run_id"])
        mlflow.log_dict(_jsonable(asdict(best_result)), "best_model_summary.json")

        if register_model_name:
            version = _register_result(best_result, register_model_name)
            mlflow.log_param("registered_model_name", register_model_name)
            mlflow.log_param("registered_model_version", version)

    return results_frame


def train_one_model(
    dataframe: pd.DataFrame,
    *,
    settings: Settings,
    target_column: str,
    drop_columns: list[str],
    model_name: str,
    register_model_name: str | None = None,
) -> ModelRunResult:
    configure_mlflow(settings)
    available_models = get_model_specs()
    if model_name not in available_models:
        raise ValueError(f"Unknown model name: {model_name}")

    with mlflow.start_run(run_name=f"train-{model_name}"):
        mlflow.set_tag("stage", "training")
        result = _train_single_model(
            available_models[model_name],
            dataframe,
            settings=settings,
            target_column=target_column,
            drop_columns=drop_columns,
        )
        if register_model_name:
            version = _register_result(result, register_model_name)
            mlflow.log_param("registered_model_name", register_model_name)
            mlflow.log_param("registered_model_version", version)

    return result


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", required=False, default=None, help="Training dataset path.")
    parser.add_argument("--target", default=None, help="Target column name.")
    parser.add_argument("--id-column", default=None, help="Identifier column to drop before training.")
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Additional feature column to drop. Can be repeated.",
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--register-model-name", default=None)
    return parser


def compare_models_main() -> None:
    configure_logging()
    parser = _common_parser("Compare candidate models with MLflow tracking.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model name to evaluate. Can be repeated. Defaults to all candidates.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    data_path = Path(args.data) if args.data else settings.dataset.default_train_path
    target_column = args.target or settings.dataset.target_column
    id_column = args.id_column or settings.dataset.id_column
    drop_columns = [column for column in dict.fromkeys([id_column, *args.drop_column]) if column]

    dataframe = read_table(data_path)
    results = compare_models(
        dataframe,
        settings=settings,
        target_column=target_column,
        drop_columns=drop_columns,
        model_names=args.model or None,
        register_model_name=args.register_model_name,
    )

    print(results.to_string(index=False))


def train_model_main() -> None:
    configure_logging()
    parser = _common_parser("Train a single model with MLflow tracking.")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(get_model_specs().keys()),
        help="Candidate model name.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    data_path = Path(args.data) if args.data else settings.dataset.default_train_path
    target_column = args.target or settings.dataset.target_column
    id_column = args.id_column or settings.dataset.id_column
    drop_columns = [column for column in dict.fromkeys([id_column, *args.drop_column]) if column]

    dataframe = read_table(data_path)
    result = train_one_model(
        dataframe,
        settings=settings,
        target_column=target_column,
        drop_columns=drop_columns,
        model_name=args.model,
        register_model_name=args.register_model_name,
    )

    print(pd.DataFrame([asdict(result)]).to_string(index=False))

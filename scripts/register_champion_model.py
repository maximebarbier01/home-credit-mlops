"""Enregistre rapidement le champion deja selectionne dans le Model Registry MLflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature

from home_credit_mlops.data.io import read_table
from home_credit_mlops.features.preprocessing import split_features_target
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.mlflow_utils import configure_mlflow, register_logged_model
from home_credit_mlops.modeling.benchmark import build_model_pipeline
from home_credit_mlops.modeling.candidates import build_candidate_model_specs
from home_credit_mlops.modeling.serving import CreditScoringModel
from home_credit_mlops.settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)

DEFAULT_REGISTERED_MODEL_NAME = "home-credit-scoring"
DEFAULT_SOURCE_CAMPAIGN = "lgbm_smote_full_cv5"
REPORTS_ROOT = Path("reports")


def _candidate_name(model_name: str, sampling_strategy: str) -> str:
    if sampling_strategy == "baseline":
        return model_name
    return f"{model_name}__{sampling_strategy}"


def _find_latest_campaign_metadata(reports_root: Path, campaign_name: str) -> Path | None:
    """Retrouve le campaign_metadata.json le plus recent pour cette campagne.

    Chaque run de campagne (`run_home_credit_experiment.py`) ecrit un
    `campaign_metadata.json` contenant le champion retenu (seuil metier et
    meilleurs hyperparametres). On evite ainsi de dupliquer ces valeurs en
    dur ici, ce qui les desynchroniserait silencieusement d'une prochaine
    campagne.
    """
    candidates: list[tuple[str, Path]] = []
    for metadata_path in reports_root.glob("*/*/campaign_metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("campaign_name") != campaign_name or "best_model" not in metadata:
            continue
        candidates.append((str(metadata.get("created_at", "")), metadata_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _load_champion_from_campaign(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    best_model = metadata["best_model"]
    return {
        "model_name": best_model["base_model_name"],
        "sampling_strategy": best_model["sampling_strategy"],
        "threshold": float(best_model["threshold"]),
        "params": dict(best_model["best_params"]),
        "campaign_name": metadata.get("campaign_name"),
        "source_path": metadata_path.as_posix(),
    }


def _parse_param_value(value: str) -> Any:
    lower_value = value.lower()
    if lower_value == "none":
        return None
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _parse_param_overrides(raw_params: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for raw_param in raw_params:
        if "=" not in raw_param:
            raise ValueError(
                "Each --param value must use the form step__parameter=value, "
                f"got: {raw_param!r}"
            )
        key, value = raw_param.split("=", 1)
        params[key] = _parse_param_value(value)
    return params


def _resolve_data_path(settings: Settings, data_path: str | None) -> Path:
    if data_path is None:
        return settings.dataset.default_train_path
    return Path(data_path).expanduser().resolve()


def _load_training_data(
    *,
    settings: Settings,
    data_path: Path,
    target_column: str,
    id_column: str,
    extra_drop_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    dataframe = read_table(data_path)
    features, target = split_features_target(
        dataframe,
        target_column=target_column,
        drop_columns=[id_column, *extra_drop_columns],
    )
    return features, target


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the selected champion once and register a business-aware "
            "MLflow pyfunc model without rerunning GridSearchCV."
        )
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--drop-column", action="append", default=[])
    parser.add_argument(
        "--model",
        default=None,
        help="Override the base model name. Defaults to the champion found in campaign artifacts.",
    )
    parser.add_argument(
        "--sampling",
        default=None,
        help="Override the sampling strategy. Defaults to the champion found in campaign artifacts.",
    )
    parser.add_argument(
        "--business-threshold",
        type=float,
        default=None,
        help="Override the business threshold. Defaults to the one found in campaign artifacts.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Pipeline parameter override, for example: model__n_estimators=500",
    )
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument(
        "--register-model-name",
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help="MLflow registered model name.",
    )
    parser.add_argument(
        "--source-campaign",
        default=DEFAULT_SOURCE_CAMPAIGN,
        help="Campaign name whose most recent campaign_metadata.json is used as champion source.",
    )
    parser.add_argument(
        "--source-report-dir",
        default=None,
        help=(
            "Explicit path to a campaign report directory (containing "
            "campaign_metadata.json). Overrides --source-campaign auto-discovery."
        ),
    )
    return parser


def _resolve_champion_params(
    *,
    champion_from_campaign: dict[str, Any] | None,
    model_name: str,
    sampling_strategy: str,
    raw_param_overrides: list[str],
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if (
        champion_from_campaign is not None
        and champion_from_campaign["model_name"] == model_name
        and champion_from_campaign["sampling_strategy"] == sampling_strategy
    ):
        params.update(champion_from_campaign["params"])
    params.update(_parse_param_overrides(raw_param_overrides))
    return params


def _resolve_threshold(
    *,
    champion_from_campaign: dict[str, Any] | None,
    model_name: str,
    sampling_strategy: str,
    business_threshold: float | None,
    parser: argparse.ArgumentParser,
) -> float:
    if business_threshold is not None:
        return business_threshold

    if (
        champion_from_campaign is not None
        and champion_from_campaign["model_name"] == model_name
        and champion_from_campaign["sampling_strategy"] == sampling_strategy
    ):
        return champion_from_campaign["threshold"]

    parser.error(
        f"No known business threshold for {model_name}/{sampling_strategy}. Provide "
        "--business-threshold, or point --source-campaign/--source-report-dir to a "
        "campaign whose champion matches this model/sampling."
    )
    raise AssertionError("unreachable")


def main() -> None:
    configure_logging()
    parser = _build_argument_parser()
    args = parser.parse_args()

    settings = load_settings(args.config)
    configure_mlflow(settings)

    source_metadata_path = (
        Path(args.source_report_dir) / "campaign_metadata.json"
        if args.source_report_dir
        else _find_latest_campaign_metadata(REPORTS_ROOT, args.source_campaign)
    )
    champion_from_campaign: dict[str, Any] | None = None
    if source_metadata_path is not None and source_metadata_path.exists():
        champion_from_campaign = _load_champion_from_campaign(source_metadata_path)
        LOGGER.info(
            "Champion artifacts found in %s (model=%s, sampling=%s, threshold=%s)",
            champion_from_campaign["source_path"],
            champion_from_campaign["model_name"],
            champion_from_campaign["sampling_strategy"],
            champion_from_campaign["threshold"],
        )
    else:
        LOGGER.warning(
            "No campaign_metadata.json found for campaign %r under %s ; "
            "falling back to explicit --model/--sampling/--business-threshold/--param.",
            args.source_campaign,
            REPORTS_ROOT,
        )

    model_name = args.model or (champion_from_campaign["model_name"] if champion_from_campaign else None)
    sampling_strategy = args.sampling or (
        champion_from_campaign["sampling_strategy"] if champion_from_campaign else None
    )
    if model_name is None or sampling_strategy is None:
        parser.error(
            "Could not resolve a champion model/sampling from campaign artifacts. "
            "Provide --model and --sampling explicitly, or point --source-campaign / "
            "--source-report-dir to a valid campaign report."
        )

    target_column = args.target or settings.dataset.target_column
    id_column = args.id_column or settings.dataset.id_column
    data_path = _resolve_data_path(settings, args.data)
    threshold = _resolve_threshold(
        champion_from_campaign=champion_from_campaign,
        model_name=model_name,
        sampling_strategy=sampling_strategy,
        business_threshold=args.business_threshold,
        parser=parser,
    )
    pipeline_params = _resolve_champion_params(
        champion_from_campaign=champion_from_campaign,
        model_name=model_name,
        sampling_strategy=sampling_strategy,
        raw_param_overrides=args.param,
    )

    candidates = build_candidate_model_specs(
        model_names=[model_name],
        sampling_strategies=[sampling_strategy],
    )
    candidate_key = _candidate_name(model_name, sampling_strategy)
    model_spec = candidates[candidate_key]

    LOGGER.info("Loading prepared dataset from %s", data_path)
    features, target = _load_training_data(
        settings=settings,
        data_path=data_path,
        target_column=target_column,
        id_column=id_column,
        extra_drop_columns=args.drop_column,
    )

    LOGGER.info(
        "Refitting champion %s with %s rows and %s features",
        candidate_key,
        len(features),
        features.shape[1],
    )
    pipeline = build_model_pipeline(model_spec, features, target, settings)
    if pipeline_params:
        pipeline.set_params(**pipeline_params)
    pipeline.fit(features, target)

    business_model = CreditScoringModel(
        pipeline=pipeline,
        business_threshold=threshold,
    )
    input_example = features.head(min(5, len(features))).copy()
    output_example = business_model.predict(context=None, model_input=input_example)
    signature = infer_signature(input_example, output_example)

    run_name = args.mlflow_run_name or f"register_{candidate_key}_business_response"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "stage": "champion_registration",
                "registration_mode": "refit_known_champion",
                "model_name": model_name,
                "sampling_strategy": sampling_strategy,
                "source_campaign": args.source_campaign,
            }
        )
        mlflow.log_params(
            {
                "candidate_model": candidate_key,
                "model_name": model_name,
                "sampling_strategy": sampling_strategy,
                "data_path": data_path.as_posix(),
                "target_column": target_column,
                "id_column": id_column,
                "drop_columns": ",".join([id_column, *args.drop_column]),
                "business_threshold": threshold,
                "fn_cost": settings.business.fn_cost,
                "fp_cost": settings.business.fp_cost,
                "source_report": (
                    champion_from_campaign["source_path"] if champion_from_campaign else "manual_override"
                ),
                **pipeline_params,
            }
        )
        mlflow.log_metrics(
            {
                "training_rows": float(len(features)),
                "training_features": float(features.shape[1]),
                "training_default_rate": float(target.mean()),
            }
        )
        mlflow.log_dict(
            {
                "candidate_model": candidate_key,
                "registered_model_name": args.register_model_name,
                "business_threshold": threshold,
                "pipeline_params": pipeline_params,
                "source_campaign": args.source_campaign,
                "source_report": (
                    champion_from_campaign["source_path"] if champion_from_campaign else "manual_override"
                ),
                "response_example": json.loads(output_example.to_json(orient="records")),
            },
            "champion_registration_summary.json",
        )

        model_info = mlflow.pyfunc.log_model(
            name="final_model",
            python_model=business_model,
            signature=signature,
            input_example=input_example,
            metadata={
                "business_threshold": float(threshold),
                "positive_class": 1,
                "positive_class_meaning": "default",
                "credit_decision_when_positive": "refused",
                "registration_mode": "refit_known_champion",
            },
        )
        version = register_logged_model(model_info.model_uri, args.register_model_name)
        mlflow.log_param("registered_model_version", version)

    print(f"MLflow run ID: {run.info.run_id}")
    print(f"Registered model: {args.register_model_name}")
    print(f"Registered version: {version}")
    print(f"Model URI: models:/{args.register_model_name}/{version}")


if __name__ == "__main__":
    main()

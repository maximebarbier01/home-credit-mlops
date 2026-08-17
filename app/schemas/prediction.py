"""Schemas Pydantic de prediction et validation d'entree.

La signature du modele MLflow reste la source de verite : le modele de
requete FastAPI est construit dynamiquement au demarrage a partir des
colonnes reellement attendues par le champion servi.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import mlflow.types
import pandas as pd
from mlflow.types import DataType
from pydantic import BaseModel, ConfigDict, create_model, field_validator

REQUEST_MODEL_NAME = "CreditScoringRequest"

BINARY_FLAG_COLUMNS = [
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    *[f"FLAG_DOCUMENT_{i}" for i in range(2, 22)],
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "DAYS_EMPLOYED_ANOM",
]

EXT_SOURCE_UNIT_INTERVAL_COLUMNS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCES_MEAN",
    "EXT_SOURCES_MIN",
    "EXT_SOURCES_MAX",
]

PREDICTION_RESPONSE_EXAMPLE = {
    "default_probability": 0.37,
    "business_threshold": 0.220331353025222,
    "predicted_default": 1,
    "credit_decision": "refused",
}


class PredictionResponse(BaseModel):
    """Reponse metier retournee par l'API."""

    model_config = ConfigDict(json_schema_extra={"example": PREDICTION_RESPONSE_EXAMPLE})

    default_probability: float
    business_threshold: float
    predicted_default: int
    credit_decision: Literal["approved", "refused"]


def _bounded_validator(field_name: str, *, predicate: Callable[[Any], bool], message: str):
    """Construit un validateur Pydantic pour une borne metier explicite."""

    def _validate(cls, value):  # noqa: ANN001 - signature imposee par Pydantic
        if value is not None and not predicate(value):
            raise ValueError(message)
        return value

    return field_validator(field_name)(classmethod(_validate))


def business_rule_validators() -> list[tuple[str, str, Any]]:
    """Validateurs pour les champs explicitement cites par la consigne."""

    return [
        (
            "AGE_YEARS",
            "validate_age_years",
            _bounded_validator(
                "AGE_YEARS",
                predicate=lambda value: 18 <= value < 100,
                message="AGE_YEARS must be between 18 and 100.",
            ),
        ),
        (
            "AMT_INCOME_TOTAL",
            "validate_income",
            _bounded_validator(
                "AMT_INCOME_TOTAL",
                predicate=lambda value: value > 0,
                message="AMT_INCOME_TOTAL must be strictly positive.",
            ),
        ),
        (
            "AMT_CREDIT",
            "validate_credit_amount",
            _bounded_validator(
                "AMT_CREDIT",
                predicate=lambda value: value > 0,
                message="AMT_CREDIT must be strictly positive.",
            ),
        ),
        (
            "CNT_CHILDREN",
            "validate_children_count",
            _bounded_validator(
                "CNT_CHILDREN",
                predicate=lambda value: value >= 0,
                message="CNT_CHILDREN cannot be negative.",
            ),
        ),
        (
            "CNT_FAM_MEMBERS",
            "validate_family_members",
            _bounded_validator(
                "CNT_FAM_MEMBERS",
                predicate=lambda value: value >= 0,
                message="CNT_FAM_MEMBERS cannot be negative.",
            ),
        ),
    ]


def plausible_range_validators() -> list[tuple[str, str, Any]]:
    """Validateurs pour les bornes connues et non ambigues du dataset."""

    validators: list[tuple[str, str, Any]] = []

    for field_name in BINARY_FLAG_COLUMNS:
        validators.append(
            (
                field_name,
                f"validate_binary_{field_name.lower()}",
                _bounded_validator(
                    field_name,
                    predicate=lambda value: value in (0, 1),
                    message=f"{field_name} must be 0 or 1.",
                ),
            )
        )

    for field_name in EXT_SOURCE_UNIT_INTERVAL_COLUMNS:
        validators.append(
            (
                field_name,
                f"validate_unit_interval_{field_name.lower()}",
                _bounded_validator(
                    field_name,
                    predicate=lambda value: 0.0 <= value <= 1.0,
                    message=f"{field_name} must be between 0 and 1.",
                ),
            )
        )

    validators.extend(
        [
            (
                "EXT_SOURCES_NA_COUNT",
                "validate_ext_sources_na_count",
                _bounded_validator(
                    "EXT_SOURCES_NA_COUNT",
                    predicate=lambda value: 0 <= value <= 3,
                    message="EXT_SOURCES_NA_COUNT must be between 0 and 3.",
                ),
            ),
            (
                "HOUR_APPR_PROCESS_START",
                "validate_hour_appr_process_start",
                _bounded_validator(
                    "HOUR_APPR_PROCESS_START",
                    predicate=lambda value: 0 <= value <= 23,
                    message="HOUR_APPR_PROCESS_START must be between 0 and 23.",
                ),
            ),
        ]
    )

    for field_name in ("REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"):
        validators.append(
            (
                field_name,
                f"validate_{field_name.lower()}",
                _bounded_validator(
                    field_name,
                    predicate=lambda value: 1 <= value <= 3,
                    message=f"{field_name} must be between 1 and 3.",
                ),
            )
        )

    return validators


def _example_value_for_col_spec(col_spec: mlflow.types.ColSpec) -> Any:
    known_examples = {
        "AGE_YEARS": 35.0,
        "AMT_INCOME_TOTAL": 50_000.0,
        "AMT_CREDIT": 200_000.0,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3.0,
        "CODE_GENDER": "F",
    }
    if col_spec.name in known_examples:
        return known_examples[col_spec.name]
    if col_spec.type == DataType.string:
        return "example"
    if col_spec.type in {DataType.integer, DataType.long}:
        return 0
    if col_spec.type == DataType.boolean:
        return False
    return 0.0


def build_request_example(input_schema: mlflow.types.Schema) -> dict[str, Any]:
    """Genere un exemple Swagger coherent avec la signature MLflow."""

    return {
        col_spec.name: _example_value_for_col_spec(col_spec)
        for col_spec in input_schema.inputs
        if col_spec.required
    }


def build_request_model(
    input_schema: mlflow.types.Schema,
    extra_validators: list[tuple[str, str, Any]] | None = None,
) -> type[BaseModel]:
    """Construit dynamiquement le modele Pydantic de requete."""

    fields: dict[str, Any] = {}
    field_names: set[str] = set()
    for col_spec in input_schema.inputs:
        python_type = col_spec.type.to_python()
        field_names.add(col_spec.name)
        if col_spec.required:
            fields[col_spec.name] = (python_type, ...)
        else:
            fields[col_spec.name] = (Optional[python_type], None)

    applicable_validators = {
        validator_name: validator
        for target_field, validator_name, validator in (extra_validators or [])
        if target_field in field_names
    }

    return create_model(
        REQUEST_MODEL_NAME,
        __base__=BaseModel,
        __config__=ConfigDict(json_schema_extra={"example": build_request_example(input_schema)}),
        __validators__=applicable_validators,
        **fields,
    )


def coerce_frame_dtypes(frame: pd.DataFrame, input_schema: mlflow.types.Schema) -> pd.DataFrame:
    """Force les dtypes attendus par la signature MLflow avant prediction."""

    coerced = frame.copy()
    for col_spec in input_schema.inputs:
        if col_spec.type == DataType.string:
            continue
        coerced[col_spec.name] = coerced[col_spec.name].astype(col_spec.type.to_numpy())
    return coerced


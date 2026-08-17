"""Tests dedies aux validateurs de schemas.py, notamment plausible_range_validators
(bornes connues d'apres le dictionnaire de donnees Home Credit, au-dela des 5
champs cites nommement par la consigne)."""

from __future__ import annotations

import pytest
from mlflow.types import ColSpec, DataType, Schema
from pydantic import ValidationError

from app.schemas.prediction import (
    build_request_model,
    business_rule_validators,
    plausible_range_validators,
)


def _schema_with_fields(*names: str) -> Schema:
    type_by_field = {
        "FLAG_MOBIL": DataType.integer,
        "FLAG_DOCUMENT_3": DataType.integer,
        "REG_CITY_NOT_LIVE_CITY": DataType.integer,
        "EXT_SOURCE_2": DataType.float,
        "EXT_SOURCES_MEAN": DataType.float,
        "EXT_SOURCES_NA_COUNT": DataType.integer,
        "HOUR_APPR_PROCESS_START": DataType.integer,
        "REGION_RATING_CLIENT": DataType.integer,
        "REGION_RATING_CLIENT_W_CITY": DataType.integer,
    }
    return Schema([ColSpec(type_by_field[name], name, required=True) for name in names])


def _build_model(*fields: str):
    schema = _schema_with_fields(*fields)
    return build_request_model(
        schema,
        business_rule_validators() + plausible_range_validators(),
    )


@pytest.mark.parametrize("value", [0, 1])
def test_binary_flag_accepts_zero_or_one(value: int) -> None:
    model = _build_model("FLAG_MOBIL")
    instance = model(FLAG_MOBIL=value)
    assert instance.FLAG_MOBIL == value


@pytest.mark.parametrize("value", [-1, 2, 5])
def test_binary_flag_rejects_anything_else(value: int) -> None:
    model = _build_model("FLAG_DOCUMENT_3")
    with pytest.raises(ValidationError, match="must be 0 or 1"):
        model(FLAG_DOCUMENT_3=value)


def test_reg_not_live_city_is_treated_as_binary_flag() -> None:
    model = _build_model("REG_CITY_NOT_LIVE_CITY")
    with pytest.raises(ValidationError, match="must be 0 or 1"):
        model(REG_CITY_NOT_LIVE_CITY=3)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_ext_source_accepts_unit_interval(value: float) -> None:
    model = _build_model("EXT_SOURCE_2")
    instance = model(EXT_SOURCE_2=value)
    assert instance.EXT_SOURCE_2 == value


@pytest.mark.parametrize("value", [-0.1, 1.1, 5.0])
def test_ext_source_rejects_outside_unit_interval(value: float) -> None:
    model = _build_model("EXT_SOURCE_2")
    with pytest.raises(ValidationError, match="must be between 0 and 1"):
        model(EXT_SOURCE_2=value)


def test_hour_appr_process_start_rejects_out_of_range() -> None:
    model = _build_model("HOUR_APPR_PROCESS_START")
    with pytest.raises(ValidationError, match="must be between 0 and 23"):
        model(HOUR_APPR_PROCESS_START=99)


def test_region_rating_client_rejects_out_of_range() -> None:
    model = _build_model("REGION_RATING_CLIENT")
    with pytest.raises(ValidationError, match="must be between 1 and 3"):
        model(REGION_RATING_CLIENT=7)


def test_plausible_range_validators_are_dropped_when_field_absent() -> None:
    # Un schema qui ne contient pas les champs cibles ne doit pas planter
    # create_model (filtrage par field_names dans build_request_model).
    model = _build_model("EXT_SOURCE_2")
    assert "FLAG_MOBIL" not in model.model_fields

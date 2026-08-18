"""Chargement et préparation des logs de production simulée."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


def _parse_json_cell(value: Any) -> dict[str, Any] | None:
    """Normalise les colonnes JSON lues depuis SQLite/PostgreSQL."""

    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else {"payload": parsed}
    return None


def _parse_json_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    parsed = frame.copy()
    for column in columns:
        if column in parsed.columns:
            parsed[column] = parsed[column].map(_parse_json_cell)
    return parsed


def _read_table(database_url: str, table_name: str) -> pd.DataFrame:
    engine = create_engine(database_url)
    try:
        frame = pd.read_sql_query(f"SELECT * FROM {table_name}", engine)
    except SQLAlchemyError:
        return pd.DataFrame()

    if "created_at" in frame.columns:
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    return frame


def load_prediction_logs(database_url: str) -> pd.DataFrame:
    """Charge les prédictions réussies journalisées par l'API."""

    frame = _read_table(database_url, "prediction_logs")
    return _parse_json_columns(frame, ("request_payload", "response_payload"))


def load_api_call_logs(database_url: str) -> pd.DataFrame:
    """Charge les logs techniques de tous les appels HTTP."""

    frame = _read_table(database_url, "api_call_logs")
    return _parse_json_columns(frame, ("request_payload",))


def load_production_inputs(database_url: str) -> pd.DataFrame:
    """Charge les inputs modele stockes dans la table physique production_inputs."""

    frame = _read_table(database_url, "production_inputs")
    frame = _parse_json_columns(frame, ("input_payload",))
    if frame.empty:
        return pd.DataFrame()

    features = flatten_json_column(frame, "input_payload")
    metadata_columns = [
        column
        for column in ("id", "prediction_log_id", "created_at", "feature_count")
        if column in frame.columns
    ]
    return pd.concat([frame[metadata_columns], features], axis=1)


def load_production_outputs(database_url: str) -> pd.DataFrame:
    """Charge les outputs modele stockes dans la table physique production_outputs."""

    frame = _read_table(database_url, "production_outputs")
    frame = _parse_json_columns(frame, ("output_payload",))
    if frame.empty:
        return pd.DataFrame()

    outputs = flatten_json_column(frame, "output_payload")
    metadata_columns = [
        column
        for column in (
            "id",
            "prediction_log_id",
            "created_at",
            "default_probability",
            "business_threshold",
            "predicted_default",
            "credit_decision",
            "latency_ms",
        )
        if column in frame.columns
    ]
    outputs = outputs.drop(
        columns=[column for column in metadata_columns if column in outputs.columns],
        errors="ignore",
    )
    return pd.concat([frame[metadata_columns], outputs], axis=1)


def flatten_json_column(
    frame: pd.DataFrame,
    column: str,
    *,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Transforme une colonne de dictionnaires JSON en colonnes tabulaires."""

    if frame.empty or column not in frame.columns:
        return pd.DataFrame(index=frame.index)

    records = [payload or {} for payload in frame[column]]
    flattened = pd.json_normalize(records)
    flattened.index = frame.index
    if prefix:
        flattened = flattened.add_prefix(prefix)
    return flattened


def production_feature_frame(prediction_logs: pd.DataFrame) -> pd.DataFrame:
    """Retourne les inputs modèle au format DataFrame, une ligne par scoring réussi."""

    return flatten_json_column(prediction_logs, "request_payload")


def production_prediction_frame(prediction_logs: pd.DataFrame) -> pd.DataFrame:
    """Retourne les outputs modèle enrichis avec les métadonnées de stockage."""

    if prediction_logs.empty:
        return pd.DataFrame()

    response = flatten_json_column(prediction_logs, "response_payload")
    metadata_columns = [
        column
        for column in ("id", "created_at", "default_probability", "credit_decision", "latency_ms")
        if column in prediction_logs.columns
    ]
    response = response.drop(
        columns=[column for column in metadata_columns if column in response.columns],
        errors="ignore",
    )
    return pd.concat([prediction_logs[metadata_columns], response], axis=1)

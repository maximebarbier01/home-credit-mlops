"""Tests dedies a load_api_config, notamment l'absence de repli SQLite
implicite : PREDICTION_DB_URL doit etre explicite des que le logging est
actif (voir app/core/config.py)."""

from __future__ import annotations

import pytest

from app.core.config import load_api_config


def test_missing_prediction_db_url_raises_when_logging_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PREDICTION_DB_URL", raising=False)
    monkeypatch.delenv("PREDICTION_LOGGING_ENABLED", raising=False)
    monkeypatch.delenv("API_CALL_LOGGING_ENABLED", raising=False)

    with pytest.raises(RuntimeError, match="PREDICTION_DB_URL is not set"):
        load_api_config()


def test_missing_prediction_db_url_is_fine_when_logging_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PREDICTION_DB_URL", raising=False)
    monkeypatch.setenv("PREDICTION_LOGGING_ENABLED", "false")
    monkeypatch.setenv("API_CALL_LOGGING_ENABLED", "false")

    config = load_api_config()

    assert config.prediction_db_url is None
    assert config.prediction_logging_enabled is False
    assert config.api_call_logging_enabled is False


def test_prediction_db_url_is_read_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREDICTION_DB_URL", "postgresql+psycopg://user:pw@host:5432/db")

    config = load_api_config()

    assert config.prediction_db_url == "postgresql+psycopg://user:pw@host:5432/db"

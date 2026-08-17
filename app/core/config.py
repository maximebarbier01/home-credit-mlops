"""Configuration specifique a l'API de serving."""

from __future__ import annotations

from dataclasses import dataclass
import os

from home_credit_mlops.settings import PROJECT_ROOT


@dataclass(frozen=True)
class ApiConfig:
    """Regroupe les options API lues depuis l'environnement."""

    api_key: str | None
    prediction_db_url: str
    prediction_logging_enabled: bool
    api_call_logging_enabled: bool


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_api_config() -> ApiConfig:
    """Charge les options API sans coupler FastAPI au fichier TOML ML."""

    default_db_path = PROJECT_ROOT / "artifacts" / "production_predictions.db"
    return ApiConfig(
        api_key=os.environ.get("HOME_CREDIT_API_KEY"),
        prediction_db_url=os.environ.get(
            "PREDICTION_DB_URL",
            f"sqlite:///{default_db_path.as_posix()}",
        ),
        prediction_logging_enabled=_env_flag(
            "PREDICTION_LOGGING_ENABLED",
            default=True,
        ),
        api_call_logging_enabled=_env_flag(
            "API_CALL_LOGGING_ENABLED",
            default=True,
        ),
    )

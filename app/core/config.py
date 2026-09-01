"""Configuration spécifique a l'API de serving."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ApiConfig:
    """Regroupe les options API lues depuis l'environnement."""

    api_key: str | None
    prediction_db_url: str | None
    prediction_logging_enabled: bool
    api_call_logging_enabled: bool


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_api_config() -> ApiConfig:
    """Charge les options API sans coupler FastAPI au fichier TOML ML.

    PREDICTION_DB_URL n'a plus de valeur par defaut SQLite : un environnement
    ou le logging est actif (par defaut) doit pointer vers un PostgreSQL
    reel, explicite. Objectif : un seul type de base partout (local, Docker,
    Render), pas un SQLite ephemere en production qui perd son historique a
    chaque redemarrage de conteneur. Desactiver PREDICTION_LOGGING_ENABLED et
    API_CALL_LOGGING_ENABLED est la façon explicite de se passer de base.
    """

    prediction_logging_enabled = _env_flag("PREDICTION_LOGGING_ENABLED", default=True)
    api_call_logging_enabled = _env_flag("API_CALL_LOGGING_ENABLED", default=True)
    prediction_db_url = os.environ.get("PREDICTION_DB_URL")

    if prediction_db_url is None and (prediction_logging_enabled or api_call_logging_enabled):
        raise RuntimeError(
            "PREDICTION_DB_URL is not set. Point it at a real PostgreSQL instance "
            "(see .env.example / docs/guide_utilisation_complet.md section 4.2), or "
            "explicitly disable logging with PREDICTION_LOGGING_ENABLED=false and "
            "API_CALL_LOGGING_ENABLED=false."
        )

    return ApiConfig(
        api_key=os.environ.get("HOME_CREDIT_API_KEY"),
        prediction_db_url=prediction_db_url,
        prediction_logging_enabled=prediction_logging_enabled,
        api_call_logging_enabled=api_call_logging_enabled,
    )

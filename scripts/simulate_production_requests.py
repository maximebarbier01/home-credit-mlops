"""Simule du trafic de production vers l'API de scoring."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from mlflow.models import Model

from app.schemas.prediction import BINARY_FLAG_COLUMNS, EXT_SOURCE_UNIT_INTERVAL_COLUMNS
from app.services.model_service import resolve_model_source
from home_credit_mlops.data.io import read_table
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.settings import load_settings

LOGGER = logging.getLogger(__name__)
DROP_COLUMNS = {"SK_ID_CURR", "TARGET"}
POSITIVE_NUMERIC_DEFAULTS = {
    "AMT_INCOME_TOTAL": 50_000.0,
    "AMT_CREDIT": 200_000.0,
}
SAFE_NUMERIC_DEFAULTS = {
    "AGE_YEARS": 35.0,
    "CNT_CHILDREN": 0,
    "CNT_FAM_MEMBERS": 1.0,
    "EXT_SOURCES_NA_COUNT": 0,
    "HOUR_APPR_PROCESS_START": 12,
    "REGION_RATING_CLIENT": 2,
    "REGION_RATING_CLIENT_W_CITY": 2,
}


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    default_data_path = settings.paths.processed_dir / "test_features.parquet"

    parser = argparse.ArgumentParser(
        description=(
            "Envoie un échantillon de clients vers l'API /predict pour remplir "
            "la base de monitoring production."
        )
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--data", default=default_data_path.as_posix())
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HOME_CREDIT_API_KEY"),
        help="Clé API optionnelle. Défaut : HOME_CREDIT_API_KEY si définie.",
    )
    parser.add_argument(
        "--invalid-requests",
        type=int,
        default=0,
        help="Nombre de requêtes invalides ajoutées pour tester les logs 422.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Chemin optionnel pour sauvegarder les réponses en JSONL.",
    )
    return parser.parse_args()


def _to_jsonable(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_to_payload(row: pd.Series, *, drop_columns: set[str]) -> dict[str, Any]:
    payload = {
        column: _to_jsonable(value) for column, value in row.items() if column not in drop_columns
    }
    return payload


def _enforce_business_safe_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Aligne les donnees simulees avec les validations metier de l'API."""

    cleaned = frame.copy()

    for column, default in POSITIVE_NUMERIC_DEFAULTS.items():
        if column in cleaned:
            values = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned[column] = values.where(values > 0, default)

    if "AGE_YEARS" in cleaned:
        values = pd.to_numeric(cleaned["AGE_YEARS"], errors="coerce")
        cleaned["AGE_YEARS"] = values.where(
            (values >= 18) & (values < 100),
            SAFE_NUMERIC_DEFAULTS["AGE_YEARS"],
        )

    for column in ("CNT_CHILDREN", "CNT_FAM_MEMBERS"):
        if column in cleaned:
            values = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned[column] = values.where(values >= 0, SAFE_NUMERIC_DEFAULTS[column])

    if "EXT_SOURCES_NA_COUNT" in cleaned:
        values = pd.to_numeric(cleaned["EXT_SOURCES_NA_COUNT"], errors="coerce")
        cleaned["EXT_SOURCES_NA_COUNT"] = values.where(
            (values >= 0) & (values <= 3),
            SAFE_NUMERIC_DEFAULTS["EXT_SOURCES_NA_COUNT"],
        )

    if "HOUR_APPR_PROCESS_START" in cleaned:
        values = pd.to_numeric(cleaned["HOUR_APPR_PROCESS_START"], errors="coerce")
        cleaned["HOUR_APPR_PROCESS_START"] = values.where(
            (values >= 0) & (values <= 23),
            SAFE_NUMERIC_DEFAULTS["HOUR_APPR_PROCESS_START"],
        )

    for column in ("REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"):
        if column in cleaned:
            values = pd.to_numeric(cleaned[column], errors="coerce")
            cleaned[column] = values.where(
                (values >= 1) & (values <= 3), SAFE_NUMERIC_DEFAULTS[column]
            )

    for column in set(BINARY_FLAG_COLUMNS).intersection(cleaned.columns):
        values = pd.to_numeric(cleaned[column], errors="coerce").fillna(0)
        cleaned[column] = np.where(values >= 0.5, 1, 0)

    for column in set(EXT_SOURCE_UNIT_INTERVAL_COLUMNS).intersection(cleaned.columns):
        values = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = values.clip(lower=0.0, upper=1.0).fillna(0.0)

    return cleaned


def _resolve_required_columns() -> set[str] | None:
    """Lit la signature MLflow (sans charger le pickle complet) pour savoir
    quelles colonnes le schema Pydantic de l'API rejette si elles sont nulles.

    `Model.load()` ne lit que le petit fichier `MLmodel`, pas le modele
    pickle de 130 Mo (pas de `mlflow.pyfunc.load_model`) : rapide, et
    reutilise `resolve_model_source` (donc le cache local si deja
    telecharge) plutot que de dupliquer la logique de resolution.
    """

    try:
        settings = load_settings()
        local_dir = resolve_model_source(settings.serving)
        schema = Model.load(local_dir.as_posix()).signature.inputs
        return {col_spec.name for col_spec in schema.inputs if col_spec.required}
    except Exception:
        LOGGER.exception(
            "Could not resolve the model's required columns; falling back to "
            "filling every missing value (may erase real missingness patterns "
            "and inflate data drift artificially)."
        )
        return None


def _fallback_value_for_column(series: pd.Series) -> Any:
    """Valeur de repli stable pour un champ requis, calculee sur les valeurs
    non manquantes de la colonne (mediane pour le numerique, mode sinon)."""

    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        finite_values = series.replace([np.inf, -np.inf], np.nan).dropna()
        if finite_values.empty:
            return 0.0
        return finite_values.median()

    mode = series.dropna().mode()
    if mode.empty:
        return "missing"
    return mode.iloc[0]


def _prepare_valid_simulation_frame(
    frame: pd.DataFrame, *, required_columns: set[str] | None
) -> pd.DataFrame:
    """Nettoie les inputs avant simulation pour eviter des erreurs 422 non voulues.

    Ne remplit les NaN que pour les colonnes requises par le schema MLflow
    (`required_columns`, resolu via `_resolve_required_columns`) : la
    majorite des colonnes optionnelles ont de vraies valeurs manquantes a
    l'entrainement (ex. CC_*/BURO_* agregees, absentes si le client n'a pas
    de carte de credit). Les remplir ecraserait ce pattern de missingness
    reel et fausserait l'analyse de derive (PSI/KS gonfles artificiellement,
    sans rapport avec une vraie derive de production). Si la resolution du
    schema a echoue (`required_columns is None`), repli sur l'ancien
    comportement (tout remplir) pour ne jamais casser la simulation.
    """

    features = frame.drop(
        columns=[column for column in DROP_COLUMNS if column in frame.columns]
    ).copy()
    features = features.replace([np.inf, -np.inf], np.nan)

    columns_to_fill = (
        features.columns if required_columns is None else required_columns.intersection(features.columns)
    )
    for column in columns_to_fill:
        if features[column].isna().any():
            features[column] = features[column].fillna(_fallback_value_for_column(features[column]))

    return _enforce_business_safe_values(features)


def _post_json(
    api_url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    request = Request(api_url, data=body, headers=headers, method="POST")
    started_at = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            status_code = response.status
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8")
        status_code = exc.code
    except (URLError, TimeoutError) as exc:
        # TimeoutError (ex. cold start Render) peut survenir pendant la
        # lecture de la reponse, apres l'ouverture de la connexion : ce
        # n'est pas toujours enveloppe dans URLError par urllib. Sans ce
        # cas, une seule requete lente fait planter toute la simulation au
        # lieu de compter cette requete comme un echec et continuer.
        return {
            "status_code": None,
            "latency_ms": (time.perf_counter() - started_at) * 1000,
            "error": str(exc),
            "response": None,
        }

    try:
        parsed_response = json.loads(response_body) if response_body else None
    except json.JSONDecodeError:
        parsed_response = response_body

    return {
        "status_code": status_code,
        "latency_ms": (time.perf_counter() - started_at) * 1000,
        "error": None,
        "response": parsed_response,
    }


def _build_payloads(
    data_path: Path,
    *,
    sample_size: int,
    random_state: int,
    invalid_requests: int,
) -> list[dict[str, Any]]:
    frame = read_table(data_path)
    if frame.empty:
        return []

    required_columns = _resolve_required_columns()
    valid_features = _prepare_valid_simulation_frame(frame, required_columns=required_columns)
    sample = valid_features.sample(
        n=min(sample_size, len(frame)),
        random_state=random_state,
    )
    payloads = [_row_to_payload(row, drop_columns=DROP_COLUMNS) for _, row in sample.iterrows()]

    for index in range(min(invalid_requests, len(payloads))):
        invalid_payload = dict(payloads[index])
        invalid_payload.pop("AMT_INCOME_TOTAL", None)
        payloads.append(invalid_payload)

    return payloads


def main() -> None:
    configure_logging()
    args = parse_args()
    payloads = _build_payloads(
        Path(args.data),
        sample_size=args.sample_size,
        random_state=args.random_state,
        invalid_requests=args.invalid_requests,
    )

    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    successes = 0
    errors = 0
    output_stream = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        for index, payload in enumerate(payloads, start=1):
            result = _post_json(
                args.api_url,
                payload,
                timeout=args.timeout,
                api_key=args.api_key,
            )
            status_code = result["status_code"]
            if status_code is not None and 200 <= int(status_code) < 400:
                successes += 1
            else:
                errors += 1

            record = {
                "request_index": index,
                "status_code": status_code,
                "latency_ms": result["latency_ms"],
                "error": result["error"],
                "response": result["response"],
            }
            if output_stream is not None:
                output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            LOGGER.info(
                "Request %s/%s -> status=%s latency=%.1fms",
                index,
                len(payloads),
                status_code,
                result["latency_ms"],
            )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        if output_stream is not None:
            output_stream.close()

    LOGGER.info("Simulation finished: %s successes, %s errors", successes, errors)


if __name__ == "__main__":
    main()

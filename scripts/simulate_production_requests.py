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

from home_credit_mlops.data.io import read_table
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.settings import load_settings

LOGGER = logging.getLogger(__name__)


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
        column: _to_jsonable(value)
        for column, value in row.items()
        if column not in drop_columns
    }
    return payload


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
    except URLError as exc:
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

    sample = frame.sample(
        n=min(sample_size, len(frame)),
        random_state=random_state,
    )
    drop_columns = {"SK_ID_CURR", "TARGET"}
    payloads = [_row_to_payload(row, drop_columns=drop_columns) for _, row in sample.iterrows()]

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

"""Initialise la base SQLAlchemy utilisee pour les logs de production API."""

from __future__ import annotations

import argparse

from app.core.config import load_api_config
from app.db.database import init_db
from home_credit_mlops.logging_utils import configure_logging


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize the API production log database.")
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy URL. Defaults to PREDICTION_DB_URL or artifacts/production_predictions.db.",
    )
    return parser


def main() -> None:
    configure_logging()
    parser = _build_argument_parser()
    args = parser.parse_args()

    database_url = args.database_url or load_api_config().prediction_db_url
    init_db(database_url)
    print(f"Prediction log database initialized: {database_url}")


if __name__ == "__main__":
    main()

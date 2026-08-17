"""Exporte les logs de production stockés en base vers un classeur Excel."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path

import pandas as pd

from app.core.config import load_api_config
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.monitoring.production import (
    load_api_call_logs,
    load_prediction_logs,
    production_feature_frame,
    production_prediction_frame,
)
from home_credit_mlops.settings import load_settings

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporte les tables de logs API dans un classeur Excel."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="URL SQLAlchemy de la base de logs. Défaut : PREDICTION_DB_URL ou SQLite local.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin du classeur Excel. Défaut : reports/YYYYMMDD_home_credit_monitoring/...",
    )
    return parser.parse_args()


def default_output_path() -> Path:
    settings = load_settings()
    now = datetime.now()
    return (
        settings.paths.reports_dir
        / f"{now:%Y%m%d}_home_credit_monitoring"
        / f"{now:%Y%m%d_%H%M%S}_production_logs.xlsx"
    )


def _write_sheet(writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame) -> None:
    frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    worksheet = writer.book[sheet_name[:31]]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions


def main() -> None:
    configure_logging()
    args = parse_args()
    api_config = load_api_config()

    database_url = args.database_url or api_config.prediction_db_url
    output_path = Path(args.output) if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_calls = load_api_call_logs(database_url)
    prediction_logs = load_prediction_logs(database_url)
    production_inputs = production_feature_frame(prediction_logs)
    production_outputs = production_prediction_frame(prediction_logs)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _write_sheet(writer, "api_call_logs", api_calls)
        _write_sheet(writer, "prediction_logs", prediction_logs)
        _write_sheet(writer, "production_inputs", production_inputs)
        _write_sheet(writer, "production_outputs", production_outputs)

    LOGGER.info("Production logs exported to %s", output_path)


if __name__ == "__main__":
    main()

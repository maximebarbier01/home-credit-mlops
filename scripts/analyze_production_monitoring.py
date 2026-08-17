"""Analyse les logs de production simulée et génère un rapport de monitoring."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.core.config import load_api_config
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.monitoring.drift import DriftConfig
from home_credit_mlops.monitoring.report import (
    build_monitoring_report,
    default_monitoring_output_dir,
)
from home_credit_mlops.settings import load_settings

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère un rapport de monitoring production : métriques API, latence, "
            "distribution des scores et data drift."
        )
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="URL SQLAlchemy de la base de logs. Défaut : PREDICTION_DB_URL ou SQLite local.",
    )
    parser.add_argument(
        "--reference-data",
        default=None,
        help="Dataset de référence pour le drift. Défaut : data/processed/train_features.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie. Défaut : reports/YYYYMMDD_home_credit_monitoring/...",
    )
    parser.add_argument(
        "--top-drift-features",
        type=int,
        default=25,
        help="Nombre de variables affichées dans le graphique des dérives.",
    )
    parser.add_argument(
        "--latency-warning-ms",
        type=float,
        default=1000.0,
        help="Seuil d'alerte sur la latence p95.",
    )
    parser.add_argument(
        "--error-rate-warning",
        type=float,
        default=0.05,
        help="Seuil d'alerte sur le taux d'erreur HTTP.",
    )
    parser.add_argument(
        "--min-current-rows",
        type=int,
        default=30,
        help="Nombre minimal de lignes de production avant de qualifier un drift.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = load_settings()
    api_config = load_api_config()

    database_url = args.database_url or api_config.prediction_db_url
    reference_data = Path(args.reference_data) if args.reference_data else settings.dataset.default_train_path
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_monitoring_output_dir(settings.paths.reports_dir)
    )

    report = build_monitoring_report(
        database_url=database_url,
        reference_data_path=reference_data,
        output_dir=output_dir,
        target_column=settings.dataset.target_column,
        id_column=settings.dataset.id_column,
        top_drift_features=args.top_drift_features,
        latency_warning_ms=args.latency_warning_ms,
        error_rate_warning=args.error_rate_warning,
        drift_config=DriftConfig(min_current_rows=args.min_current_rows),
    )

    LOGGER.info("Monitoring workbook written to %s", report.workbook_path)
    LOGGER.info("Monitoring HTML report written to %s", report.html_path)
    LOGGER.info("Monitoring plots written: %s", ", ".join(path.name for path in report.plots))


if __name__ == "__main__":
    main()

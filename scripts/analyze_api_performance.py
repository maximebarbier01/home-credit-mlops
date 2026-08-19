"""Génère le rapport d'analyse de performance post-déploiement."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.core.config import load_api_config
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.performance.report import (
    build_performance_report,
    default_performance_output_dir,
)
from home_credit_mlops.settings import load_settings

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse les latences et goulots d'étranglement depuis les logs de production de l'API."
        )
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="URL SQLAlchemy de la base de logs. Défaut : PREDICTION_DB_URL ou SQLite local.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie. Défaut : reports/YYYYMMDD_home_credit_performance/...",
    )
    parser.add_argument(
        "--latency-warning-ms",
        type=float,
        default=1000.0,
        help="Seuil d'alerte sur la latence API p95.",
    )
    parser.add_argument(
        "--error-rate-warning",
        type=float,
        default=0.05,
        help="Seuil d'alerte sur le taux d'erreur HTTP.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = load_settings()
    api_config = load_api_config()

    database_url = args.database_url or api_config.prediction_db_url
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_performance_output_dir(settings.paths.reports_dir)
    )

    report = build_performance_report(
        database_url=database_url,
        output_dir=output_dir,
        latency_warning_ms=args.latency_warning_ms,
        error_rate_warning=args.error_rate_warning,
    )

    LOGGER.info("Performance workbook written to %s", report.workbook_path)
    LOGGER.info("Performance markdown report written to %s", report.markdown_path)


if __name__ == "__main__":
    main()

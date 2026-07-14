"""Configuration centralisee des logs pour les scripts et pipelines du projet."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure le format et le niveau des logs du projet."""

    logging.basicConfig(
        # Niveau minimal affiché : INFO, WARNING, ERROR et CRITICAL
        level=level,
        # Format de chaque message :
        # date | niveau | module d'origine | message
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

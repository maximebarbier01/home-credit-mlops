"""Configuration centralisee des logs pour les scripts et pipelines du projet."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formate les logs en JSON pour une collecte exploitable en production."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO, *, json_format: bool | None = None) -> None:
    """Configure le format et le niveau des logs du projet."""

    if json_format is None:
        json_format = os.environ.get("LOG_FORMAT", "").strip().lower() == "json"

    formatter: logging.Formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        # Niveau minimal affiché : INFO, WARNING, ERROR et CRITICAL
        level=level,
        handlers=[handler],
    )

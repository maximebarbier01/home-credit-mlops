"""Orchestration metier d'une demande de scoring."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from pydantic import BaseModel

from app.schemas.prediction import PredictionResponse
from app.services.model_service import ModelService

LOGGER = logging.getLogger(__name__)

PredictionRepository = Callable[[dict[str, Any], dict[str, Any], float], int | None]


class PredictionService:
    """Coordonne inference, chronometrage et journalisation optionnelle."""

    def __init__(
        self,
        model_service: ModelService,
        *,
        prediction_repository: PredictionRepository | None = None,
    ) -> None:
        self._model_service = model_service
        self._prediction_repository = prediction_repository

    def predict(self, payload: BaseModel | dict[str, Any]) -> PredictionResponse:
        request_payload = payload.model_dump() if isinstance(payload, BaseModel) else dict(payload)
        start = time.perf_counter()
        response = self._model_service.predict(request_payload)
        latency_ms = (time.perf_counter() - start) * 1000

        if self._prediction_repository is not None:
            try:
                self._prediction_repository(
                    request_payload,
                    response.model_dump(),
                    latency_ms,
                )
            except Exception:
                LOGGER.exception("Prediction logging failed; returning scoring response anyway.")

        return response


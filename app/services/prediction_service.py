"""Orchestration métier d'une demande de scoring."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from app.schemas.prediction import PredictionResponse
from app.services.model_service import ModelService

LOGGER = logging.getLogger(__name__)

PredictionRepository = Callable[[dict[str, Any], dict[str, Any], float], int | None]


@dataclass(frozen=True)
class PredictionResult:
    """Résultat complet d'une inférence, prêt pour réponse et journalisation."""

    request_payload: dict[str, Any]
    response: PredictionResponse
    latency_ms: float


class PredictionService:
    """Coordonne inférence, chronomètrage et journalisation optionnelle."""

    def __init__(
        self,
        model_service: ModelService,
        *,
        prediction_repository: PredictionRepository | None = None,
    ) -> None:
        self._model_service = model_service
        self._prediction_repository = prediction_repository

    def score(self, payload: BaseModel | dict[str, Any]) -> PredictionResult:
        """Calcule le score sans écrire en base, pour garder /predict rapide."""

        request_payload = payload.model_dump() if isinstance(payload, BaseModel) else dict(payload)
        start = time.perf_counter()
        response = self._model_service.predict(request_payload)
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            request_payload=request_payload,
            response=response,
            latency_ms=latency_ms,
        )

    def log_prediction(
        self,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Persiste une prédiction si la journalisation est activée."""

        if self._prediction_repository is not None:
            try:
                self._prediction_repository(
                    request_payload,
                    response_payload,
                    latency_ms,
                )
            except Exception:
                LOGGER.exception("La journalisation de la prédiction a échoué.")

    def predict(self, payload: BaseModel | dict[str, Any]) -> PredictionResponse:
        """Calcule le score et journalise en mode synchrone.

        Ce chemin reste utile pour les tests unitaires et les appels directs au
        service. La route FastAPI utilise `score()` puis planifie
        `log_prediction()` en tâche de fond.
        """

        result = self.score(payload)
        self.log_prediction(
            result.request_payload,
            result.response.model_dump(),
            result.latency_ms,
        )
        return result.response

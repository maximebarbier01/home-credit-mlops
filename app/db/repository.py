"""Operations de lecture/ecriture autour des logs de production."""

from __future__ import annotations

from typing import Any

from app.db.database import SessionLocal
from app.db.models import ApiCallLog, PredictionLog

MAX_ERROR_MESSAGE_LENGTH = 500


def save_prediction_log(
    request_payload: dict,
    response_payload: dict,
    latency_ms: float,
) -> int:
    """Enregistre une prediction et retourne son identifiant technique."""

    with SessionLocal() as session:
        log = PredictionLog(
            request_payload=request_payload,
            response_payload=response_payload,
            default_probability=float(response_payload["default_probability"]),
            credit_decision=str(response_payload["credit_decision"]),
            latency_ms=float(latency_ms),
        )
        session.add(log)
        session.commit()
        session.refresh(log)
        return int(log.id)


def save_api_call_log(
    *,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    request_payload: dict[str, Any] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    client_host: str | None = None,
    user_agent: str | None = None,
) -> int:
    """Enregistre un appel HTTP pour le monitoring opérationnel."""

    if error_message is not None:
        error_message = error_message[:MAX_ERROR_MESSAGE_LENGTH]

    with SessionLocal() as session:
        log = ApiCallLog(
            method=method,
            path=path,
            status_code=int(status_code),
            latency_ms=float(latency_ms),
            request_payload=request_payload,
            error_type=error_type,
            error_message=error_message,
            client_host=client_host,
            user_agent=user_agent,
        )
        session.add(log)
        session.commit()
        session.refresh(log)
        return int(log.id)

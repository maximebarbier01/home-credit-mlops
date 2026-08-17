"""Operations de lecture/ecriture autour des predictions journalisees."""

from __future__ import annotations

from app.db.database import SessionLocal
from app.db.models import PredictionLog


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


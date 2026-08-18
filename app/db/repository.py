"""Opérations de lecture/écriture autour des logs de production."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import ApiCallLog, PredictionLog, ProductionInput, ProductionOutput

MAX_ERROR_MESSAGE_LENGTH = 500


def save_prediction_log(
    request_payload: dict,
    response_payload: dict,
    latency_ms: float,
) -> int:
    """Enregistre une prédiction et retourne son identifiant technique."""

    with SessionLocal.begin() as session:
        log = PredictionLog(
            request_payload=request_payload,
            response_payload=response_payload,
            default_probability=float(response_payload["default_probability"]),
            credit_decision=str(response_payload["credit_decision"]),
            latency_ms=float(latency_ms),
        )
        session.add(log)
        session.flush()

        production_input = ProductionInput(
            prediction_log_id=int(log.id),
            input_payload=request_payload,
            feature_count=len(request_payload),
        )
        production_output = ProductionOutput(
            prediction_log_id=int(log.id),
            output_payload=response_payload,
            default_probability=float(response_payload["default_probability"]),
            business_threshold=float(response_payload["business_threshold"]),
            predicted_default=int(response_payload["predicted_default"]),
            credit_decision=str(response_payload["credit_decision"]),
            latency_ms=float(latency_ms),
        )
        session.add_all([production_input, production_output])
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


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    """Calcule un percentile simple sans dépendance SQL spécifique."""

    cleaned = sorted(value for value in values if not math.isnan(value))
    if not cleaned:
        return 0.0

    position = (len(cleaned) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(cleaned[int(position)])

    weight = position - lower
    return float(cleaned[lower] * (1 - weight) + cleaned[upper] * weight)


def get_monitoring_summary() -> dict[str, Any]:
    """Retourne un résumé opérationnel léger pour l'endpoint /monitoring/summary."""

    with SessionLocal() as session:
        api_calls = session.scalars(select(ApiCallLog)).all()
        predictions = session.scalars(select(PredictionLog)).all()

    total_api_calls = len(api_calls)
    predict_calls = sum(call.path == "/predict" for call in api_calls)
    error_calls = sum(call.status_code >= 400 for call in api_calls)
    latencies = [float(call.latency_ms) for call in api_calls if call.latency_ms is not None]
    latency_mean_ms = float(sum(latencies) / len(latencies)) if latencies else 0.0
    status_code_counts = Counter(str(call.status_code) for call in api_calls)

    decision_counts = Counter(log.credit_decision for log in predictions)
    prediction_count = len(predictions)
    refused_count = int(decision_counts.get("refused", 0))
    approved_count = int(decision_counts.get("approved", 0))

    return {
        "total_api_calls": total_api_calls,
        "predict_calls": int(predict_calls),
        "error_calls": int(error_calls),
        "error_rate": _rate(int(error_calls), total_api_calls),
        "latency_mean_ms": latency_mean_ms,
        "latency_p95_ms": _percentile(latencies, 0.95),
        "prediction_count": prediction_count,
        "refused_count": refused_count,
        "approved_count": approved_count,
        "refused_rate": _rate(refused_count, prediction_count),
        "status_code_counts": dict(status_code_counts),
        "decision_counts": dict(decision_counts),
    }

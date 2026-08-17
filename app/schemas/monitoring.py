"""Schémas Pydantic des endpoints de monitoring API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MonitoringSummaryResponse(BaseModel):
    """Résumé opérationnel exposé par l'API."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_api_calls": 125,
                "predict_calls": 100,
                "error_calls": 3,
                "error_rate": 0.024,
                "latency_mean_ms": 82.4,
                "latency_p95_ms": 210.7,
                "prediction_count": 100,
                "refused_count": 37,
                "approved_count": 63,
                "refused_rate": 0.37,
                "status_code_counts": {"200": 122, "422": 3},
                "decision_counts": {"approved": 63, "refused": 37},
            }
        }
    )

    total_api_calls: int
    predict_calls: int
    error_calls: int
    error_rate: float
    latency_mean_ms: float
    latency_p95_ms: float
    prediction_count: int
    refused_count: int
    approved_count: int
    refused_rate: float
    status_code_counts: dict[str, int]
    decision_counts: dict[str, int]

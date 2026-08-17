"""Indicateurs opérationnels calculés depuis les logs API."""

from __future__ import annotations

import pandas as pd


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_api_call_summary(api_calls: pd.DataFrame) -> pd.DataFrame:
    """Produit un résumé global des appels HTTP."""

    if api_calls.empty:
        return pd.DataFrame(
            [
                {
                    "total_calls": 0,
                    "predict_calls": 0,
                    "error_calls": 0,
                    "error_rate": 0.0,
                    "latency_mean_ms": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p95_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "latency_max_ms": 0.0,
                    "first_call_at": pd.NaT,
                    "last_call_at": pd.NaT,
                }
            ]
        )

    status_codes = api_calls["status_code"].astype(int)
    latencies = pd.to_numeric(api_calls["latency_ms"], errors="coerce").dropna()
    total_calls = len(api_calls)
    error_calls = int((status_codes >= 400).sum())
    predict_calls = int((api_calls["path"] == "/predict").sum()) if "path" in api_calls else 0

    return pd.DataFrame(
        [
            {
                "total_calls": total_calls,
                "predict_calls": predict_calls,
                "error_calls": error_calls,
                "error_rate": _safe_rate(error_calls, total_calls),
                "latency_mean_ms": float(latencies.mean()) if not latencies.empty else 0.0,
                "latency_p50_ms": float(latencies.quantile(0.50)) if not latencies.empty else 0.0,
                "latency_p95_ms": float(latencies.quantile(0.95)) if not latencies.empty else 0.0,
                "latency_p99_ms": float(latencies.quantile(0.99)) if not latencies.empty else 0.0,
                "latency_max_ms": float(latencies.max()) if not latencies.empty else 0.0,
                "first_call_at": api_calls["created_at"].min()
                if "created_at" in api_calls
                else pd.NaT,
                "last_call_at": api_calls["created_at"].max()
                if "created_at" in api_calls
                else pd.NaT,
            }
        ]
    )


def compute_status_code_summary(api_calls: pd.DataFrame) -> pd.DataFrame:
    """Compte les statuts HTTP observés."""

    if api_calls.empty or "status_code" not in api_calls:
        return pd.DataFrame(columns=["status_code", "count", "rate"])

    summary = (
        api_calls.assign(status_code=api_calls["status_code"].astype(int))
        .groupby("status_code", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("status_code")
    )
    summary["rate"] = summary["count"] / summary["count"].sum()
    return summary


def compute_latency_by_path(api_calls: pd.DataFrame) -> pd.DataFrame:
    """Résume les latences par endpoint."""

    if api_calls.empty or not {"path", "latency_ms"}.issubset(api_calls.columns):
        return pd.DataFrame(
            columns=[
                "path",
                "count",
                "latency_mean_ms",
                "latency_p95_ms",
                "latency_max_ms",
            ]
        )

    frame = api_calls.copy()
    frame["latency_ms"] = pd.to_numeric(frame["latency_ms"], errors="coerce")
    return (
        frame.groupby("path", dropna=False)["latency_ms"]
        .agg(
            count="count",
            latency_mean_ms="mean",
            latency_p95_ms=lambda values: values.quantile(0.95),
            latency_max_ms="max",
        )
        .reset_index()
        .sort_values("latency_p95_ms", ascending=False)
    )


def compute_prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Résume les scores et décisions retournés par le modèle."""

    if predictions.empty:
        return pd.DataFrame(
            [
                {
                    "prediction_count": 0,
                    "default_probability_mean": 0.0,
                    "default_probability_p50": 0.0,
                    "default_probability_p95": 0.0,
                    "refused_count": 0,
                    "approved_count": 0,
                    "refused_rate": 0.0,
                    "approved_rate": 0.0,
                }
            ]
        )

    probability = pd.to_numeric(predictions["default_probability"], errors="coerce").dropna()
    decision_counts = predictions["credit_decision"].value_counts(dropna=False)
    total = len(predictions)
    refused_count = int(decision_counts.get("refused", 0))
    approved_count = int(decision_counts.get("approved", 0))

    return pd.DataFrame(
        [
            {
                "prediction_count": total,
                "default_probability_mean": float(probability.mean()) if not probability.empty else 0.0,
                "default_probability_p50": float(probability.quantile(0.50))
                if not probability.empty
                else 0.0,
                "default_probability_p95": float(probability.quantile(0.95))
                if not probability.empty
                else 0.0,
                "refused_count": refused_count,
                "approved_count": approved_count,
                "refused_rate": _safe_rate(refused_count, total),
                "approved_rate": _safe_rate(approved_count, total),
            }
        ]
    )


def detect_operational_alerts(
    api_summary: pd.DataFrame,
    *,
    latency_warning_ms: float,
    error_rate_warning: float,
) -> pd.DataFrame:
    """Détecte des alertes simples et explicables sur les métriques API."""

    if api_summary.empty:
        return pd.DataFrame(columns=["severity", "metric", "observed_value", "threshold", "message"])

    row = api_summary.iloc[0]
    alerts: list[dict[str, object]] = []

    if float(row["error_rate"]) > error_rate_warning:
        alerts.append(
            {
                "severity": "warning",
                "metric": "error_rate",
                "observed_value": float(row["error_rate"]),
                "threshold": error_rate_warning,
                "message": "Le taux d'erreur HTTP dépasse le seuil de vigilance.",
            }
        )

    if float(row["latency_p95_ms"]) > latency_warning_ms:
        alerts.append(
            {
                "severity": "warning",
                "metric": "latency_p95_ms",
                "observed_value": float(row["latency_p95_ms"]),
                "threshold": latency_warning_ms,
                "message": "La latence p95 dépasse le seuil de vigilance.",
            }
        )

    return pd.DataFrame(
        alerts,
        columns=["severity", "metric", "observed_value", "threshold", "message"],
    )

"""Rapport de performance post-déploiement de l'API de scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from home_credit_mlops.monitoring.operational import (
    compute_api_call_summary,
    compute_latency_by_path,
    compute_status_code_summary,
)
from home_credit_mlops.monitoring.production import (
    load_api_call_logs,
    load_prediction_logs,
    load_production_outputs,
    production_prediction_frame,
)


@dataclass(frozen=True)
class PerformanceReport:
    """Chemins des livrables de performance générés."""

    output_dir: Path
    workbook_path: Path
    markdown_path: Path


def default_performance_output_dir(reports_dir: str | Path) -> Path:
    """Construit un dossier daté pour l'analyse de performance."""

    now = datetime.now()
    return (
        Path(reports_dir)
        / f"{now:%Y%m%d}_home_credit_performance"
        / f"{now:%Y%m%d_%H%M%S}_performance"
    )


def summarize_latency(
    frame: pd.DataFrame,
    *,
    source: str,
    latency_column: str = "latency_ms",
) -> pd.DataFrame:
    """Résume une colonne de latence en millisecondes."""

    if frame.empty or latency_column not in frame.columns:
        return pd.DataFrame(
            [
                {
                    "source": source,
                    "count": 0,
                    "latency_mean_ms": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p95_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "latency_max_ms": 0.0,
                }
            ]
        )

    latencies = pd.to_numeric(frame[latency_column], errors="coerce").dropna()
    if latencies.empty:
        return summarize_latency(pd.DataFrame(), source=source, latency_column=latency_column)

    return pd.DataFrame(
        [
            {
                "source": source,
                "count": int(latencies.count()),
                "latency_mean_ms": float(latencies.mean()),
                "latency_p50_ms": float(latencies.quantile(0.50)),
                "latency_p95_ms": float(latencies.quantile(0.95)),
                "latency_p99_ms": float(latencies.quantile(0.99)),
                "latency_max_ms": float(latencies.max()),
            }
        ]
    )


def build_optimization_decisions() -> pd.DataFrame:
    """Liste les optimisations appliquées ou écartées."""

    return pd.DataFrame(
        [
            {
                "decision": "Chargement du modèle au démarrage",
                "status": "conservé",
                "rationale": (
                    "Le modèle MLflow est chargé une seule fois dans le lifespan FastAPI, "
                    "ce qui évite un coût de chargement à chaque requête."
                ),
                "risk": "Faible, déjà couvert par les tests API.",
            },
            {
                "decision": "Journalisation PostgreSQL en tâche de fond",
                "status": "appliqué",
                "rationale": (
                    "Le score est renvoyé dès la fin de l'inférence. Les écritures "
                    "prediction_logs, production_inputs, production_outputs et api_call_logs "
                    "sont exécutées après l'envoi de la réponse HTTP."
                ),
                "risk": "Faible à modéré : la réponse n'est plus bloquée par une erreur de log.",
            },
            {
                "decision": "Pas de duplication des payloads valides dans api_call_logs",
                "status": "appliqué",
                "rationale": (
                    "Les inputs valides sont déjà stockés dans production_inputs. "
                    "api_call_logs conserve le payload complet surtout pour les erreurs 4xx/5xx."
                ),
                "risk": "Faible : la traçabilité métier reste portée par production_inputs.",
            },
            {
                "decision": "ONNX Runtime",
                "status": "écarté à ce stade",
                "rationale": (
                    "Le modèle servi est un pipeline Python/MLflow avec preprocessing et LightGBM. "
                    "Une conversion ONNX introduirait un risque de régression fonctionnelle et "
                    "doit être justifiée par un goulot modèle clairement mesuré."
                ),
                "risk": "À réévaluer si la latence modèle devient le goulot principal.",
            },
            {
                "decision": "CPU plutôt que GPU",
                "status": "retenu",
                "rationale": (
                    "LightGBM en inférence tabulaire est adapté au CPU. Le GPU ajouterait de la "
                    "complexité de déploiement pour un gain incertain sur des requêtes unitaires."
                ),
                "risk": "À réévaluer uniquement en cas de très fort volume ou batch scoring.",
            },
        ]
    )


def identify_bottlenecks(
    *,
    api_summary: pd.DataFrame,
    model_latency_summary: pd.DataFrame,
    latency_warning_ms: float,
    error_rate_warning: float,
) -> pd.DataFrame:
    """Déduit les points de vigilance depuis les métriques observées."""

    rows: list[dict[str, str | float]] = []
    api = api_summary.iloc[0]
    model = model_latency_summary.iloc[0]
    api_p95 = float(api.get("latency_p95_ms", 0.0))
    model_p95 = float(model.get("latency_p95_ms", 0.0))
    error_rate = float(api.get("error_rate", 0.0))

    if api_p95 > latency_warning_ms:
        rows.append(
            {
                "bottleneck": "Latence API p95 élevée",
                "severity": "high",
                "evidence": f"p95 API = {api_p95:.1f} ms > seuil {latency_warning_ms:.1f} ms",
                "recommendation": "Profiler le endpoint /predict et surveiller CPU/RAM du conteneur.",
            }
        )
    else:
        rows.append(
            {
                "bottleneck": "Latence API",
                "severity": "low",
                "evidence": f"p95 API = {api_p95:.1f} ms",
                "recommendation": "Configuration acceptable sur les seuils actuels.",
            }
        )

    if error_rate > error_rate_warning:
        rows.append(
            {
                "bottleneck": "Taux d'erreur HTTP",
                "severity": "high",
                "evidence": f"taux d'erreur = {error_rate:.1%}",
                "recommendation": "Distinguer erreurs de validation attendues et erreurs serveur.",
            }
        )

    if api_p95 and model_p95 / api_p95 >= 0.70:
        rows.append(
            {
                "bottleneck": "Inférence modèle",
                "severity": "moderate",
                "evidence": f"p95 modèle = {model_p95:.1f} ms pour p95 API = {api_p95:.1f} ms",
                "recommendation": "Étudier batching, simplification du preprocessing ou conversion ONNX.",
            }
        )
    elif api_p95:
        rows.append(
            {
                "bottleneck": "Overhead API / sérialisation / logs",
                "severity": "moderate",
                "evidence": f"p95 modèle = {model_p95:.1f} ms pour p95 API = {api_p95:.1f} ms",
                "recommendation": "Conserver les logs en tâche de fond et éviter les payloads dupliqués.",
            }
        )

    return pd.DataFrame(rows)


def _write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            safe_name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.book[safe_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions


def _metric(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame:
        return default
    return float(frame.iloc[0][column])


def _write_markdown_report(
    path: Path,
    *,
    api_summary: pd.DataFrame,
    model_latency_summary: pd.DataFrame,
    bottlenecks: pd.DataFrame,
    optimization_decisions: pd.DataFrame,
) -> None:
    lines = [
        "# Rapport d'analyse et d'optimisation des performances",
        "",
        "## Synthèse",
        "",
        (
            f"- Appels API analysés : {int(_metric(api_summary, 'total_calls'))} "
            f"dont {int(_metric(api_summary, 'predict_calls'))} appels `/predict`."
        ),
        f"- Taux d'erreur HTTP : {_metric(api_summary, 'error_rate'):.1%}.",
        f"- Latence API p95 : {_metric(api_summary, 'latency_p95_ms'):.1f} ms.",
        f"- Latence modèle p95 : {_metric(model_latency_summary, 'latency_p95_ms'):.1f} ms.",
        "",
        "## Goulots d'étranglement identifiés",
        "",
    ]

    for row in bottlenecks.to_dict(orient="records"):
        lines.append(
            f"- **{row['bottleneck']}** ({row['severity']}) : "
            f"{row['evidence']} ; recommandation : {row['recommendation']}"
        )

    lines.extend(["", "## Optimisations et justification", ""])
    for row in optimization_decisions.to_dict(orient="records"):
        lines.append(
            f"- **{row['decision']}** — {row['status']} : {row['rationale']} Risque : {row['risk']}"
        )

    lines.extend(
        [
            "",
            "## Configuration finale retenue",
            "",
            "- API FastAPI conteneurisée, modèle chargé une seule fois au démarrage.",
            "- PostgreSQL pour la traçabilité des appels, inputs et outputs.",
            "- Inférence CPU LightGBM conservée, sans GPU ni ONNX à ce stade.",
            "- Tests CI/CD inchangés : lint, tests unitaires, build Docker et smoke test API.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_performance_report(
    *,
    database_url: str,
    output_dir: str | Path,
    latency_warning_ms: float = 1000.0,
    error_rate_warning: float = 0.05,
) -> PerformanceReport:
    """Génère un rapport de performance depuis les logs de monitoring."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    api_calls = load_api_call_logs(database_url)
    prediction_logs = load_prediction_logs(database_url)
    production_outputs = load_production_outputs(database_url)
    if production_outputs.empty:
        production_outputs = production_prediction_frame(prediction_logs)

    api_summary = compute_api_call_summary(api_calls)
    status_code_summary = compute_status_code_summary(api_calls)
    latency_by_path = compute_latency_by_path(api_calls)
    api_latency_summary = summarize_latency(api_calls, source="api_calls")
    model_latency_summary = summarize_latency(production_outputs, source="model_inference")
    optimization_decisions = build_optimization_decisions()
    bottlenecks = identify_bottlenecks(
        api_summary=api_summary,
        model_latency_summary=model_latency_summary,
        latency_warning_ms=latency_warning_ms,
        error_rate_warning=error_rate_warning,
    )

    workbook_path = output / "performance_summary.xlsx"
    _write_workbook(
        workbook_path,
        {
            "api_summary": api_summary,
            "api_latency_summary": api_latency_summary,
            "model_latency_summary": model_latency_summary,
            "latency_by_path": latency_by_path,
            "status_code_summary": status_code_summary,
            "bottlenecks": bottlenecks,
            "optimization_decisions": optimization_decisions,
            "api_calls_sample": api_calls.head(100),
            "model_outputs_sample": production_outputs.head(100),
        },
    )

    markdown_path = output / "performance_report.md"
    _write_markdown_report(
        markdown_path,
        api_summary=api_summary,
        model_latency_summary=model_latency_summary,
        bottlenecks=bottlenecks,
        optimization_decisions=optimization_decisions,
    )

    return PerformanceReport(
        output_dir=output,
        workbook_path=workbook_path,
        markdown_path=markdown_path,
    )

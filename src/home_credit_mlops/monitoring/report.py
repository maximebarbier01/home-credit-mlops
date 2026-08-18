"""Génération du rapport de monitoring production."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from home_credit_mlops.data.io import read_table
from home_credit_mlops.monitoring.drift import (
    DriftConfig,
    compute_categorical_drift,
    compute_numeric_drift,
    summarize_drift,
)
from home_credit_mlops.monitoring.operational import (
    compute_api_call_summary,
    compute_latency_by_path,
    compute_prediction_summary,
    compute_status_code_summary,
    detect_operational_alerts,
)
from home_credit_mlops.monitoring.production import (
    load_api_call_logs,
    load_prediction_logs,
    load_production_inputs,
    load_production_outputs,
    production_feature_frame,
    production_prediction_frame,
)


@dataclass(frozen=True)
class MonitoringReport:
    """Chemins des livrables générés."""

    output_dir: Path
    workbook_path: Path
    html_path: Path
    plots: list[Path]


def default_monitoring_output_dir(reports_dir: str | Path) -> Path:
    """Construit un dossier daté pour les livrables de monitoring."""

    now = datetime.now()
    date_prefix = now.strftime("%Y%m%d")
    run_prefix = now.strftime("%Y%m%d_%H%M%S")
    return Path(reports_dir) / f"{date_prefix}_home_credit_monitoring" / f"{run_prefix}_monitoring"


def _drop_non_feature_columns(
    frame: pd.DataFrame,
    *,
    target_column: str,
    id_column: str,
) -> pd.DataFrame:
    columns_to_drop = [column for column in (target_column, id_column) if column in frame.columns]
    return frame.drop(columns=columns_to_drop)


def _write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            safe_name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.book[safe_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions


def _save_score_distribution(predictions: pd.DataFrame, path: Path) -> Path | None:
    if predictions.empty or "default_probability" not in predictions:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    predictions["default_probability"].astype(float).plot(
        kind="hist", bins=30, ax=ax, color="#2f6f9f"
    )
    ax.set_title("Distribution des probabilités de défaut")
    ax.set_xlabel("Probabilité de défaut")
    ax.set_ylabel("Nombre de clients")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _save_decision_distribution(predictions: pd.DataFrame, path: Path) -> Path | None:
    if predictions.empty or "credit_decision" not in predictions:
        return None

    counts = predictions["credit_decision"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    counts.plot(kind="bar", ax=ax, color=["#8f2d56", "#218380"])
    ax.set_title("Distribution des décisions crédit")
    ax.set_xlabel("Décision")
    ax.set_ylabel("Nombre de clients")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _save_latency_distribution(api_calls: pd.DataFrame, path: Path) -> Path | None:
    if api_calls.empty or "latency_ms" not in api_calls:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    api_calls["latency_ms"].astype(float).plot(kind="hist", bins=30, ax=ax, color="#f18f01")
    ax.set_title("Distribution des latences API")
    ax.set_xlabel("Latence (ms)")
    ax.set_ylabel("Nombre d'appels")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _save_top_drift_features(drift: pd.DataFrame, path: Path, *, top_n: int) -> Path | None:
    if drift.empty or "psi" not in drift:
        return None

    top = drift.sort_values("psi", ascending=False).head(top_n).sort_values("psi")
    if top.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.35)))
    ax.barh(top["feature"], top["psi"], color="#6a4c93")
    ax.axvline(0.10, color="#f18f01", linestyle="--", linewidth=1, label="PSI modéré")
    ax.axvline(0.25, color="#c1121f", linestyle="--", linewidth=1, label="PSI élevé")
    ax.set_title("Variables les plus dérivées")
    ax.set_xlabel("PSI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _write_html_report(
    path: Path,
    *,
    tables: dict[str, pd.DataFrame],
    plots: list[Path],
) -> None:
    relative_plots = [plot.name for plot in plots]
    html_sections = [
        "<html><head><meta charset='utf-8'><title>Home Credit Monitoring</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;}"
        "table{border-collapse:collapse;margin-bottom:28px;width:100%;font-size:13px;}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;}"
        "th{background:#f2f5f7;}img{max-width:100%;height:auto;margin:12px 0 28px;}"
        "h1,h2{color:#102a43;}</style></head><body>",
        "<h1>Rapport de monitoring - Home Credit Scoring API</h1>",
        "<p>PoC local : stockage SQLAlchemy des appels API, analyse opérationnelle et data drift.</p>",
    ]
    for plot_name in relative_plots:
        html_sections.append(f"<img src='{plot_name}' alt='{plot_name}'>")
    for name, frame in tables.items():
        html_sections.append(f"<h2>{name}</h2>")
        html_sections.append(frame.head(50).to_html(index=False, escape=False))
    html_sections.append("</body></html>")
    path.write_text("\n".join(html_sections), encoding="utf-8")


def build_monitoring_report(
    *,
    database_url: str,
    reference_data_path: str | Path,
    output_dir: str | Path,
    target_column: str,
    id_column: str,
    top_drift_features: int = 25,
    latency_warning_ms: float = 1000.0,
    error_rate_warning: float = 0.05,
    drift_config: DriftConfig | None = None,
) -> MonitoringReport:
    """Génère les livrables de monitoring production."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    api_calls = load_api_call_logs(database_url)
    prediction_logs = load_prediction_logs(database_url)
    production_features = load_production_inputs(database_url)
    predictions = load_production_outputs(database_url)
    if production_features.empty:
        production_features = production_feature_frame(prediction_logs)
    if predictions.empty:
        predictions = production_prediction_frame(prediction_logs)

    reference_features = _drop_non_feature_columns(
        read_table(reference_data_path),
        target_column=target_column,
        id_column=id_column,
    )
    shared_features = [
        column for column in reference_features.columns if column in production_features
    ]
    reference_aligned = reference_features[shared_features] if shared_features else pd.DataFrame()
    production_aligned = production_features[shared_features] if shared_features else pd.DataFrame()

    numeric_drift = compute_numeric_drift(
        reference_aligned,
        production_aligned,
        config=drift_config,
    )
    categorical_drift = compute_categorical_drift(
        reference_aligned,
        production_aligned,
        config=drift_config,
    )
    drift_summary = summarize_drift(numeric_drift, categorical_drift)

    api_summary = compute_api_call_summary(api_calls)
    prediction_summary = compute_prediction_summary(predictions)
    operational_alerts = detect_operational_alerts(
        api_summary,
        latency_warning_ms=latency_warning_ms,
        error_rate_warning=error_rate_warning,
    )
    status_code_summary = compute_status_code_summary(api_calls)
    latency_by_path = compute_latency_by_path(api_calls)
    drift_ranked = pd.concat([numeric_drift, categorical_drift], ignore_index=True)
    if "psi" in drift_ranked:
        drift_ranked = drift_ranked.sort_values(
            "psi",
            ascending=False,
            ignore_index=True,
        )

    tables = {
        "api_summary": api_summary,
        "prediction_summary": prediction_summary,
        "operational_alerts": operational_alerts,
        "status_code_summary": status_code_summary,
        "latency_by_path": latency_by_path,
        "drift_summary": drift_summary,
        "numeric_drift": numeric_drift,
        "categorical_drift": categorical_drift,
        "api_calls_sample": api_calls.head(100),
        "predictions_sample": predictions.head(100),
        "production_inputs_sample": production_features.head(100),
        "production_outputs_sample": predictions.head(100),
    }

    workbook_path = output / "monitoring_summary.xlsx"
    _write_workbook(workbook_path, tables)

    plots = [
        plot
        for plot in (
            _save_score_distribution(predictions, output / "score_distribution.png"),
            _save_decision_distribution(predictions, output / "decision_distribution.png"),
            _save_latency_distribution(api_calls, output / "latency_distribution.png"),
            _save_top_drift_features(
                drift_ranked,
                output / "top_drift_features.png",
                top_n=top_drift_features,
            ),
        )
        if plot is not None
    ]

    html_path = output / "monitoring_report.html"
    _write_html_report(html_path, tables=tables, plots=plots)

    return MonitoringReport(
        output_dir=output,
        workbook_path=workbook_path,
        html_path=html_path,
        plots=plots,
    )

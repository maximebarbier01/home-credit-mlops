"""Dashboard Streamlit de monitoring de l'API Home Credit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core.config import load_api_config
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
from home_credit_mlops.settings import load_settings


@st.cache_data(show_spinner=False)
def load_monitoring_tables(
    database_url: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_logs = load_prediction_logs(database_url)
    production_inputs = load_production_inputs(database_url)
    production_outputs = load_production_outputs(database_url)
    if production_inputs.empty:
        production_inputs = production_feature_frame(prediction_logs)
    if production_outputs.empty:
        production_outputs = production_prediction_frame(prediction_logs)
    return load_api_call_logs(database_url), prediction_logs, production_inputs, production_outputs


@st.cache_data(show_spinner=False)
def load_reference_features(
    reference_path: str, target_column: str, id_column: str
) -> pd.DataFrame:
    reference = read_table(reference_path)
    return reference.drop(
        columns=[column for column in (target_column, id_column) if column in reference.columns],
        errors="ignore",
    )


def _metric_value(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame:
        return default
    return float(frame.iloc[0][column])


def _render_operational_tab(api_calls: pd.DataFrame, predictions: pd.DataFrame) -> None:
    api_summary = compute_api_call_summary(api_calls)
    prediction_summary = compute_prediction_summary(predictions)
    status_summary = compute_status_code_summary(api_calls)
    latency_by_path = compute_latency_by_path(api_calls)
    alerts = detect_operational_alerts(
        api_summary,
        latency_warning_ms=1000.0,
        error_rate_warning=0.05,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Appels API", int(_metric_value(api_summary, "total_calls")))
    col2.metric("Taux d'erreur", f"{_metric_value(api_summary, 'error_rate'):.1%}")
    col3.metric("Latence p95", f"{_metric_value(api_summary, 'latency_p95_ms'):.0f} ms")
    col4.metric("Prédictions", int(_metric_value(prediction_summary, "prediction_count")))

    if not alerts.empty:
        st.warning("Alertes opérationnelles détectées")
        st.dataframe(alerts, width="stretch")
    else:
        st.success("Aucune alerte opérationnelle sur les seuils configurés.")

    st.subheader("Codes HTTP")
    if status_summary.empty:
        st.info("Aucun appel API journalisé.")
    else:
        st.bar_chart(status_summary.set_index("status_code")["count"])

    st.subheader("Latence par endpoint")
    st.dataframe(latency_by_path, width="stretch")


def _render_predictions_tab(predictions: pd.DataFrame) -> None:
    prediction_summary = compute_prediction_summary(predictions)
    col1, col2, col3 = st.columns(3)
    col1.metric("Taux de refus", f"{_metric_value(prediction_summary, 'refused_rate'):.1%}")
    col2.metric(
        "Score moyen", f"{_metric_value(prediction_summary, 'default_probability_mean'):.3f}"
    )
    col3.metric("Score p95", f"{_metric_value(prediction_summary, 'default_probability_p95'):.3f}")

    if predictions.empty:
        st.info("Aucune prédiction journalisée.")
        return

    st.subheader("Distribution des décisions")
    st.bar_chart(predictions["credit_decision"].value_counts())

    st.subheader("Distribution des probabilités de défaut")
    probability = pd.to_numeric(predictions["default_probability"], errors="coerce").dropna()
    histogram = pd.cut(probability, bins=20).value_counts().sort_index()
    histogram.index = histogram.index.astype(str)
    st.bar_chart(histogram)

    st.subheader("Dernières prédictions")
    st.dataframe(predictions.sort_values("created_at", ascending=False).head(50), width="stretch")


def _render_drift_tab(
    *,
    production_features: pd.DataFrame,
    reference_path: str,
    target_column: str,
    id_column: str,
    min_current_rows: int,
) -> None:
    if production_features.empty:
        st.info("Aucun input de production disponible pour calculer le drift.")
        return

    reference_features = load_reference_features(reference_path, target_column, id_column)
    shared_features = [
        column for column in reference_features.columns if column in production_features
    ]
    if not shared_features:
        st.warning("Aucune variable commune entre la référence et les inputs de production.")
        return

    drift_config = DriftConfig(min_current_rows=min_current_rows)
    numeric_drift = compute_numeric_drift(
        reference_features[shared_features],
        production_features[shared_features],
        config=drift_config,
    )
    categorical_drift = compute_categorical_drift(
        reference_features[shared_features],
        production_features[shared_features],
        config=drift_config,
    )
    drift = pd.concat([numeric_drift, categorical_drift], ignore_index=True)
    drift_summary = summarize_drift(numeric_drift, categorical_drift)

    col1, col2, col3 = st.columns(3)
    col1.metric("Variables analysées", int(_metric_value(drift_summary, "feature_count")))
    col2.metric("Drifts élevés", int(_metric_value(drift_summary, "high_drift_count")))
    col3.metric("PSI max", f"{_metric_value(drift_summary, 'max_psi'):.3f}")

    if drift.empty:
        st.info("Aucun drift calculable avec les données disponibles.")
        return

    top_drift = drift.sort_values("psi", ascending=False).head(20)
    st.subheader("Top variables dérivées")
    st.bar_chart(top_drift.set_index("feature")["psi"])
    st.dataframe(top_drift, width="stretch")


def main() -> None:
    settings = load_settings()
    api_config = load_api_config()

    st.set_page_config(
        page_title="Home Credit Monitoring",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Monitoring de l'API Home Credit")
    st.caption("PoC local : logs SQLAlchemy, métriques opérationnelles, scores et data drift.")

    with st.sidebar:
        st.header("Configuration")
        database_url = st.text_input("Base de logs SQLAlchemy", value=api_config.prediction_db_url)
        reference_path = st.text_input(
            "Dataset de référence",
            value=settings.dataset.default_train_path.as_posix(),
        )
        min_current_rows = st.number_input(
            "Minimum de lignes pour qualifier le drift",
            min_value=1,
            max_value=1000,
            value=30,
            step=1,
        )
        if st.button("Rafraîchir les données"):
            st.cache_data.clear()

    api_calls, prediction_logs, production_inputs, predictions = load_monitoring_tables(
        database_url
    )

    tab_ops, tab_predictions, tab_drift, tab_raw = st.tabs(
        ["Opérations", "Scores", "Data drift", "Logs bruts"]
    )

    with tab_ops:
        _render_operational_tab(api_calls, predictions)

    with tab_predictions:
        _render_predictions_tab(predictions)

    with tab_drift:
        _render_drift_tab(
            production_features=production_inputs,
            reference_path=reference_path,
            target_column=settings.dataset.target_column,
            id_column=settings.dataset.id_column,
            min_current_rows=int(min_current_rows),
        )

    with tab_raw:
        st.subheader("Appels API")
        st.dataframe(api_calls.tail(200), width="stretch")
        st.subheader("Prédictions brutes")
        st.dataframe(prediction_logs.tail(200), width="stretch")
        st.subheader("Inputs modèle")
        st.dataframe(production_inputs.tail(200), width="stretch")
        st.subheader("Outputs modèle")
        st.dataframe(predictions.tail(200), width="stretch")


if __name__ == "__main__":
    main()

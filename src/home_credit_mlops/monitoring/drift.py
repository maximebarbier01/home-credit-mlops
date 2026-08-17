"""Détection simple et explicable de data drift."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
except ImportError:  # pragma: no cover - scipy est normalement installé via scikit-learn
    ks_2samp = None


PSI_MODERATE_THRESHOLD = 0.10
PSI_HIGH_THRESHOLD = 0.25
KS_MODERATE_THRESHOLD = 0.10
KS_HIGH_THRESHOLD = 0.20
MISSING_RATE_DELTA_THRESHOLD = 0.10


@dataclass(frozen=True)
class DriftConfig:
    """Paramètres de calcul du drift."""

    bins: int = 10
    min_current_rows: int = 30
    max_categories: int = 20


def _drift_level(
    *,
    psi: float,
    ks_statistic: float | None = None,
    missing_rate_delta: float = 0.0,
    current_rows: int,
    min_current_rows: int,
) -> str:
    if current_rows < min_current_rows:
        return "insufficient_data"
    if (
        psi >= PSI_HIGH_THRESHOLD
        or (ks_statistic is not None and ks_statistic >= KS_HIGH_THRESHOLD)
        or abs(missing_rate_delta) >= MISSING_RATE_DELTA_THRESHOLD
    ):
        return "high"
    if psi >= PSI_MODERATE_THRESHOLD or (
        ks_statistic is not None and ks_statistic >= KS_MODERATE_THRESHOLD
    ):
        return "moderate"
    return "low"


def _population_stability_index(expected: np.ndarray, observed: np.ndarray) -> float:
    epsilon = 1e-6
    expected = np.clip(expected, epsilon, None)
    observed = np.clip(observed, epsilon, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def _numeric_edges(reference: pd.Series, bins: int) -> np.ndarray | None:
    values = pd.to_numeric(reference, errors="coerce").dropna()
    if values.nunique() < 2:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(values.to_numpy(dtype=float), quantiles))
    if len(edges) < 3:
        minimum = float(values.min())
        maximum = float(values.max())
        if math.isclose(minimum, maximum):
            return None
        edges = np.array([minimum, maximum])

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _numeric_distribution(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    counts, _ = np.histogram(numeric_values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(edges) - 1)
    return counts / total


def compute_numeric_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    config: DriftConfig | None = None,
) -> pd.DataFrame:
    """Compare les distributions numériques de référence et de production."""

    config = config or DriftConfig()
    shared_columns = [column for column in reference.columns if column in current.columns]
    numeric_columns = [
        column for column in shared_columns if pd.api.types.is_numeric_dtype(reference[column])
    ]
    columns = [
        "feature",
        "feature_type",
        "reference_missing_rate",
        "current_missing_rate",
        "missing_rate_delta",
        "reference_mean",
        "current_mean",
        "mean_delta",
        "psi",
        "ks_statistic",
        "ks_pvalue",
        "current_rows",
        "drift_level",
    ]
    rows: list[dict[str, object]] = []

    for column in numeric_columns:
        reference_values = pd.to_numeric(reference[column], errors="coerce")
        current_values = pd.to_numeric(current[column], errors="coerce")
        edges = _numeric_edges(reference_values, config.bins)
        if edges is None:
            continue

        expected = _numeric_distribution(reference_values, edges)
        observed = _numeric_distribution(current_values, edges)
        psi = _population_stability_index(expected, observed)

        reference_non_missing = reference_values.dropna()
        current_non_missing = current_values.dropna()
        ks_statistic: float | None = None
        ks_pvalue: float | None = None
        if (
            ks_2samp is not None
            and len(reference_non_missing) > 1
            and len(current_non_missing) > 1
        ):
            ks_result = ks_2samp(reference_non_missing, current_non_missing)
            ks_statistic = float(ks_result.statistic)
            ks_pvalue = float(ks_result.pvalue)

        reference_missing_rate = float(reference[column].isna().mean())
        current_missing_rate = float(current[column].isna().mean())
        missing_rate_delta = current_missing_rate - reference_missing_rate

        rows.append(
            {
                "feature": column,
                "feature_type": "numeric",
                "reference_missing_rate": reference_missing_rate,
                "current_missing_rate": current_missing_rate,
                "missing_rate_delta": missing_rate_delta,
                "reference_mean": float(reference_non_missing.mean())
                if not reference_non_missing.empty
                else np.nan,
                "current_mean": float(current_non_missing.mean())
                if not current_non_missing.empty
                else np.nan,
                "mean_delta": float(current_non_missing.mean() - reference_non_missing.mean())
                if not reference_non_missing.empty and not current_non_missing.empty
                else np.nan,
                "psi": psi,
                "ks_statistic": ks_statistic,
                "ks_pvalue": ks_pvalue,
                "current_rows": int(len(current)),
                "drift_level": _drift_level(
                    psi=psi,
                    ks_statistic=ks_statistic,
                    missing_rate_delta=missing_rate_delta,
                    current_rows=len(current),
                    min_current_rows=config.min_current_rows,
                ),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "psi",
        ascending=False,
        ignore_index=True,
    )


def _categorical_distribution(
    values: pd.Series,
    categories: list[object],
) -> np.ndarray:
    normalized = values.astype("object").where(values.notna(), "__MISSING__")
    normalized = normalized.where(normalized.isin(categories), "__OTHER__")
    counts = normalized.value_counts(dropna=False)
    buckets = categories + ["__OTHER__"]
    distribution = np.array([counts.get(bucket, 0) for bucket in buckets], dtype=float)
    total = distribution.sum()
    if total == 0:
        return np.zeros(len(buckets))
    return distribution / total


def compute_categorical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    config: DriftConfig | None = None,
) -> pd.DataFrame:
    """Compare les distributions catégorielles de référence et de production."""

    config = config or DriftConfig()
    shared_columns = [column for column in reference.columns if column in current.columns]
    categorical_columns = [
        column
        for column in shared_columns
        if not pd.api.types.is_numeric_dtype(reference[column])
        or pd.api.types.is_bool_dtype(reference[column])
    ]
    columns = [
        "feature",
        "feature_type",
        "reference_missing_rate",
        "current_missing_rate",
        "missing_rate_delta",
        "reference_unique_count",
        "current_unique_count",
        "psi",
        "current_rows",
        "drift_level",
    ]
    rows: list[dict[str, object]] = []

    for column in categorical_columns:
        reference_values = reference[column]
        current_values = current[column]
        top_categories = (
            reference_values.astype("object")
            .where(reference_values.notna(), "__MISSING__")
            .value_counts(dropna=False)
            .head(config.max_categories)
            .index.tolist()
        )
        if "__MISSING__" not in top_categories:
            top_categories.append("__MISSING__")

        expected = _categorical_distribution(reference_values, top_categories)
        observed = _categorical_distribution(current_values, top_categories)
        psi = _population_stability_index(expected, observed)
        reference_missing_rate = float(reference_values.isna().mean())
        current_missing_rate = float(current_values.isna().mean())
        missing_rate_delta = current_missing_rate - reference_missing_rate

        rows.append(
            {
                "feature": column,
                "feature_type": "categorical",
                "reference_missing_rate": reference_missing_rate,
                "current_missing_rate": current_missing_rate,
                "missing_rate_delta": missing_rate_delta,
                "reference_unique_count": int(reference_values.nunique(dropna=True)),
                "current_unique_count": int(current_values.nunique(dropna=True)),
                "psi": psi,
                "current_rows": int(len(current)),
                "drift_level": _drift_level(
                    psi=psi,
                    missing_rate_delta=missing_rate_delta,
                    current_rows=len(current),
                    min_current_rows=config.min_current_rows,
                ),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "psi",
        ascending=False,
        ignore_index=True,
    )


def summarize_drift(numeric_drift: pd.DataFrame, categorical_drift: pd.DataFrame) -> pd.DataFrame:
    """Synthétise le niveau de dérive calculé."""

    drift = pd.concat([numeric_drift, categorical_drift], ignore_index=True)
    if drift.empty:
        return pd.DataFrame(
            [
                {
                    "feature_count": 0,
                    "high_drift_count": 0,
                    "moderate_drift_count": 0,
                    "insufficient_data_count": 0,
                    "max_psi": 0.0,
                    "mean_psi": 0.0,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "feature_count": int(len(drift)),
                "high_drift_count": int((drift["drift_level"] == "high").sum()),
                "moderate_drift_count": int((drift["drift_level"] == "moderate").sum()),
                "insufficient_data_count": int(
                    (drift["drift_level"] == "insufficient_data").sum()
                ),
                "max_psi": float(drift["psi"].max()),
                "mean_psi": float(drift["psi"].mean()),
            }
        ]
    )

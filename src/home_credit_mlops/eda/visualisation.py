"""Mesures et graphiques d'association entre les variables et la target."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute a bias-corrected Cramer's V for two categorical variables."""
    data = pd.DataFrame({"x": x, "y": y}).dropna()

    if data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan

    confusion_matrix = pd.crosstab(data["x"], data["y"])
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n_obs = confusion_matrix.to_numpy().sum()
    if n_obs == 0:
        return np.nan

    phi2 = chi2 / n_obs
    n_rows, n_cols = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((n_cols - 1) * (n_rows - 1)) / (n_obs - 1))
    rows_corr = n_rows - ((n_rows - 1) ** 2) / (n_obs - 1)
    cols_corr = n_cols - ((n_cols - 1) ** 2) / (n_obs - 1)
    denominator = min(cols_corr - 1, rows_corr - 1)

    if denominator <= 0:
        return np.nan
    return float(np.sqrt(phi2_corr / denominator))


def compute_feature_target_associations(
    df: pd.DataFrame,
    target: str,
    cat_threshold: int = 40,
) -> pd.DataFrame:
    """Measure feature-target association for binary classification."""
    results: list[dict[str, object]] = []
    target_series = df[target]

    for column in df.columns:
        if column == target:
            continue

        feature = df[column]
        if pd.api.types.is_numeric_dtype(feature):
            data = pd.DataFrame({"x": feature, "y": target_series}).dropna()
            if data["x"].nunique() < 2 or data["y"].nunique() < 2:
                score = np.nan
            else:
                score, _ = pointbiserialr(data["y"], data["x"])

            results.append(
                {
                    "feature": column,
                    "type_feature": "numeric",
                    "method": "point_biserial",
                    "score": score,
                    "score_abs": abs(score) if pd.notna(score) else np.nan,
                }
            )
            continue

        if feature.nunique(dropna=True) > cat_threshold:
            continue

        score = cramers_v(feature.astype("object"), target_series.astype("object"))
        results.append(
            {
                "feature": column,
                "type_feature": "categorical",
                "method": "cramers_v",
                "score": score,
                "score_abs": score,
            }
        )

    return (
        pd.DataFrame(results)
        .dropna(subset=["score_abs"])
        .sort_values("score_abs", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_target_associations(
    associations: pd.DataFrame,
    top_n: int = 30,
    figsize: tuple[int, int] = (10, 7),
    num_color: str = "#FFADA6",
    cat_color: str = "#A7DEB7",
    save_path: str | Path | None = None,
) -> None:
    """Plot the strongest global associations with the target."""
    frame = associations.head(top_n).copy()
    if frame.empty:
        return

    frame["feature_label"] = (
        frame["feature"].str.replace("_", " ", regex=False).str.capitalize()
    )
    frame = frame.sort_values("score_abs", ascending=True)
    colors = [num_color if kind == "numeric" else cat_color for kind in frame["type_feature"]]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        frame["feature_label"],
        frame["score_abs"],
        color=colors,
        edgecolor="#444444",
        linewidth=0.8,
        height=0.7,
    )
    ax.set_title("Variables most associated with the target", fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("Association strength", fontsize=11)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_score = frame["score_abs"].max()
    ax.set_xlim(0, max_score * 1.15)
    for bar, value in zip(bars, frame["score_abs"], strict=False):
        x_coord = bar.get_width()
        y_coord = bar.get_y() + bar.get_height() / 2
        ax.text(
            x_coord + max_score * 0.02,
            y_coord,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#333333",
        )

    plt.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_feature_target_signed_associations(
    df: pd.DataFrame,
    target: str,
    positive_label: int = 1,
    cat_threshold: int = 40,
    min_modal_count: int = 30,
) -> pd.DataFrame:
    """Compute signed associations to explain what increases or decreases risk."""
    results: list[dict[str, object]] = []
    target_series = (df[target] == positive_label).astype(float)

    for column in df.columns:
        if column == target:
            continue

        feature = df[column]
        if pd.api.types.is_numeric_dtype(feature):
            data = pd.DataFrame({"x": feature, "y": target_series}).dropna()
            if data["x"].nunique() < 2 or data["y"].nunique() < 2:
                continue

            score, _ = pointbiserialr(data["y"], data["x"])
            results.append(
                {
                    "feature": column,
                    "modality": None,
                    "label": column,
                    "type_feature": "numeric",
                    "method": "point_biserial",
                    "score": score,
                    "score_abs": abs(score),
                    "n": len(data),
                    "target_rate_modality": np.nan,
                    "target_rate_rest": np.nan,
                }
            )
            continue

        if feature.nunique(dropna=True) > cat_threshold:
            continue

        feature_as_object = feature.astype("object").where(feature.notna(), "MISSING")
        for modality, count in feature_as_object.value_counts(dropna=False).items():
            if count < min_modal_count:
                continue

            dummy = (feature_as_object == modality).astype(int)
            data = pd.DataFrame({"dummy": dummy, "y": target_series}).dropna()
            if data["dummy"].nunique() < 2 or data["y"].nunique() < 2:
                continue

            score, _ = pointbiserialr(data["dummy"], data["y"])
            target_rate_modality = data.loc[data["dummy"] == 1, "y"].mean()
            target_rate_rest = data.loc[data["dummy"] == 0, "y"].mean()
            results.append(
                {
                    "feature": column,
                    "modality": modality,
                    "label": f"{column} = {modality}",
                    "type_feature": "categorical",
                    "method": "modality_vs_rest",
                    "score": score,
                    "score_abs": abs(score),
                    "n": int(count),
                    "target_rate_modality": target_rate_modality,
                    "target_rate_rest": target_rate_rest,
                }
            )

    return (
        pd.DataFrame(results)
        .dropna(subset=["score"])
        .sort_values("score_abs", ascending=False)
        .reset_index(drop=True)
    )


def plot_signed_feature_target_associations(
    associations: pd.DataFrame,
    top_n: int = 25,
    figsize: tuple[int, int] = (11, 8),
    pos_color: str = "#FFADA6",
    neg_color: str = "#A7DEB7",
    save_path: str | Path | None = None,
) -> None:
    """Plot the strongest positive and negative associations with risk."""
    frame = associations.head(top_n).copy()
    if frame.empty:
        return

    frame["label_clean"] = (
        frame["label"].astype(str).str.replace("_", " ", regex=False).str.capitalize()
    )
    frame = frame.sort_values("score", ascending=True)
    colors = [pos_color if value > 0 else neg_color for value in frame["score"]]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        frame["label_clean"],
        frame["score"],
        color=colors,
        edgecolor="#444444",
        linewidth=0.8,
        height=0.7,
    )
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_title(
        "Variables and modalities positively or negatively associated with risk",
        fontsize=15,
        weight="bold",
        pad=12,
    )
    ax.set_xlabel("Signed association with the target", fontsize=11)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_abs = frame["score"].abs().max()
    ax.set_xlim(-max_abs * 1.20, max_abs * 1.20)
    for bar, value in zip(bars, frame["score"], strict=False):
        x_coord = bar.get_width()
        y_coord = bar.get_y() + bar.get_height() / 2
        ax.text(
            x_coord + (max_abs * 0.03 if value >= 0 else -max_abs * 0.03),
            y_coord,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            color="#333333",
        )

    plt.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "cramers_v",
    "compute_feature_target_associations",
    "compute_feature_target_signed_associations",
    "plot_feature_target_associations",
    "plot_signed_feature_target_associations",
]

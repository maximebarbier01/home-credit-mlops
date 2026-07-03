from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr


# ============================================================
# Association catégorielle <-> target : V de Cramer
# ============================================================


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calcule le V de Cramer corrigé du biais.
    Adapté pour mesurer l'association entre deux variables catégorielles.
    """
    data = pd.DataFrame({"x": x, "y": y}).dropna()

    if data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan

    confusion_matrix = pd.crosstab(data["x"], data["y"])

    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()

    if n == 0:
        return np.nan

    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    # Correction du biais
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    denominator = min(k_corr - 1, r_corr - 1)

    if denominator <= 0:
        return np.nan

    return np.sqrt(phi2_corr / denominator)


def compute_feature_target_associations(
    df: pd.DataFrame,
    target: str,
    cat_threshold: int = 40,
) -> pd.DataFrame:
    """
    Calcule une mesure d'association entre chaque feature et la target.

    - Variables numériques : corrélation point-bisériale
    - Variables catégorielles : V de Cramer

    Hypothèse : target binaire, encodée en 0/1.
    """

    results = []

    y = df[target]

    for col in df.columns:
        if col == target:
            continue

        x = df[col]

        # Variable numérique
        if pd.api.types.is_numeric_dtype(x):
            data = pd.DataFrame({"x": x, "y": y}).dropna()

            if data["x"].nunique() < 2 or data["y"].nunique() < 2:
                score = np.nan
            else:
                score, _ = pointbiserialr(data["y"], data["x"])

            results.append(
                {
                    "feature": col,
                    "type_feature": "numérique",
                    "method": "point_biserial",
                    "score": score,
                    "score_abs": abs(score) if pd.notna(score) else np.nan,
                }
            )

        # Variable catégorielle
        else:
            if x.nunique(dropna=True) <= cat_threshold:
                score = cramers_v(x.astype("object"), y.astype("object"))

                results.append(
                    {
                        "feature": col,
                        "type_feature": "catégorielle",
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
    figsize: tuple = (10, 7),
    num_color: str = "#FFADA6",
    cat_color: str = "#A7DEB7",
    save_path: str | Path | None = None,
):
    """
    Affiche les variables les plus associées à la target.
    Sauvegarde le graphique si save_path est renseigné.
    """

    df_plot = associations.head(top_n).copy()

    df_plot["feature_label"] = (
        df_plot["feature"].str.replace("_", " ", regex=False).str.capitalize()
    )

    df_plot = df_plot.sort_values("score_abs", ascending=True)

    colors = [num_color if t == "numérique" else cat_color for t in df_plot["type_feature"]]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        df_plot["feature_label"],
        df_plot["score_abs"],
        color=colors,
        edgecolor="#444444",
        linewidth=0.8,
        height=0.7,
    )

    ax.set_title(
        "Variables les plus associées à la cible",
        fontsize=16,
        weight="bold",
        pad=12,
    )

    ax.set_xlabel("Force d'association avec la cible", fontsize=11)
    ax.set_ylabel("")

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_score = df_plot["score_abs"].max()
    ax.set_xlim(0, max_score * 1.15)

    for bar, val in zip(bars, df_plot["score_abs"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        ax.text(
            x + max_score * 0.02,
            y,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#333333",
        )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def compute_feature_target_signed_associations(
    df: pd.DataFrame,
    target: str,
    positive_label=1,
    cat_threshold: int = 40,
    min_modal_count: int = 30,
) -> pd.DataFrame:
    """
    Calcule une association signée avec la cible.

    - Variables numériques :
        corrélation point-bisériale signée

    - Variables catégorielles :
        transformation modalité vs reste, puis corrélation signée
        Exemple : SEGMENT = AGRI vs autres segments

    Interprétation :
        score > 0 : associé à une probabilité plus forte de target=1
        score < 0 : associé à une probabilité plus faible de target=1
    """

    results = []

    y = (df[target] == positive_label).astype(float)

    for col in df.columns:
        if col == target:
            continue

        x = df[col]

        # -----------------------------
        # Variables numériques
        # -----------------------------
        if pd.api.types.is_numeric_dtype(x):
            data = pd.DataFrame({"x": x, "y": y}).dropna()

            if data["x"].nunique() < 2 or data["y"].nunique() < 2:
                continue

            score, _ = pointbiserialr(data["y"], data["x"])

            results.append(
                {
                    "feature": col,
                    "modality": None,
                    "label": col,
                    "type_feature": "numérique",
                    "method": "point_biserial",
                    "score": score,
                    "score_abs": abs(score),
                    "n": len(data),
                    "target_rate_modality": np.nan,
                    "target_rate_rest": np.nan,
                }
            )

        # -----------------------------
        # Variables catégorielles
        # -----------------------------
        else:
            if x.nunique(dropna=True) > cat_threshold:
                continue

            x_cat = x.astype("object").where(x.notna(), "MANQUANT")

            for modality, count in x_cat.value_counts(dropna=False).items():
                if count < min_modal_count:
                    continue

                dummy = (x_cat == modality).astype(int)

                data = pd.DataFrame({"dummy": dummy, "y": y}).dropna()

                if data["dummy"].nunique() < 2 or data["y"].nunique() < 2:
                    continue

                score, _ = pointbiserialr(data["dummy"], data["y"])

                target_rate_modality = data.loc[data["dummy"] == 1, "y"].mean()
                target_rate_rest = data.loc[data["dummy"] == 0, "y"].mean()

                results.append(
                    {
                        "feature": col,
                        "modality": modality,
                        "label": f"{col} = {modality}",
                        "type_feature": "catégorielle",
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
    figsize: tuple = (11, 8),
    pos_color: str = "#FFADA6",
    neg_color: str = "#A7DEB7",
    save_path: str | Path | None = None,
):
    """
    Affiche les associations positives et négatives avec la cible.

    score > 0 : plus associé à target=1
    score < 0 : moins associé à target=1
    """

    df_plot = associations.head(top_n).copy()

    df_plot["label_clean"] = (
        df_plot["label"].astype(str).str.replace("_", " ", regex=False).str.capitalize()
    )

    df_plot = df_plot.sort_values("score", ascending=True)

    colors = [pos_color if val > 0 else neg_color for val in df_plot["score"]]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        df_plot["label_clean"],
        df_plot["score"],
        color=colors,
        edgecolor="#444444",
        linewidth=0.8,
        height=0.7,
    )

    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.2)

    ax.set_title(
        "Variables et modalités associées positivement ou négativement à la cible",
        fontsize=15,
        weight="bold",
        pad=12,
    )

    ax.set_xlabel("Association signée avec la cible", fontsize=11)
    ax.set_ylabel("")

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_abs = df_plot["score"].abs().max()
    ax.set_xlim(-max_abs * 1.20, max_abs * 1.20)

    for bar, val in zip(bars, df_plot["score"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        ax.text(
            x + (max_abs * 0.03 if val >= 0 else -max_abs * 0.03),
            y,
            f"{val:+.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=10,
            color="#333333",
        )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

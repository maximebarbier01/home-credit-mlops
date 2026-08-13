"""Metriques de biais/fairness par groupe sensible, au seuil metier deja retenu.

Perimetre volontairement restreint :
- seulement `selection_rate`, `recall`, `fpr` et `business_cost` par groupe,
  plus deux indicateurs de disparite (`disparate_impact_ratio`,
  `equal_opportunity_difference`). Pas de precision/F1/ROC AUC par groupe :
  ces metriques n'ont pas de lecture metier claire pour une decision binaire
  a seuil (accorde/refuse) et ajouteraient du bruit sans eclairer la
  decision.
- pas de croisement des attributs sensibles (ex. genre x tranche d'age) :
  les effectifs du holdout sont trop faibles par cellule pour produire des
  ratios stables. C'est une limite connue, pas un oubli.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from home_credit_mlops.modeling.metrics import business_cost

DISPARATE_IMPACT_FLAG_THRESHOLD = 0.8

# Tranches d'age fixes par decennie : contrairement a des bandes par
# quantile, les bornes ne changent pas d'une campagne a l'autre, ce qui
# permet de comparer la fairness dans le temps. Convention standard et
# lisible en scoring credit.
AGE_BAND_EDGES = [20, 30, 40, 50, 60, 100]
AGE_BAND_LABELS = ["20-29", "30-39", "40-49", "50-59", "60+"]


@dataclass(frozen=True)
class GroupFairnessResult:
    group: str
    n: int
    n_positive_target: int
    selection_rate: float
    recall: float
    fpr: float
    business_cost: float


def build_age_bands(age_years: pd.Series) -> pd.Series:
    """Decoupe l'age en tranches fixes par decennie.

    Les ages hors de [20, 100) (bornes observees dans ce projet) sont
    laisses a NaN plutot que forces dans une tranche extreme.
    """
    return pd.cut(
        age_years,
        bins=AGE_BAND_EDGES,
        labels=AGE_BAND_LABELS,
        right=False,
    )


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> dict[str, float | int]:
    """Calcule selection_rate, recall, fpr et business_cost pour un groupe."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    n_positive_target = int((y_true == 1).sum())
    n_negative_target = int((y_true == 0).sum())
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())

    return {
        "n": n,
        "n_positive_target": n_positive_target,
        "selection_rate": float((y_pred == 1).mean()) if n > 0 else float("nan"),
        "recall": float(true_positive / n_positive_target) if n_positive_target > 0 else float("nan"),
        "fpr": float(false_positive / n_negative_target) if n_negative_target > 0 else float("nan"),
        "business_cost": (
            business_cost(y_true, y_pred, fn_cost=fn_cost, fp_cost=fp_cost, normalize=True)
            if n > 0
            else float("nan")
        ),
    }


def compute_fairness_table(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_column: str = "TARGET",
    prediction_column: str = "prediction",
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> pd.DataFrame:
    """Une ligne par valeur de `group_column` (NaN exclus, sans imputation)."""
    valid = frame.dropna(subset=[group_column])

    rows: list[dict[str, object]] = []
    for group_value, group_frame in valid.groupby(group_column, observed=True):
        metrics = compute_group_metrics(
            group_frame[target_column].to_numpy(),
            group_frame[prediction_column].to_numpy(),
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        rows.append(asdict(GroupFairnessResult(group=str(group_value), **metrics)))

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def compute_disparity_summary(fairness_table: pd.DataFrame) -> dict[str, object]:
    """Ratio de disparate impact (regle des 4/5e) et ecart d'equal opportunity.

    `disparate_impact_ratio` = min/max des selection_rate entre groupes,
    flagge si < 0.8 (seuil standard, pas invente pour ce projet).
    `equal_opportunity_difference` = ecart max des recall entre groupes,
    rapporte sans seuil de flag (aucune convention standard ne fixe de
    seuil pour cet ecart).
    """
    if fairness_table.empty or len(fairness_table) < 2:
        return {
            "disparate_impact_ratio": float("nan"),
            "equal_opportunity_difference": float("nan"),
            "flag_disparate_impact": False,
        }

    selection_rates = fairness_table["selection_rate"].dropna()
    recalls = fairness_table["recall"].dropna()

    disparate_impact_ratio = (
        float(selection_rates.min() / selection_rates.max())
        if len(selection_rates) >= 2 and selection_rates.max() > 0
        else float("nan")
    )
    equal_opportunity_difference = (
        float(recalls.max() - recalls.min()) if len(recalls) >= 2 else float("nan")
    )
    flag_disparate_impact = bool(
        not np.isnan(disparate_impact_ratio)
        and disparate_impact_ratio < DISPARATE_IMPACT_FLAG_THRESHOLD
    )

    return {
        "disparate_impact_ratio": disparate_impact_ratio,
        "equal_opportunity_difference": equal_opportunity_difference,
        "flag_disparate_impact": flag_disparate_impact,
    }

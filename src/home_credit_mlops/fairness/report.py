"""Export des rapports de fairness : CSV, graphiques et classeur Excel.

Suit le meme decoupage compute/export que `modeling/interpretability.py` :
`fairness/metrics.py` calcule, ce module se contente d'ecrire les artefacts
sur disque (style graphique identique au reste du projet : `#4C78A8`,
dpi=150).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from home_credit_mlops.fairness.metrics import (
    AGE_BAND_EDGES,
    build_age_bands,
    compute_disparity_summary,
    compute_fairness_table,
)
from home_credit_mlops.reporting.excel import build_workbook_from_directory

LOGGER = logging.getLogger(__name__)

BAR_COLOR = "#4C78A8"
METRIC_LABELS = {
    "selection_rate": "Taux de selection (refus)",
    "recall": "Recall (detection des defauts)",
    "fpr": "Taux de faux positifs",
    "business_cost": "Cout metier",
}
KNOWN_LIMITATIONS = [
    "pas d'analyse croisee genre x tranche d'age : effectifs du holdout "
    "trop faibles par cellule pour des ratios stables",
    "pas de precision/F1/ROC AUC par groupe : hors perimetre pour une "
    "decision binaire a seuil (voir docstring fairness/metrics.py)",
]


def _plot_metric_bar(
    table: pd.DataFrame,
    *,
    metric: str,
    attribute_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=table, x="group", y=metric, color=BAR_COLOR)
    plt.title(f"{METRIC_LABELS.get(metric, metric)} par {attribute_label}")
    plt.xlabel(attribute_label)
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _export_attribute_report(
    frame: pd.DataFrame,
    *,
    group_column: str,
    attribute_slug: str,
    attribute_label: str,
    target_column: str,
    prediction_column: str,
    fn_cost: float,
    fp_cost: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    table = compute_fairness_table(
        frame,
        group_column=group_column,
        target_column=target_column,
        prediction_column=prediction_column,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    table.to_csv(output_dir / f"fairness_by_{attribute_slug}.csv", index=False)

    for metric in ("selection_rate", "recall", "fpr", "business_cost"):
        _plot_metric_bar(
            table,
            metric=metric,
            attribute_label=attribute_label,
            output_path=output_dir / f"fairness_{attribute_slug}_{metric}.png",
        )

    summary = compute_disparity_summary(table)
    return table, summary


def export_fairness_report(
    joined_frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_column: str = "TARGET",
    prediction_column: str = "prediction",
    gender_column: str = "CODE_GENDER",
    age_column: str = "AGE_YEARS",
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    source_campaign: str | None = None,
    model_name: str | None = None,
    threshold: float | None = None,
) -> dict[str, object]:
    """Calcule et exporte les rapports de fairness (genre + tranche d'age).

    `joined_frame` doit contenir `SK_ID_CURR`, `target_column`,
    `prediction_column`, `gender_column` et `age_column` (typiquement la
    jointure entre des predictions holdout et le dataset de features).
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    working_frame = joined_frame.copy()
    working_frame["AGE_BAND"] = build_age_bands(working_frame[age_column])

    gender_table, gender_summary = _export_attribute_report(
        working_frame,
        group_column=gender_column,
        attribute_slug="gender",
        attribute_label="genre",
        target_column=target_column,
        prediction_column=prediction_column,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        output_dir=destination,
    )
    age_table, age_summary = _export_attribute_report(
        working_frame,
        group_column="AGE_BAND",
        attribute_slug="age_band",
        attribute_label="tranche d'age",
        target_column=target_column,
        prediction_column=prediction_column,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        output_dir=destination,
    )

    summary_frame = pd.DataFrame(
        [
            {"attribute": "gender", **gender_summary},
            {"attribute": "age_band", **age_summary},
        ]
    )
    summary_frame.to_csv(destination / "fairness_summary.csv", index=False)

    metadata = {
        "source_campaign": source_campaign,
        "model_name": model_name,
        "threshold": threshold,
        "sensitive_attributes": [gender_column, "AGE_BAND"],
        "age_band_edges": AGE_BAND_EDGES,
        "gender": gender_summary,
        "age_band": age_summary,
        "known_limitations": KNOWN_LIMITATIONS,
    }
    (destination / "fairness_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    workbook_path = destination / "fairness.xlsx"
    build_workbook_from_directory(destination, workbook_path)

    return {
        "gender_table": gender_table,
        "age_table": age_table,
        "gender_summary": gender_summary,
        "age_summary": age_summary,
        "workbook_path": workbook_path.as_posix(),
    }

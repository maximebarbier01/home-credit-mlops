from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from home_credit_mlops.eda.report import generate_eda_report
from home_credit_mlops.eda.visualisation import (
    compute_feature_target_associations,
    compute_feature_target_signed_associations,
    plot_feature_target_associations,
    plot_signed_feature_target_associations,
)


def build_missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    missing_ratio = frame.isna().mean().sort_values(ascending=False)
    missing_ratio = missing_ratio[missing_ratio > 0]
    if missing_ratio.empty:
        return pd.DataFrame(
            columns=["column", "coverage_rate", "missing_ratio", "missing_count", "dtype"]
        )

    missing_count = frame.isna().sum().loc[missing_ratio.index]
    return pd.DataFrame(
        {
            "column": missing_ratio.index,
            "coverage_rate": 1 - missing_ratio.values,
            "missing_ratio": missing_ratio.values,
            "missing_count": missing_count.values,
            "dtype": [str(frame[column].dtype) for column in missing_ratio.index],
        }
    )


def _sample_for_eda(
    frame: pd.DataFrame,
    *,
    target_column: str | None,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    if sample_size <= 0 or len(frame) <= sample_size:
        return frame.copy()

    if target_column and target_column in frame.columns:
        sampled, _ = train_test_split(
            frame,
            train_size=sample_size,
            stratify=frame[target_column],
            random_state=random_state,
        )
        return sampled.copy()

    return frame.sample(n=sample_size, random_state=random_state).copy()


def _plot_missingness(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str,
    top_n: int,
    matrix_sample_size: int,
    random_state: int,
) -> None:
    missingness = build_missingness_report(frame)
    missingness.to_csv(output_dir / f"{stem}_missingness.csv", index=False)
    if missingness.empty:
        return

    top_columns = missingness.head(top_n)["column"].tolist()
    subset = frame[top_columns]
    sampled_subset = _sample_for_eda(
        subset,
        target_column=None,
        sample_size=min(matrix_sample_size, len(subset)),
        random_state=random_state,
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=missingness.head(top_n),
        x="missing_ratio",
        y="column",
        orient="h",
        color="#4C78A8",
    )
    plt.title(f"Top {min(top_n, len(missingness))} missing features")
    plt.xlabel("Missing ratio")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_missing_ratio.png", dpi=150)
    plt.close()

    msno.bar(subset, figsize=(12, 8), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_missingno_bar.png", dpi=150)
    plt.close()

    msno.matrix(sampled_subset, figsize=(14, 8), sparkline=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_missingno_matrix.png", dpi=150)
    plt.close()


def generate_home_credit_eda_artifacts(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_column: str,
    association_sample_size: int = 100_000,
    top_associations: int = 25,
    cat_threshold: int = 40,
    min_modal_count: int = 30,
    missing_top_n: int = 40,
    matrix_sample_size: int = 2_000,
    random_state: int = 42,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    generate_eda_report(dataframe, destination, target_column=target_column)
    _plot_missingness(
        dataframe,
        destination,
        stem="train_features",
        top_n=missing_top_n,
        matrix_sample_size=matrix_sample_size,
        random_state=random_state,
    )

    sampled = _sample_for_eda(
        dataframe,
        target_column=target_column,
        sample_size=association_sample_size,
        random_state=random_state,
    )

    associations = compute_feature_target_associations(
        sampled,
        target=target_column,
        cat_threshold=cat_threshold,
    )
    associations.to_csv(destination / "feature_target_associations.csv", index=False)
    if not associations.empty:
        plot_feature_target_associations(
            associations,
            top_n=top_associations,
            save_path=destination / "feature_target_associations.png",
        )
        plt.close("all")

    signed_associations = compute_feature_target_signed_associations(
        sampled,
        target=target_column,
        positive_label=1,
        cat_threshold=cat_threshold,
        min_modal_count=min_modal_count,
    )
    signed_associations.to_csv(
        destination / "feature_target_signed_associations.csv",
        index=False,
    )
    if not signed_associations.empty:
        plot_signed_feature_target_associations(
            signed_associations,
            top_n=top_associations,
            save_path=destination / "feature_target_signed_associations.png",
        )
        plt.close("all")

    metadata = {
        "rows": int(len(dataframe)),
        "columns": int(dataframe.shape[1]),
        "association_sample_rows": int(len(sampled)),
        "target_rate": float(dataframe[target_column].mean()) if target_column in dataframe else None,
    }
    (destination / "eda_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

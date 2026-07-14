"""Generation des diagnostics EDA et des rapports de qualite sur le dataset prepare."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

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


def _schema_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": dataframe.columns,
            "dtype": [str(dtype) for dtype in dataframe.dtypes],
            "non_null_count": dataframe.notna().sum().to_list(),
            "missing_count": dataframe.isna().sum().to_list(),
            "missing_ratio": (dataframe.isna().mean() * 100).round(2).to_list(),
            "nunique": dataframe.nunique(dropna=False).to_list(),
        }
    ).sort_values(["missing_ratio", "nunique"], ascending=[False, False])


def _numerical_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    numerical = dataframe.select_dtypes(include=["number"])
    if numerical.empty:
        return pd.DataFrame()
    return numerical.describe().transpose()


def _categorical_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    categorical = dataframe.select_dtypes(include=["object", "category", "bool"])
    if categorical.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "column": categorical.columns,
            "nunique": categorical.nunique(dropna=False).to_list(),
            "missing_ratio": (categorical.isna().mean() * 100).round(2).to_list(),
            "top_value": [
                categorical[column].mode(dropna=False).iloc[0]
                if not categorical[column].mode(dropna=False).empty
                else None
                for column in categorical.columns
            ],
        }
    ).sort_values("nunique", ascending=False)


def _target_correlations(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    numerical = dataframe.select_dtypes(include=["number"])
    if target_column not in numerical.columns:
        return pd.DataFrame()

    correlation = numerical.corr(numeric_only=True)[target_column].sort_values(ascending=False)
    return correlation.reset_index().rename(
        columns={"index": "feature", target_column: "correlation"}
    )


def write_data_quality_report(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_column: str | None = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _schema_summary(dataframe).to_csv(destination / "schema_summary.csv", index=False)

    numerical = _numerical_summary(dataframe)
    if not numerical.empty:
        numerical.to_csv(destination / "numerical_summary.csv")

    categorical = _categorical_summary(dataframe)
    if not categorical.empty:
        categorical.to_csv(destination / "categorical_summary.csv", index=False)

    if target_column and target_column in dataframe.columns:
        target_distribution = (
            dataframe[target_column]
            .value_counts(dropna=False)
            .rename_axis("target")
            .reset_index(name="count")
        )
        target_distribution["ratio"] = (
            target_distribution["count"] / target_distribution["count"].sum()
        )
        target_distribution.to_csv(destination / "target_distribution.csv", index=False)

        plt.figure(figsize=(6, 4))
        sns.countplot(data=dataframe, x=target_column)
        plt.title("Target distribution")
        plt.tight_layout()
        plt.savefig(destination / "target_distribution.png", dpi=150)
        plt.close()

        target_correlations = _target_correlations(dataframe, target_column)
        if not target_correlations.empty:
            target_correlations.to_csv(destination / "target_correlations.csv", index=False)


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

    write_data_quality_report(dataframe, destination, target_column=target_column)
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


__all__ = [
    "build_missingness_report",
    "generate_home_credit_eda_artifacts",
    "write_data_quality_report",
]

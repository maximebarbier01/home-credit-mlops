from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from home_credit_mlops.data.io import read_table
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.settings import load_settings


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
    return correlation.reset_index().rename(columns={"index": "feature", target_column: "correlation"})


def generate_eda_report(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_column: str | None = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    schema = _schema_summary(dataframe)
    schema.to_csv(destination / "schema_summary.csv", index=False)

    numerical = _numerical_summary(dataframe)
    if not numerical.empty:
        numerical.to_csv(destination / "numerical_summary.csv")

    categorical = _categorical_summary(dataframe)
    if not categorical.empty:
        categorical.to_csv(destination / "categorical_summary.csv", index=False)

    if target_column and target_column in dataframe.columns:
        target_distribution = dataframe[target_column].value_counts(dropna=False).rename_axis(
            "target"
        ).reset_index(name="count")
        target_distribution["ratio"] = target_distribution["count"] / target_distribution["count"].sum()
        target_distribution.to_csv(destination / "target_distribution.csv", index=False)

        plt.figure(figsize=(6, 4))
        sns.countplot(data=dataframe, x=target_column)
        plt.title("Target distribution")
        plt.tight_layout()
        plt.savefig(destination / "target_distribution.png", dpi=150)
        plt.close()

        correlations = _target_correlations(dataframe, target_column)
        if not correlations.empty:
            correlations.to_csv(destination / "target_correlations.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a lightweight EDA report.")
    parser.add_argument("--input", required=True, help="Input CSV or Parquet file.")
    parser.add_argument("--target", default=None, help="Target column, if available.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to reports/eda/<input_stem>.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = load_settings()

    input_path = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else settings.paths.reports_dir / "eda" / input_path.stem
    )

    dataframe = read_table(input_path)
    generate_eda_report(dataframe, output_dir, target_column=args.target)

    print(f"EDA report saved to: {output_dir}")

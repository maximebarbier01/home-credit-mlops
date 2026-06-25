from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from home_credit_mlops.data.io import read_table, write_table
from home_credit_mlops.logging_utils import configure_logging


def clean_dataset(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None = None,
    id_column: str | None = None,
    drop_constant_columns: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    cleaned = dataframe.drop_duplicates().copy()
    removed_columns: list[str] = []

    protected_columns = {column for column in [target_column, id_column] if column}

    if drop_constant_columns:
        constant_columns = [
            column
            for column in cleaned.columns
            if column not in protected_columns and cleaned[column].nunique(dropna=False) <= 1
        ]
        if constant_columns:
            cleaned = cleaned.drop(columns=constant_columns)
            removed_columns.extend(constant_columns)

    return cleaned, removed_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a cleaned dataset for training.")
    parser.add_argument("--input", required=True, help="Input CSV or Parquet file.")
    parser.add_argument("--output", required=True, help="Output CSV or Parquet file.")
    parser.add_argument("--target", default=None, help="Target column name.")
    parser.add_argument("--id-column", default=None, help="Identifier column name.")
    parser.add_argument(
        "--keep-constant-columns",
        action="store_true",
        help="Do not drop constant columns.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    dataframe = read_table(input_path)
    cleaned, removed_columns = clean_dataset(
        dataframe,
        target_column=args.target,
        id_column=args.id_column,
        drop_constant_columns=not args.keep_constant_columns,
    )
    write_table(cleaned, output_path)

    print(f"Input rows: {len(dataframe):,}")
    print(f"Output rows: {len(cleaned):,}")
    print(f"Input columns: {dataframe.shape[1]}")
    print(f"Output columns: {cleaned.shape[1]}")
    print(f"Removed constant columns: {len(removed_columns)}")
    print(f"Saved cleaned dataset to: {output_path}")

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)

    raise ValueError(f"Unsupported file format: {source}")


def write_table(dataframe: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    suffix = destination.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(destination, index=False)
        return
    if suffix == ".parquet":
        dataframe.to_parquet(destination, index=False)
        return

    raise ValueError(f"Unsupported file format: {destination}")

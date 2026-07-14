"""Preprocessing sklearn reutilisable pour tous les modeles du projet."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def split_features_target(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    drop_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    unique_drop_columns = list(dict.fromkeys(drop_columns or []))
    unique_drop_columns = [column for column in unique_drop_columns if column != target_column]

    missing_columns = [column for column in [target_column, *unique_drop_columns] if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    features = dataframe.drop(columns=[target_column, *unique_drop_columns])
    target = dataframe[target_column]
    return features, target


def build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Les variables numeriques sont seulement imputees ici : les modeles
    # arbres se chargent ensuite tres bien des echelles heterogenes.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    # Les variables categorielle passent par une imputation simple puis
    # un OneHotEncoder compatible avec les categories jamais vues.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    # Le ColumnTransformer garantit que ce preprocessing est applique
    # de facon identique en CV, en holdout et lors du refit final.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_columns, categorical_columns

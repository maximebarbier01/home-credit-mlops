from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

from home_credit_mlops.data.io import write_table
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.reporting.excel import (
    build_workbook_from_directory,
    remove_files_by_suffix,
)
from home_credit_mlops.settings import load_settings

DAYS_SENTINEL = 365243
RAW_TABLE_KEYS = {
    "application_train.csv": ["SK_ID_CURR", "TARGET"],
    "application_test.csv": ["SK_ID_CURR"],
    "bureau.csv": ["SK_ID_CURR", "SK_ID_BUREAU"],
    "bureau_balance.csv": ["SK_ID_BUREAU"],
    "previous_application.csv": ["SK_ID_CURR", "SK_ID_PREV"],
    "POS_CASH_balance.csv": ["SK_ID_CURR", "SK_ID_PREV"],
    "credit_card_balance.csv": ["SK_ID_CURR", "SK_ID_PREV"],
    "installments_payments.csv": ["SK_ID_CURR", "SK_ID_PREV"],
}
APPLICATION_TEST_COLUMNS = ["TARGET"]
PREVIOUS_DAY_COLUMNS = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]


def reduce_memory_usage(frame: pd.DataFrame) -> pd.DataFrame:
    optimized = frame.copy()
    for column in optimized.columns:
        if pd.api.types.is_integer_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], downcast="integer")
        elif pd.api.types.is_float_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], downcast="float")
    return optimized


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace({0: np.nan}))


def one_hot_encode(
    frame: pd.DataFrame,
    categorical_columns: Iterable[str] | None = None,
    *,
    dummy_na: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    categorical_columns = list(
        categorical_columns
        or frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    )
    if not categorical_columns:
        return frame, []

    original_columns = set(frame.columns)
    encoded = pd.get_dummies(
        frame,
        columns=categorical_columns,
        dummy_na=dummy_na,
        dtype=np.uint8,
    )
    new_columns = [column for column in encoded.columns if column not in original_columns]
    return reduce_memory_usage(encoded), new_columns


def flatten_groupby_columns(columns: pd.Index, prefix: str) -> list[str]:
    flattened: list[str] = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part).strip().upper() for part in column if part not in (None, "")]
            name = "_".join(parts)
        else:
            name = str(column).strip().upper()
        flattened.append(f"{prefix}_{name}")
    return flattened


def build_missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    missing_ratio = frame.isna().mean().sort_values(ascending=False)
    missing_ratio = missing_ratio[missing_ratio > 0]
    if missing_ratio.empty:
        return pd.DataFrame(columns=["column", "missing_ratio", "missing_count", "dtype"])

    missing_count = frame.isna().sum().loc[missing_ratio.index]
    return pd.DataFrame(
        {
            "column": missing_ratio.index,
            "missing_ratio": missing_ratio.values,
            "missing_count": missing_count.values,
            "dtype": [str(frame[column].dtype) for column in missing_ratio.index],
        }
    )


def collect_table_profile(frame: pd.DataFrame, table_name: str) -> dict[str, object]:
    profile: dict[str, object] = {
        "table_name": table_name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "total_missing_ratio": float(frame.isna().sum().sum() / frame.size),
        "numeric_columns": int(frame.select_dtypes(include=["number"]).shape[1]),
        "categorical_columns": int(
            frame.select_dtypes(include=["object", "category", "bool"]).shape[1]
        ),
    }
    for key_column in RAW_TABLE_KEYS.get(table_name, []):
        if key_column in frame.columns:
            profile[f"{key_column.lower()}_nunique"] = int(frame[key_column].nunique(dropna=False))
            profile[f"{key_column.lower()}_duplicate_rows"] = int(
                frame[key_column].duplicated().sum()
            )
    return profile


def plot_target_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=frame, x="TARGET")
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_missingness(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    top_n: int = 40,
    sample_rows: int = 2_000,
) -> None:
    missingness = build_missingness_report(frame)
    if missingness.empty:
        return

    top_columns = missingness.head(top_n)["column"].tolist()
    bar_subset = frame[top_columns]
    matrix_subset = bar_subset.sample(n=min(sample_rows, len(bar_subset)), random_state=42)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=missingness.head(top_n),
        x="missing_ratio",
        y="column",
        orient="h",
        color="#4C78A8",
    )
    plt.title(f"Top {min(top_n, len(missingness))} missing columns - {stem}")
    plt.xlabel("Missing ratio")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_missing_ratio.png", dpi=150)
    plt.close()

    msno.bar(bar_subset, figsize=(12, 8), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_missingno_bar.png", dpi=150)
    plt.close()

    msno.matrix(matrix_subset, figsize=(14, 8), sparkline=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_missingno_matrix.png", dpi=150)
    plt.close()


def drop_constant_columns(
    frame: pd.DataFrame,
    *,
    excluded_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    excluded_columns = set(excluded_columns or [])
    unique_counts = frame.nunique(dropna=False)
    constant_columns = [
        column
        for column, count in unique_counts.items()
        if count <= 1 and column not in excluded_columns
    ]
    if not constant_columns:
        return frame, []
    return frame.drop(columns=constant_columns), constant_columns


def clean_application_data(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()

    cleaned["DAYS_EMPLOYED_ANOM"] = (cleaned["DAYS_EMPLOYED"] == DAYS_SENTINEL).astype(np.int8)
    cleaned["DAYS_EMPLOYED"] = cleaned["DAYS_EMPLOYED"].replace(DAYS_SENTINEL, np.nan)
    cleaned.loc[cleaned["CODE_GENDER"] == "XNA", "CODE_GENDER"] = np.nan

    ext_source_columns = [
        column for column in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if column in cleaned
    ]
    document_columns = [column for column in cleaned.columns if column.startswith("FLAG_DOCUMENT_")]
    phone_flag_columns = [
        column
        for column in [
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        ]
        if column in cleaned
    ]
    address_mismatch_columns = [
        column
        for column in [
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY",
        ]
        if column in cleaned
    ]

    cleaned["AGE_YEARS"] = (-cleaned["DAYS_BIRTH"]).div(365.25)
    cleaned["EMPLOYED_YEARS"] = (-cleaned["DAYS_EMPLOYED"]).div(365.25)
    cleaned["EMPLOYED_TO_AGE_RATIO"] = safe_ratio(
        cleaned["DAYS_EMPLOYED"].abs(),
        cleaned["DAYS_BIRTH"].abs(),
    )
    cleaned["CREDIT_TO_INCOME_RATIO"] = safe_ratio(
        cleaned["AMT_CREDIT"],
        cleaned["AMT_INCOME_TOTAL"],
    )
    cleaned["ANNUITY_TO_INCOME_RATIO"] = safe_ratio(
        cleaned["AMT_ANNUITY"],
        cleaned["AMT_INCOME_TOTAL"],
    )
    cleaned["CREDIT_TO_ANNUITY_RATIO"] = safe_ratio(
        cleaned["AMT_CREDIT"],
        cleaned["AMT_ANNUITY"],
    )
    cleaned["GOODS_TO_CREDIT_RATIO"] = safe_ratio(
        cleaned["AMT_GOODS_PRICE"],
        cleaned["AMT_CREDIT"],
    )
    cleaned["INCOME_PER_PERSON"] = safe_ratio(
        cleaned["AMT_INCOME_TOTAL"],
        cleaned["CNT_FAM_MEMBERS"],
    )
    cleaned["ANNUITY_PER_PERSON"] = safe_ratio(
        cleaned["AMT_ANNUITY"],
        cleaned["CNT_FAM_MEMBERS"],
    )
    cleaned["CHILDREN_RATIO"] = safe_ratio(
        cleaned["CNT_CHILDREN"],
        cleaned["CNT_FAM_MEMBERS"],
    )

    if ext_source_columns:
        ext_sources = cleaned[ext_source_columns]
        cleaned["EXT_SOURCES_MEAN"] = ext_sources.mean(axis=1)
        cleaned["EXT_SOURCES_STD"] = ext_sources.std(axis=1)
        cleaned["EXT_SOURCES_MIN"] = ext_sources.min(axis=1)
        cleaned["EXT_SOURCES_MAX"] = ext_sources.max(axis=1)
        cleaned["EXT_SOURCES_NA_COUNT"] = ext_sources.isna().sum(axis=1)

    if document_columns:
        cleaned["DOCUMENT_COUNT"] = cleaned[document_columns].sum(axis=1)
    if phone_flag_columns:
        cleaned["PHONE_FLAG_COUNT"] = cleaned[phone_flag_columns].sum(axis=1)
    if address_mismatch_columns:
        cleaned["ADDRESS_MISMATCH_COUNT"] = cleaned[address_mismatch_columns].sum(axis=1)

    return reduce_memory_usage(cleaned)


def aggregate_bureau_features(raw_dir: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    bureau = reduce_memory_usage(pd.read_csv(raw_dir / "bureau.csv"))
    bureau_balance = reduce_memory_usage(pd.read_csv(raw_dir / "bureau_balance.csv"))
    profiles = [
        collect_table_profile(bureau, "bureau.csv"),
        collect_table_profile(bureau_balance, "bureau_balance.csv"),
    ]

    bureau_balance, balance_category_columns = one_hot_encode(
        bureau_balance,
        categorical_columns=["STATUS"],
    )
    bureau_balance_aggregations: dict[str, list[str]] = {
        "MONTHS_BALANCE": ["min", "max", "size"],
    }
    bureau_balance_aggregations.update({column: ["mean"] for column in balance_category_columns})
    bureau_balance_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(bureau_balance_aggregations)
    bureau_balance_agg.columns = flatten_groupby_columns(bureau_balance_agg.columns, "BB")
    bureau = bureau.merge(bureau_balance_agg, how="left", on="SK_ID_BUREAU")

    bureau, bureau_category_columns = one_hot_encode(bureau)
    bureau_aggregations: dict[str, list[str]] = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_ENDDATE_FACT": ["min", "max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum", "mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_LIMIT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "mean", "sum"],
        "DAYS_CREDIT_UPDATE": ["min", "max", "mean"],
        "AMT_ANNUITY": ["max", "mean"],
        "BB_MONTHS_BALANCE_MIN": ["min"],
        "BB_MONTHS_BALANCE_MAX": ["max"],
        "BB_MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }
    bureau_aggregations.update({column: ["mean"] for column in bureau_category_columns})

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(bureau_aggregations)
    bureau_agg.columns = flatten_groupby_columns(bureau_agg.columns, "BURO")
    bureau_loan_count = bureau.groupby("SK_ID_CURR").size().astype(np.int32)
    bureau_agg = bureau_agg.join(bureau_loan_count.rename("BURO_LOAN_COUNT"), how="left")

    if "CREDIT_ACTIVE_Active" in bureau.columns:
        active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
        if not active.empty:
            active_agg = active.groupby("SK_ID_CURR").agg(
                {
                    "AMT_CREDIT_SUM": ["sum", "mean"],
                    "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
                    "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean"],
                    "AMT_ANNUITY": ["mean", "max"],
                }
            )
            active_agg.columns = flatten_groupby_columns(active_agg.columns, "ACTIVE")
            bureau_agg = bureau_agg.join(active_agg, how="left")

    if "CREDIT_ACTIVE_Closed" in bureau.columns:
        closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
        if not closed.empty:
            closed_agg = closed.groupby("SK_ID_CURR").agg(
                {
                    "AMT_CREDIT_SUM": ["sum", "mean"],
                    "DAYS_ENDDATE_FACT": ["max", "mean"],
                }
            )
            closed_agg.columns = flatten_groupby_columns(closed_agg.columns, "CLOSED")
            bureau_agg = bureau_agg.join(closed_agg, how="left")

    result = reduce_memory_usage(bureau_agg.reset_index())
    del bureau, bureau_balance, bureau_balance_agg, bureau_agg
    gc.collect()
    return result, profiles


def aggregate_previous_application_features(
    raw_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    previous = reduce_memory_usage(pd.read_csv(raw_dir / "previous_application.csv"))
    profiles = [collect_table_profile(previous, "previous_application.csv")]

    for column in PREVIOUS_DAY_COLUMNS:
        if column in previous.columns:
            previous[column] = previous[column].replace(DAYS_SENTINEL, np.nan)

    previous["APP_CREDIT_RATIO"] = safe_ratio(
        previous["AMT_APPLICATION"],
        previous["AMT_CREDIT"],
    )
    previous["DOWN_PAYMENT_RATIO"] = safe_ratio(
        previous["AMT_DOWN_PAYMENT"],
        previous["AMT_CREDIT"],
    )
    previous["GOODS_CREDIT_RATIO"] = safe_ratio(
        previous["AMT_GOODS_PRICE"],
        previous["AMT_CREDIT"],
    )

    previous, previous_category_columns = one_hot_encode(previous)
    previous_aggregations: dict[str, list[str]] = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_RATIO": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["max", "mean", "sum"],
        "DAYS_FIRST_DUE": ["min", "max", "mean"],
        "DAYS_LAST_DUE_1ST_VERSION": ["min", "max", "mean"],
        "DAYS_LAST_DUE": ["min", "max", "mean"],
        "DAYS_TERMINATION": ["min", "max", "mean"],
    }
    previous_aggregations.update({column: ["mean"] for column in previous_category_columns})
    previous_agg = previous.groupby("SK_ID_CURR").agg(previous_aggregations)
    previous_agg.columns = flatten_groupby_columns(previous_agg.columns, "PREV")
    previous_count = previous.groupby("SK_ID_CURR").size().astype(np.int32)
    previous_agg = previous_agg.join(
        previous_count.rename("PREV_APPLICATION_COUNT"),
        how="left",
    )

    if "NAME_CONTRACT_STATUS_Approved" in previous.columns:
        approved = previous[previous["NAME_CONTRACT_STATUS_Approved"] == 1]
        if not approved.empty:
            approved_agg = approved.groupby("SK_ID_CURR").agg(
                {
                    "AMT_CREDIT": ["max", "mean"],
                    "AMT_ANNUITY": ["max", "mean"],
                    "CNT_PAYMENT": ["mean", "sum"],
                }
            )
            approved_agg.columns = flatten_groupby_columns(approved_agg.columns, "APPROVED")
            previous_agg = previous_agg.join(approved_agg, how="left")

    if "NAME_CONTRACT_STATUS_Refused" in previous.columns:
        refused = previous[previous["NAME_CONTRACT_STATUS_Refused"] == 1]
        if not refused.empty:
            refused_agg = refused.groupby("SK_ID_CURR").agg(
                {
                    "AMT_APPLICATION": ["max", "mean"],
                    "AMT_CREDIT": ["max", "mean"],
                }
            )
            refused_agg.columns = flatten_groupby_columns(refused_agg.columns, "REFUSED")
            previous_agg = previous_agg.join(refused_agg, how="left")

    result = reduce_memory_usage(previous_agg.reset_index())
    del previous, previous_agg
    gc.collect()
    return result, profiles


def aggregate_pos_cash_features(raw_dir: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pos_cash = reduce_memory_usage(pd.read_csv(raw_dir / "POS_CASH_balance.csv"))
    profiles = [collect_table_profile(pos_cash, "POS_CASH_balance.csv")]

    pos_cash, category_columns = one_hot_encode(pos_cash)
    aggregations: dict[str, list[str]] = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "CNT_INSTALMENT": ["mean", "sum"],
        "CNT_INSTALMENT_FUTURE": ["mean", "sum"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    aggregations.update({column: ["mean"] for column in category_columns})
    pos_cash_agg = pos_cash.groupby("SK_ID_CURR").agg(aggregations)
    pos_cash_agg.columns = flatten_groupby_columns(pos_cash_agg.columns, "POS")
    pos_cash_count = pos_cash.groupby("SK_ID_CURR").size().astype(np.int32)
    pos_cash_agg = pos_cash_agg.join(pos_cash_count.rename("POS_CASH_COUNT"), how="left")

    result = reduce_memory_usage(pos_cash_agg.reset_index())
    del pos_cash, pos_cash_agg
    gc.collect()
    return result, profiles


def aggregate_installments_features(
    raw_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    installments = reduce_memory_usage(pd.read_csv(raw_dir / "installments_payments.csv"))
    profiles = [collect_table_profile(installments, "installments_payments.csv")]

    installments["PAYMENT_RATIO"] = safe_ratio(
        installments["AMT_PAYMENT"],
        installments["AMT_INSTALMENT"],
    )
    installments["PAYMENT_DIFF"] = installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
    installments["DPD"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    ).clip(lower=0)
    installments["DBD"] = (
        installments["DAYS_INSTALMENT"] - installments["DAYS_ENTRY_PAYMENT"]
    ).clip(lower=0)

    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_RATIO": ["min", "max", "mean", "var"],
        "PAYMENT_DIFF": ["min", "max", "mean", "sum"],
        "AMT_INSTALMENT": ["min", "max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["min", "max", "mean"],
        "DAYS_INSTALMENT": ["min", "max", "mean"],
    }
    installments_agg = installments.groupby("SK_ID_CURR").agg(aggregations)
    installments_agg.columns = flatten_groupby_columns(installments_agg.columns, "INSTAL")
    installments_count = installments.groupby("SK_ID_CURR").size().astype(np.int32)
    installments_agg = installments_agg.join(
        installments_count.rename("INSTALLMENTS_COUNT"),
        how="left",
    )

    result = reduce_memory_usage(installments_agg.reset_index())
    del installments, installments_agg
    gc.collect()
    return result, profiles


def aggregate_credit_card_features(
    raw_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    credit_card = reduce_memory_usage(pd.read_csv(raw_dir / "credit_card_balance.csv"))
    profiles = [collect_table_profile(credit_card, "credit_card_balance.csv")]

    credit_card, category_columns = one_hot_encode(credit_card)
    aggregations: dict[str, list[str]] = {
        "MONTHS_BALANCE": ["min", "max", "mean", "size"],
        "AMT_BALANCE": ["min", "max", "mean", "sum"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["max", "mean"],
        "AMT_DRAWINGS_ATM_CURRENT": ["max", "mean", "sum"],
        "AMT_DRAWINGS_CURRENT": ["max", "mean", "sum"],
        "AMT_DRAWINGS_OTHER_CURRENT": ["max", "mean", "sum"],
        "AMT_DRAWINGS_POS_CURRENT": ["max", "mean", "sum"],
        "AMT_INST_MIN_REGULARITY": ["max", "mean"],
        "AMT_PAYMENT_CURRENT": ["max", "mean", "sum"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["max", "mean", "sum"],
        "AMT_RECEIVABLE_PRINCIPAL": ["max", "mean", "sum"],
        "AMT_RECIVABLE": ["max", "mean", "sum"],
        "AMT_TOTAL_RECEIVABLE": ["max", "mean", "sum"],
        "CNT_DRAWINGS_ATM_CURRENT": ["max", "mean", "sum"],
        "CNT_DRAWINGS_CURRENT": ["max", "mean", "sum"],
        "CNT_DRAWINGS_OTHER_CURRENT": ["max", "mean", "sum"],
        "CNT_DRAWINGS_POS_CURRENT": ["max", "mean", "sum"],
        "CNT_INSTALMENT_MATURE_CUM": ["max", "mean", "sum"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    aggregations.update({column: ["mean"] for column in category_columns})

    credit_card_agg = credit_card.groupby("SK_ID_CURR").agg(aggregations)
    credit_card_agg.columns = flatten_groupby_columns(credit_card_agg.columns, "CC")
    credit_card_counts = pd.DataFrame(
        {
            "CREDIT_CARD_LINE_COUNT": credit_card.groupby("SK_ID_CURR").size().astype(np.int32),
            "CREDIT_CARD_PREV_COUNT": credit_card.groupby("SK_ID_CURR")["SK_ID_PREV"]
            .nunique()
            .astype(np.int32),
        }
    )
    credit_card_agg = credit_card_agg.join(credit_card_counts, how="left")

    result = reduce_memory_usage(credit_card_agg.reset_index())
    del credit_card, credit_card_agg
    gc.collect()
    return result, profiles


def build_raw_overview(raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    profiles: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("*.csv")):
        read_kwargs = (
            {"encoding": "latin1"} if path.name == "HomeCredit_columns_description.csv" else {}
        )
        columns = pd.read_csv(path, nrows=0, **read_kwargs).columns.tolist()
        usecols = [column for column in RAW_TABLE_KEYS.get(path.name, []) if column in columns]
        if usecols:
            frame = pd.read_csv(path, usecols=usecols, **read_kwargs)
        else:
            frame = pd.read_csv(path, **read_kwargs)

        profile: dict[str, object] = {
            "table_name": path.name,
            "rows": int(len(frame)),
            "columns": int(len(columns)),
            "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "available_key_columns": ", ".join(usecols) if usecols else "",
            "profile_scope": "key_columns_only" if usecols else "full_table",
        }
        for key_column in usecols:
            profile[f"{key_column.lower()}_nunique"] = int(frame[key_column].nunique(dropna=False))
            profile[f"{key_column.lower()}_duplicate_rows"] = int(
                frame[key_column].duplicated().sum()
            )
        profiles.append(profile)
        del frame
        gc.collect()

    overview = pd.DataFrame(profiles).sort_values("table_name")
    overview.to_csv(output_dir / "raw_tables_overview.csv", index=False)
    return overview


def profile_raw_home_credit(raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    build_raw_overview(raw_dir, output_dir)

    application_train = pd.read_csv(raw_dir / "application_train.csv")
    target_distribution = (
        application_train["TARGET"]
        .value_counts(dropna=False)
        .rename_axis("target")
        .reset_index(name="count")
    )
    target_distribution["ratio"] = target_distribution["count"] / target_distribution["count"].sum()
    target_distribution.to_csv(
        output_dir / "application_train_target_distribution.csv", index=False
    )

    missingness = build_missingness_report(application_train)
    missingness.to_csv(output_dir / "application_train_missingness.csv", index=False)
    plot_target_distribution(
        application_train, output_dir / "application_train_target_distribution.png"
    )
    plot_missingness(application_train, output_dir, "application_train_raw")

    schema = pd.DataFrame(
        {
            "column": application_train.columns,
            "dtype": [str(dtype) for dtype in application_train.dtypes],
            "nunique": application_train.nunique(dropna=False).to_list(),
        }
    )
    schema.to_csv(output_dir / "application_train_schema.csv", index=False)


def merge_feature_frames(
    base_frame: pd.DataFrame,
    feature_frames: list[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = base_frame.copy()
    coverage_rows: list[dict[str, object]] = []

    for source_name, feature_frame in feature_frames:
        feature_columns = [column for column in feature_frame.columns if column != "SK_ID_CURR"]
        merged = merged.merge(feature_frame, on="SK_ID_CURR", how="left")
        source_coverage = merged[feature_columns].notna().any(axis=1)
        train_mask = merged["TARGET"].notna()
        test_mask = ~train_mask
        coverage_rows.append(
            {
                "source": source_name,
                "feature_columns": len(feature_columns),
                "matched_clients_all": int(source_coverage.sum()),
                "coverage_ratio_all": float(source_coverage.mean()),
                "matched_clients_train": int(source_coverage[train_mask].sum()),
                "coverage_ratio_train": float(source_coverage[train_mask].mean()),
                "matched_clients_test": int(source_coverage[test_mask].sum()),
                "coverage_ratio_test": float(source_coverage[test_mask].mean()),
            }
        )
    return merged, pd.DataFrame(coverage_rows)


def build_home_credit_dataset(
    raw_dir: Path,
    train_output_path: Path,
    test_output_path: Path,
    report_dir: Path,
) -> dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)

    application_train = pd.read_csv(raw_dir / "application_train.csv")
    application_test = pd.read_csv(raw_dir / "application_test.csv")
    application_test = application_test.assign(TARGET=np.nan)

    raw_profiles = [
        collect_table_profile(application_train, "application_train.csv"),
        collect_table_profile(application_test, "application_test.csv"),
    ]

    combined_application = pd.concat(
        [application_train, application_test],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    combined_application = clean_application_data(combined_application)
    raw_profiles.append(collect_table_profile(combined_application, "application_combined_cleaned"))

    feature_frames: list[tuple[str, pd.DataFrame]] = []
    aggregation_profiles: list[dict[str, object]] = []

    bureau_features, profiles = aggregate_bureau_features(raw_dir)
    feature_frames.append(("bureau_and_balance", bureau_features))
    aggregation_profiles.extend(profiles)

    previous_features, profiles = aggregate_previous_application_features(raw_dir)
    feature_frames.append(("previous_application", previous_features))
    aggregation_profiles.extend(profiles)

    pos_cash_features, profiles = aggregate_pos_cash_features(raw_dir)
    feature_frames.append(("pos_cash_balance", pos_cash_features))
    aggregation_profiles.extend(profiles)

    installments_features, profiles = aggregate_installments_features(raw_dir)
    feature_frames.append(("installments_payments", installments_features))
    aggregation_profiles.extend(profiles)

    credit_card_features, profiles = aggregate_credit_card_features(raw_dir)
    feature_frames.append(("credit_card_balance", credit_card_features))
    aggregation_profiles.extend(profiles)

    merged, merge_coverage = merge_feature_frames(combined_application, feature_frames)
    merged, constant_columns = drop_constant_columns(
        merged, excluded_columns=["TARGET", "SK_ID_CURR"]
    )
    merged = reduce_memory_usage(merged)

    train_frame = merged[merged["TARGET"].notna()].copy()
    train_frame["TARGET"] = train_frame["TARGET"].astype(np.int8)
    test_frame = merged[merged["TARGET"].isna()].drop(columns=["TARGET"]).copy()

    train_frame = train_frame.sort_values("SK_ID_CURR").reset_index(drop=True)
    test_frame = test_frame.sort_values("SK_ID_CURR").reset_index(drop=True)

    write_table(train_frame, train_output_path)
    write_table(test_frame, test_output_path)

    pd.DataFrame(raw_profiles + aggregation_profiles).to_csv(
        report_dir / "table_profiles.csv",
        index=False,
    )
    merge_coverage.to_csv(report_dir / "merge_coverage.csv", index=False)
    build_missingness_report(train_frame).to_csv(
        report_dir / "train_features_missingness.csv",
        index=False,
    )
    build_missingness_report(test_frame).to_csv(
        report_dir / "test_features_missingness.csv",
        index=False,
    )

    target_distribution = (
        train_frame["TARGET"]
        .value_counts(dropna=False)
        .rename_axis("target")
        .reset_index(name="count")
    )
    target_distribution["ratio"] = target_distribution["count"] / target_distribution["count"].sum()
    target_distribution.to_csv(report_dir / "train_target_distribution.csv", index=False)
    plot_target_distribution(train_frame, report_dir / "train_target_distribution.png")
    plot_missingness(train_frame, report_dir, "train_features")

    (report_dir / "constant_columns_removed.json").write_text(
        json.dumps(constant_columns, indent=2),
        encoding="utf-8",
    )

    report_workbook_path = report_dir / f"{report_dir.name}.xlsx"
    metadata = {
        "pipeline_steps": [
            "data_preparation",
            "variable_cleaning",
            "feature_engineering",
            "dataset_export",
        ],
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_columns": int(train_frame.shape[1]),
        "test_columns": int(test_frame.shape[1]),
        "target_rate": float(train_frame["TARGET"].mean()),
        "constant_columns_removed": constant_columns,
        "train_output_path": train_output_path.as_posix(),
        "test_output_path": test_output_path.as_posix(),
        "report_dir": report_dir.as_posix(),
        "report_workbook_path": report_workbook_path.as_posix(),
    }
    (report_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    build_workbook_from_directory(report_dir, report_workbook_path)
    remove_files_by_suffix(report_dir)

    return metadata


def parse_profile_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the raw Home Credit tables.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def parse_build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleaned and aggregated Home Credit feature dataset."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--test-output", default=None)
    parser.add_argument("--report-dir", default=None)
    return parser.parse_args()


def profile_main() -> None:
    configure_logging()
    args = parse_profile_args()
    settings = load_settings(args.config)

    raw_dir = Path(args.raw_dir) if args.raw_dir else settings.paths.raw_dir
    output_dir = (
        Path(args.output_dir) if args.output_dir else settings.paths.reports_dir / "home_credit_raw"
    )
    profile_raw_home_credit(raw_dir, output_dir)
    print(f"Raw profiling artifacts saved to: {output_dir}")


def build_main() -> None:
    configure_logging()
    args = parse_build_args()
    settings = load_settings(args.config)

    raw_dir = Path(args.raw_dir) if args.raw_dir else settings.paths.raw_dir
    train_output = (
        Path(args.train_output) if args.train_output else settings.dataset.default_train_path
    )
    test_output = (
        Path(args.test_output)
        if args.test_output
        else settings.paths.processed_dir / "test_features.parquet"
    )
    date_prefix = pd.Timestamp.now().strftime("%Y%m%d")
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else settings.paths.reports_dir / f"{date_prefix}_home_credit_eda"
    )

    metadata = build_home_credit_dataset(raw_dir, train_output, test_output, report_dir)
    print(json.dumps(metadata, indent=2))

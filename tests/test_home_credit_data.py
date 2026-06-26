import numpy as np
import pandas as pd

from home_credit_mlops.data.home_credit import (
    clean_application_data,
    flatten_groupby_columns,
    safe_ratio,
)


def test_safe_ratio_handles_zero_denominator() -> None:
    numerator = pd.Series([10.0, 4.0])
    denominator = pd.Series([2.0, 0.0])

    result = safe_ratio(numerator, denominator)

    assert result.iloc[0] == 5.0
    assert np.isnan(result.iloc[1])


def test_flatten_groupby_columns_adds_prefix() -> None:
    columns = pd.MultiIndex.from_tuples([
        ("amt_credit", "mean"),
        ("amt_credit", "max"),
    ])

    result = flatten_groupby_columns(columns, "PREV")

    assert result == ["PREV_AMT_CREDIT_MEAN", "PREV_AMT_CREDIT_MAX"]


def test_clean_application_data_replaces_anomalies_and_creates_features() -> None:
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            "DAYS_EMPLOYED": [365243],
            "DAYS_BIRTH": [-3650],
            "CODE_GENDER": ["XNA"],
            "AMT_CREDIT": [100000.0],
            "AMT_INCOME_TOTAL": [50000.0],
            "AMT_ANNUITY": [10000.0],
            "AMT_GOODS_PRICE": [90000.0],
            "CNT_FAM_MEMBERS": [2.0],
            "CNT_CHILDREN": [1],
            "EXT_SOURCE_1": [0.1],
            "EXT_SOURCE_2": [0.2],
            "EXT_SOURCE_3": [0.3],
            "FLAG_DOCUMENT_2": [1],
            "FLAG_DOCUMENT_3": [0],
            "FLAG_MOBIL": [1],
            "FLAG_EMP_PHONE": [1],
            "FLAG_WORK_PHONE": [0],
            "FLAG_CONT_MOBILE": [1],
            "FLAG_PHONE": [0],
            "FLAG_EMAIL": [1],
            "REG_REGION_NOT_LIVE_REGION": [0],
            "REG_REGION_NOT_WORK_REGION": [1],
            "LIVE_REGION_NOT_WORK_REGION": [0],
            "REG_CITY_NOT_LIVE_CITY": [1],
            "REG_CITY_NOT_WORK_CITY": [0],
            "LIVE_CITY_NOT_WORK_CITY": [1],
        }
    )

    result = clean_application_data(frame)

    assert np.isnan(result.loc[0, "DAYS_EMPLOYED"])
    assert result.loc[0, "DAYS_EMPLOYED_ANOM"] == 1
    assert pd.isna(result.loc[0, "CODE_GENDER"])
    assert result.loc[0, "CREDIT_TO_INCOME_RATIO"] == 2.0
    assert result.loc[0, "DOCUMENT_COUNT"] == 1
    assert result.loc[0, "PHONE_FLAG_COUNT"] == 4
    assert result.loc[0, "ADDRESS_MISMATCH_COUNT"] == 3

import numpy as np
import pandas as pd

from home_credit_mlops.data.home_credit import (
    clean_application_data,
    collect_table_profile,
    flatten_groupby_columns,
    load_raw_table,
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


def test_load_raw_table_drops_exact_duplicate_rows_but_keeps_repeated_keys(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_PREV": [10, 11, 20],
            "AMT_CREDIT": [100.0, 150.0, 200.0],
        }
    )
    # SK_ID_CURR=1 apparait deux fois pour deux credits differents
    # (repetition legitime, a conserver) ; on ajoute en plus une vraie
    # ligne dupliquee (doublon de saisie) a supprimer.
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(tmp_path / "toy.csv", index=False)

    loaded, n_duplicates = load_raw_table(tmp_path, "toy.csv")

    assert n_duplicates == 1
    assert len(loaded) == 3
    assert loaded["SK_ID_CURR"].tolist() == [1, 1, 2]


def test_load_raw_table_reports_zero_when_no_duplicates(tmp_path) -> None:
    frame = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "AMT_CREDIT": [10.0, 20.0, 30.0]})
    frame.to_csv(tmp_path / "toy.csv", index=False)

    loaded, n_duplicates = load_raw_table(tmp_path, "toy.csv")

    assert n_duplicates == 0
    assert len(loaded) == 3


def test_collect_table_profile_separates_full_duplicates_from_key_repetition() -> None:
    frame = pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "SK_ID_BUREAU": [100, 101, 102]})

    profile = collect_table_profile(frame, "bureau.csv", full_row_duplicates_removed=0)

    # SK_ID_CURR se repete legitimement (plusieurs credits par client) :
    # ce n'est pas un doublon de saisie.
    assert profile["sk_id_curr_key_repetitions"] == 1
    assert profile["sk_id_bureau_key_repetitions"] == 0
    assert profile["full_row_duplicates_removed"] == 0

from __future__ import annotations

import json

import pandas as pd

from home_credit_mlops.fairness.report import export_fairness_report


def _build_joined_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(1, 13),
            "CODE_GENDER": ["M", "F"] * 6,
            "AGE_YEARS": [22, 28, 35, 42, 48, 55, 33, 61, 45, 29, 38, 52],
            "TARGET": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "prediction": [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        }
    )


def test_export_fairness_report_writes_expected_files(tmp_path) -> None:
    joined_frame = _build_joined_frame()

    result = export_fairness_report(
        joined_frame,
        tmp_path,
        source_campaign="smoke_test",
        model_name="lightgbm__smote",
        threshold=0.22,
    )

    assert (tmp_path / "fairness_by_gender.csv").exists()
    assert (tmp_path / "fairness_by_age_band.csv").exists()
    assert (tmp_path / "fairness_summary.csv").exists()
    assert (tmp_path / "fairness_metadata.json").exists()
    assert (tmp_path / "fairness_gender_selection_rate.png").exists()
    assert (tmp_path / "fairness_age_band_recall.png").exists()
    assert (tmp_path / "fairness.xlsx").exists()

    assert result["gender_table"]["group"].tolist() == ["F", "M"]
    assert len(result["age_table"]) <= 5


def test_export_fairness_report_metadata_contains_summaries_and_limitations(tmp_path) -> None:
    joined_frame = _build_joined_frame()

    export_fairness_report(joined_frame, tmp_path)

    metadata = json.loads((tmp_path / "fairness_metadata.json").read_text(encoding="utf-8"))

    assert "gender" in metadata and "disparate_impact_ratio" in metadata["gender"]
    assert "age_band" in metadata and "equal_opportunity_difference" in metadata["age_band"]
    assert len(metadata["known_limitations"]) == 2


def test_export_fairness_report_gender_group_cardinality_matches_data(tmp_path) -> None:
    joined_frame = _build_joined_frame()

    result = export_fairness_report(joined_frame, tmp_path)

    gender_table = result["gender_table"]
    assert gender_table.loc[gender_table["group"] == "M", "n"].iloc[0] == 6
    assert gender_table.loc[gender_table["group"] == "F", "n"].iloc[0] == 6

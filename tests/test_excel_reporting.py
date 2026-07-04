from __future__ import annotations

from pathlib import Path

import pandas as pd

from home_credit_mlops.reporting.excel import build_experiment_workbooks, build_workbook_from_directory


def test_build_workbook_from_directory_reads_nested_files(tmp_path: Path) -> None:
    diagnostics_dir = tmp_path / "diagnostics"
    model_dir = diagnostics_dir / "rf"
    model_dir.mkdir(parents=True)
    pd.DataFrame({"metric": ["roc_auc"], "value": [0.71]}).to_csv(
        model_dir / "metrics.csv",
        index=False,
    )

    workbook = build_workbook_from_directory(diagnostics_dir, diagnostics_dir / "diagnostics.xlsx")

    assert workbook == diagnostics_dir / "diagnostics.xlsx"
    manifest = pd.read_excel(workbook, sheet_name="manifest")
    nested_sheet = pd.read_excel(workbook, sheet_name="rf__metrics")

    assert manifest["relative_path"].tolist() == ["rf/metrics.csv"]
    assert nested_sheet.loc[0, "metric"] == "roc_auc"


def test_build_experiment_workbooks_creates_nested_directory_workbooks(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    model_dir = experiment_dir / "diagnostics" / "rf"
    model_dir.mkdir(parents=True)
    pd.DataFrame({"metric": ["roc_auc"], "value": [0.71]}).to_csv(
        model_dir / "metrics.csv",
        index=False,
    )

    workbooks = build_experiment_workbooks(experiment_dir)

    workbook_paths = {path.relative_to(experiment_dir).as_posix() for path in workbooks}
    assert "summary.xlsx" in workbook_paths
    assert "diagnostics/diagnostics.xlsx" in workbook_paths
    assert "diagnostics/rf/rf.xlsx" in workbook_paths

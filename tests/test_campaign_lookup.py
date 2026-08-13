from __future__ import annotations

import json
from pathlib import Path

import pytest

from home_credit_mlops.reporting.campaign_lookup import (
    find_latest_campaign_metadata,
    load_champion_from_campaign,
)


def _candidate_name(model_name: str, sampling_strategy: str) -> str:
    if sampling_strategy == "baseline":
        return model_name
    return f"{model_name}__{sampling_strategy}"


def _write_campaign_metadata(
    reports_root: Path,
    *,
    campaign_name: str,
    created_at: str,
    base_model_name: str = "lightgbm",
    sampling_strategy: str = "smote",
    threshold: float = 0.2203,
    best_params: dict | None = None,
) -> Path:
    run_dir = reports_root / f"{campaign_name}_experiments" / f"{created_at}_{campaign_name}"
    run_dir.mkdir(parents=True)
    metadata = {
        "campaign_name": campaign_name,
        "created_at": created_at,
        "best_model": {
            "model_name": _candidate_name(base_model_name, sampling_strategy),
            "base_model_name": base_model_name,
            "sampling_strategy": sampling_strategy,
            "threshold": threshold,
            "best_params": best_params or {"model__n_estimators": 500},
        },
    }
    metadata_path = run_dir / "campaign_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def test_find_latest_campaign_metadata_picks_most_recent_matching_campaign(tmp_path) -> None:
    _write_campaign_metadata(
        tmp_path, campaign_name="champion_run", created_at="2026-07-01T00:00:00", threshold=0.10
    )
    latest_path = _write_campaign_metadata(
        tmp_path, campaign_name="champion_run", created_at="2026-07-11T00:00:00", threshold=0.20
    )
    _write_campaign_metadata(
        tmp_path, campaign_name="other_campaign", created_at="2026-07-15T00:00:00", threshold=0.99
    )

    found = find_latest_campaign_metadata(tmp_path, "champion_run")

    assert found == latest_path


def test_find_latest_campaign_metadata_returns_none_when_no_match(tmp_path) -> None:
    found = find_latest_campaign_metadata(tmp_path, "unknown_campaign")

    assert found is None


def test_load_champion_from_campaign_extracts_expected_fields(tmp_path) -> None:
    metadata_path = _write_campaign_metadata(
        tmp_path,
        campaign_name="champion_run",
        created_at="2026-07-11T00:00:00",
        base_model_name="lightgbm",
        sampling_strategy="smote",
        threshold=0.220331353025222,
        best_params={
            "model__learning_rate": 0.03,
            "model__n_estimators": 500,
            "model__num_leaves": 63,
        },
    )

    champion = load_champion_from_campaign(metadata_path)

    assert champion["model_name"] == "lightgbm"
    assert champion["candidate_key"] == "lightgbm__smote"
    assert champion["sampling_strategy"] == "smote"
    assert champion["threshold"] == pytest.approx(0.220331353025222)
    assert champion["params"] == {
        "model__learning_rate": 0.03,
        "model__n_estimators": 500,
        "model__num_leaves": 63,
    }

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "register_champion_model.py"
SPEC = importlib.util.spec_from_file_location("register_champion_model", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
register_champion_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(register_champion_model)


def test_candidate_name_includes_sampling_suffix_only_when_needed() -> None:
    assert register_champion_model._candidate_name("lightgbm", "baseline") == "lightgbm"
    assert register_champion_model._candidate_name("lightgbm", "smote") == "lightgbm__smote"


def test_parse_param_overrides_converts_common_scalar_types() -> None:
    params = register_champion_model._parse_param_overrides(
        [
            "model__n_estimators=500",
            "model__learning_rate=0.03",
            "model__use_flag=true",
            "model__optional=none",
            "model__label=champion",
        ]
    )

    assert params == {
        "model__n_estimators": 500,
        "model__learning_rate": 0.03,
        "model__use_flag": True,
        "model__optional": None,
        "model__label": "champion",
    }


def test_parse_param_overrides_requires_key_value_format() -> None:
    with pytest.raises(ValueError, match="step__parameter=value"):
        register_champion_model._parse_param_overrides(["model__n_estimators"])


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
            "model_name": register_champion_model._candidate_name(
                base_model_name, sampling_strategy
            ),
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

    found = register_champion_model._find_latest_campaign_metadata(tmp_path, "champion_run")

    assert found == latest_path


def test_find_latest_campaign_metadata_returns_none_when_no_match(tmp_path) -> None:
    found = register_champion_model._find_latest_campaign_metadata(tmp_path, "unknown_campaign")

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

    champion = register_champion_model._load_champion_from_campaign(metadata_path)

    assert champion["model_name"] == "lightgbm"
    assert champion["sampling_strategy"] == "smote"
    assert champion["threshold"] == pytest.approx(0.220331353025222)
    assert champion["params"] == {
        "model__learning_rate": 0.03,
        "model__n_estimators": 500,
        "model__num_leaves": 63,
    }


def test_resolve_threshold_prefers_explicit_override_over_campaign() -> None:
    champion = {"model_name": "lightgbm", "sampling_strategy": "smote", "threshold": 0.11}

    resolved = register_champion_model._resolve_threshold(
        champion_from_campaign=champion,
        model_name="lightgbm",
        sampling_strategy="smote",
        business_threshold=0.5,
        parser=None,
    )

    assert resolved == 0.5


def test_resolve_threshold_ignores_campaign_when_model_does_not_match() -> None:
    champion = {"model_name": "lightgbm", "sampling_strategy": "smote", "threshold": 0.11}
    parser = register_champion_model._build_argument_parser()

    with pytest.raises(SystemExit):
        register_champion_model._resolve_threshold(
            champion_from_campaign=champion,
            model_name="xgboost",
            sampling_strategy="smote",
            business_threshold=None,
            parser=parser,
        )


def test_resolve_champion_params_merges_overrides_on_top_of_campaign() -> None:
    champion = {
        "model_name": "lightgbm",
        "sampling_strategy": "smote",
        "params": {"model__n_estimators": 500, "model__num_leaves": 63},
    }

    resolved = register_champion_model._resolve_champion_params(
        champion_from_campaign=champion,
        model_name="lightgbm",
        sampling_strategy="smote",
        raw_param_overrides=["model__num_leaves=127"],
    )

    assert resolved == {"model__n_estimators": 500, "model__num_leaves": 127}

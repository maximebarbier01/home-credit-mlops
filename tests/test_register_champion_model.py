from __future__ import annotations

import importlib.util
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

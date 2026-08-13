from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from home_credit_mlops.features.preprocessing import build_preprocessor
from home_credit_mlops.modeling.interpretability import (
    _build_shap_explainer,
    _build_transformed_feature_frame,
    compute_feature_importance,
    export_feature_importance,
)


def _tiny_features_and_target() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(42)
    features = pd.DataFrame(
        {
            "num_a": rng.normal(loc=1000.0, scale=50.0, size=40),
            "num_b": rng.normal(loc=0.0, scale=1.0, size=40),
            "cat": rng.choice(["x", "y", "z"], size=40),
        }
    )
    target = pd.Series(rng.randint(0, 2, size=40))
    return features, target


def _fit_mlp_pipeline() -> tuple[Pipeline, pd.DataFrame]:
    features, target = _tiny_features_and_target()
    preprocessor, _, _ = build_preprocessor(features)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", MLPClassifier(hidden_layer_sizes=(4,), max_iter=50, random_state=42)),
        ]
    )
    pipeline.fit(features, target)
    return pipeline, features


def test_compute_feature_importance_raises_for_mlp() -> None:
    pipeline, _ = _fit_mlp_pipeline()

    with pytest.raises(ValueError):
        compute_feature_importance(pipeline)


def test_export_feature_importance_degrades_gracefully_for_mlp(tmp_path) -> None:
    pipeline, _ = _fit_mlp_pipeline()

    result = export_feature_importance(pipeline, tmp_path, top_n=5)

    assert result == {}
    assert not (tmp_path / "feature_importance_transformed.csv").exists()


def test_transformed_feature_frame_applies_scaler_step() -> None:
    pipeline, features = _fit_mlp_pipeline()

    transformed_frame, _ = _build_transformed_feature_frame(pipeline, features)
    preprocessor_only = pipeline.named_steps["preprocessor"].transform(features)
    if hasattr(preprocessor_only, "toarray"):
        preprocessor_only = preprocessor_only.toarray()

    # Le scaler doit changer les valeurs par rapport a la sortie brute du
    # preprocessor, sinon l'explication SHAP porterait sur des donnees que
    # le modele ne voit jamais réellement.
    assert not np.allclose(transformed_frame.to_numpy(), preprocessor_only)


def test_shap_explainer_falls_back_for_model_without_native_structure() -> None:
    pipeline, features = _fit_mlp_pipeline()
    transformed_frame, _ = _build_transformed_feature_frame(pipeline, features)

    explainer = _build_shap_explainer(pipeline.named_steps["model"], transformed_frame)
    # On explique reellement quelques lignes : c'est ce qui plante si
    # max_evals est trop bas ou si "auto" choisit un algorithme incompatible.
    explanation = explainer(transformed_frame.head(3))

    assert explanation is not None
    assert len(explanation.values) == 3

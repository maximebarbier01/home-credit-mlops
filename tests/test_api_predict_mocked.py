"""Chemin de succes de /predict contre un modele factice (StubScoringModel)."""

from __future__ import annotations

import pandas as pd

from conftest import StubScoringModel, build_small_schema


def test_predict_returns_stub_model_response_shape(api_client, valid_payload: dict) -> None:
    response = api_client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {
        "default_probability": 0.42,
        "business_threshold": 0.22,
        "predicted_default": 1,
        "credit_decision": "refused",
    }


def test_predict_reflects_custom_stub_response(api_app_factory, valid_payload: dict) -> None:
    custom_response = pd.DataFrame(
        {
            "default_probability": [0.05],
            "business_threshold": [0.22],
            "predicted_default": [0],
            "credit_decision": ["approved"],
        }
    )
    stub = StubScoringModel(build_small_schema(), response_frame=custom_response)
    client = api_app_factory(stub)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["credit_decision"] == "approved"
    assert response.json()["predicted_default"] == 0

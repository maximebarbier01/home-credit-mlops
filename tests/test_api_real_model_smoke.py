"""Smoke test optionnel contre le vrai modele publie sur Hugging Face Hub.

Hors du job CI standard (necessite reseau + le depot HF reellement publie).
Lancer manuellement avec HF_TOKEN defini si le depot est prive, ou juste
apres avoir execute scripts/export_model_for_serving.py."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from home_credit_mlops.settings import load_settings

pytestmark = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN")
    and load_settings().serving.model_repo_id.startswith("REPLACE_WITH"),
    reason="requires a published Hugging Face Hub model repo (configs/default.toml [serving])",
)


def test_predict_against_real_published_model() -> None:
    app = create_app()

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

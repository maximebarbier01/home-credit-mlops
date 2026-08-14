from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok_before_lifespan(api_app) -> None:
    # Sans context manager, le lifespan (donc le chargement du modele) ne
    # s'execute pas : /health doit repondre quand meme, model_loaded=False.
    client = TestClient(api_app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_health_ok_after_lifespan(api_client) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

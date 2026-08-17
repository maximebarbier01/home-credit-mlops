from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from mlflow.types import ColSpec, DataType, Schema


class StubScoringModel:
    """Imite l'interface PyFuncModel.predict (+ .metadata) pour les tests API."""

    def __init__(self, schema: Schema, response_frame: pd.DataFrame | None = None) -> None:
        self.metadata = _StubMetadata(schema)
        self._response_frame = response_frame

    def predict(self, input_frame: pd.DataFrame) -> pd.DataFrame:
        if self._response_frame is not None:
            return self._response_frame.set_axis(input_frame.index)
        return pd.DataFrame(
            {
                "default_probability": [0.42],
                "business_threshold": [0.22],
                "predicted_default": [1],
                "credit_decision": ["refused"],
            },
            index=input_frame.index,
        )


class _StubMetadata:
    def __init__(self, schema: Schema) -> None:
        self._schema = schema

    def get_input_schema(self) -> Schema:
        return self._schema


def build_small_schema() -> Schema:
    """Schema reduit couvrant les 5 champs valides metier, pour des tests rapides."""
    return Schema(
        [
            ColSpec(DataType.float, "AGE_YEARS", required=True),
            ColSpec(DataType.double, "AMT_INCOME_TOTAL", required=True),
            ColSpec(DataType.float, "AMT_CREDIT", required=True),
            ColSpec(DataType.integer, "CNT_CHILDREN", required=True),
            ColSpec(DataType.float, "CNT_FAM_MEMBERS", required=True),
            ColSpec(DataType.string, "CODE_GENDER", required=False),
        ]
    )


@pytest.fixture
def stub_model() -> StubScoringModel:
    return StubScoringModel(build_small_schema())


@pytest.fixture
def valid_payload() -> dict:
    return {
        "AGE_YEARS": 35.0,
        "AMT_INCOME_TOTAL": 50000.0,
        "AMT_CREDIT": 200000.0,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3.0,
        "CODE_GENDER": "F",
    }


@pytest.fixture
def api_app(stub_model: StubScoringModel):
    """Construit une app FastAPI isolee avec un modele factice injecte (pas de reseau)."""
    from app.main import create_app

    return create_app(
        resolve_model=lambda serving: Path("unused"),
        load_model=lambda local_dir: stub_model,
        prediction_repository=None,
        api_call_repository=None,
        init_prediction_storage=None,
    )


@pytest.fixture
def api_client(api_app):
    from fastapi.testclient import TestClient

    with TestClient(api_app) as client:
        yield client


@pytest.fixture
def api_app_factory():
    """Retourne un callable(stub_model) -> TestClient, pour les tests qui
    veulent controler finement la reponse du modele factice."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    created_clients: list[TestClient] = []

    def _factory(stub_model: StubScoringModel) -> TestClient:
        app = create_app(
            resolve_model=lambda serving: Path("unused"),
            load_model=lambda local_dir: stub_model,
            prediction_repository=None,
            api_call_repository=None,
            init_prediction_storage=None,
        )
        client = TestClient(app)
        client.__enter__()
        created_clients.append(client)
        return client

    yield _factory

    for client in created_clients:
        client.__exit__(None, None, None)

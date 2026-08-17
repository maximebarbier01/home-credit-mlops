"""Test d'integration reel : vrai chargement MLflow + vrai schema dynamique,
mais avec un petit modele de fixture (pas le vrai modele de 132 Mo) pour
rester rapide et hors reseau."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from mlflow.models import infer_signature

from app.main import create_app
from home_credit_mlops.modeling.serving import CreditScoringModel


class TinyPipelineStub:
    """Pipeline factice a 3 features numeriques, predict_proba deterministe."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        score = (features["AMT_INCOME_TOTAL"] < 20000).astype(float)
        return np.column_stack([1.0 - score, score])


def test_predict_end_to_end_with_real_mlflow_model(tmp_path) -> None:
    example_features = pd.DataFrame(
        {
            "AGE_YEARS": [30.0],
            "AMT_INCOME_TOTAL": [50000.0],
            "AMT_CREDIT": [100000.0],
        }
    )
    scoring_model = CreditScoringModel(
        pipeline=TinyPipelineStub(),
        business_threshold=0.5,
    )
    example_output = scoring_model.predict(context=None, model_input=example_features)
    signature = infer_signature(example_features, example_output)

    model_dir = tmp_path / "tiny_model"
    import mlflow.pyfunc

    mlflow.pyfunc.save_model(
        path=model_dir.as_posix(),
        python_model=scoring_model,
        signature=signature,
    )

    app = create_app(
        resolve_model=lambda serving: model_dir,
        load_model=lambda local_dir: mlflow.pyfunc.load_model(local_dir.as_posix()),
        prediction_repository=None,
        api_call_repository=None,
        init_prediction_storage=None,
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.json() == {"status": "ok", "model_loaded": True}

        low_income_response = client.post(
            "/predict",
            json={"AGE_YEARS": 25.0, "AMT_INCOME_TOTAL": 10000.0, "AMT_CREDIT": 50000.0},
        )
        assert low_income_response.status_code == 200
        assert low_income_response.json()["credit_decision"] == "refused"

        high_income_response = client.post(
            "/predict",
            json={"AGE_YEARS": 40.0, "AMT_INCOME_TOTAL": 80000.0, "AMT_CREDIT": 50000.0},
        )
        assert high_income_response.status_code == 200
        assert high_income_response.json()["credit_decision"] == "approved"

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import ApiConfig
from app.db.database import SessionLocal, init_db
from app.db.models import ApiCallLog, PredictionLog
from app.db.repository import save_api_call_log, save_prediction_log
from app.main import create_app
from conftest import StubScoringModel


def test_api_logs_success_and_validation_error(
    tmp_path: Path,
    stub_model: StubScoringModel,
    valid_payload: dict,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'monitoring.db').as_posix()}"
    app = create_app(
        resolve_model=lambda serving: Path("unused"),
        load_model=lambda local_dir: stub_model,
        prediction_repository=save_prediction_log,
        api_call_repository=save_api_call_log,
        init_prediction_storage=init_db,
        api_config_loader=lambda: ApiConfig(
            api_key=None,
            prediction_db_url=database_url,
            prediction_logging_enabled=True,
            api_call_logging_enabled=True,
        ),
    )

    with TestClient(app) as client:
        valid_response = client.post("/predict", json=valid_payload)
        invalid_payload = dict(valid_payload)
        del invalid_payload["AMT_INCOME_TOTAL"]
        invalid_response = client.post("/predict", json=invalid_payload)

    assert valid_response.status_code == 200
    assert invalid_response.status_code == 422

    with SessionLocal() as session:
        api_logs = session.scalars(
            select(ApiCallLog).where(ApiCallLog.path == "/predict").order_by(ApiCallLog.id)
        ).all()
        prediction_logs = session.scalars(select(PredictionLog)).all()

    assert [log.status_code for log in api_logs] == [200, 422]
    assert api_logs[0].request_payload["AGE_YEARS"] == valid_payload["AGE_YEARS"]
    assert api_logs[1].error_type == "http_422"
    assert len(prediction_logs) == 1
    assert prediction_logs[0].response_payload["credit_decision"] == "refused"

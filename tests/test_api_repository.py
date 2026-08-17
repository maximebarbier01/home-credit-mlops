from __future__ import annotations

from sqlalchemy import select

from app.db.database import SessionLocal, init_db
from app.db.models import PredictionLog
from app.db.repository import save_prediction_log


def test_save_prediction_log_persists_request_response_and_latency(tmp_path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'predictions.db').as_posix()}"
    init_db(database_url)

    log_id = save_prediction_log(
        request_payload={"AGE_YEARS": 35.0},
        response_payload={
            "default_probability": 0.42,
            "business_threshold": 0.22,
            "predicted_default": 1,
            "credit_decision": "refused",
        },
        latency_ms=12.5,
    )

    with SessionLocal() as session:
        log = session.scalars(select(PredictionLog).where(PredictionLog.id == log_id)).one()

    assert log.request_payload == {"AGE_YEARS": 35.0}
    assert log.response_payload["predicted_default"] == 1
    assert log.default_probability == 0.42
    assert log.credit_decision == "refused"
    assert log.latency_ms == 12.5

"""Modeles SQLAlchemy pour les données de production simulées."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class PredictionLog(Base):
    """Trace une requete de scoring et sa reponse metier."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    request_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    response_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    default_probability: Mapped[float] = mapped_column(Float, nullable=False)
    credit_decision: Mapped[str] = mapped_column(String(20), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProductionInput(Base):
    """Snapshot des variables envoyees au modele pour une prediction reussie."""

    __tablename__ = "production_inputs"
    __table_args__ = (
        UniqueConstraint("prediction_log_id", name="uq_production_inputs_prediction_log_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction_log_id: Mapped[int] = mapped_column(
        ForeignKey("prediction_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    input_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProductionOutput(Base):
    """Snapshot de la reponse metier retournee par le modele."""

    __tablename__ = "production_outputs"
    __table_args__ = (
        UniqueConstraint("prediction_log_id", name="uq_production_outputs_prediction_log_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction_log_id: Mapped[int] = mapped_column(
        ForeignKey("prediction_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    output_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    default_probability: Mapped[float] = mapped_column(Float, nullable=False)
    business_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_default: Mapped[int] = mapped_column(Integer, nullable=False)
    credit_decision: Mapped[str] = mapped_column(String(20), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ApiCallLog(Base):
    """Trace technique de chaque appel HTTP reçu par l'API."""

    __tablename__ = "api_call_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    method: Mapped[str] = mapped_column(String(12), nullable=False)
    path: Mapped[str] = mapped_column(String(255), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    request_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    client_host: Mapped[str | None] = mapped_column(String(255), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

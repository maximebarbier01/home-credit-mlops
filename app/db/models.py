"""Modeles SQLAlchemy pour les donnees de production simulees."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, func
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

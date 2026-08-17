"""Connexion SQLAlchemy utilisee pour journaliser les predictions."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    """Base declarative commune aux modeles SQLAlchemy."""


SessionLocal = sessionmaker(autocommit=False, autoflush=False)
_engine: Engine | None = None


def _sqlite_path_from_url(database_url: str) -> Path | None:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return None
    return Path(database_url.removeprefix(prefix))


def configure_database(database_url: str) -> Engine:
    """Configure l'engine SQLAlchemy et prepare le dossier SQLite si besoin."""

    global _engine
    sqlite_path = _sqlite_path_from_url(database_url)
    connect_args = {}
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False}

    _engine = create_engine(database_url, connect_args=connect_args)
    SessionLocal.configure(bind=_engine)
    return _engine


def init_db(database_url: str) -> Engine:
    """Initialise les tables de monitoring si elles n'existent pas."""

    from app.db import models  # noqa: F401 - importe les tables avant create_all

    engine = configure_database(database_url)
    Base.metadata.create_all(bind=engine)
    return engine


def get_engine() -> Engine:
    """Retourne l'engine actif ou leve une erreur explicite."""

    if _engine is None:
        raise RuntimeError("Database has not been configured.")
    return _engine


"""Sécurite minimale de l'API."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import load_api_config


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Valide une cle API si HOME_CREDIT_API_KEY est definie.

    En local et dans les tests, l'absence de variable d'environnement laisse
    l'API ouverte. En deploiement, definir HOME_CREDIT_API_KEY active la
    protection sans modifier le code.
    """

    expected_api_key = load_api_config().api_key
    if not expected_api_key:
        return

    if x_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

"""Sécurite minimale de l'API."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request, status


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Valide une cle API si HOME_CREDIT_API_KEY est definie.

    Lit la config deja chargee une fois au demarrage (app.state.api_config,
    voir le lifespan dans app/main.py) plutot que d'appeler load_api_config()
    a chaque requete : respecte l'injection de dependance utilisee par les
    tests (api_config_loader) et evite de re-parser l'environnement a
    chaque appel.

    En local et dans les tests, l'absence de cle laisse l'API ouverte. En
    deploiement, definir HOME_CREDIT_API_KEY active la protection sans
    modifier le code.
    """

    expected_api_key = request.app.state.api_config.api_key
    if not expected_api_key:
        return

    if x_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

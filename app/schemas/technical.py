"""Schémas Pydantic des endpoints techniques (racine, santé).

Sans modèle Pydantic explicite, FastAPI décrit ces routes comme un objet
générique dans OpenAPI, et Swagger affiche un exemple placeholder
(`additionalProp1`) sans rapport avec la vraie réponse — trompeur en démo,
puisque `/` et `/health` sont les tout premiers endpoints montrés.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RootResponse(BaseModel):
    """Message d'accueil de l'API."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"message": "Home Credit Scoring API"}}
    )

    message: str


class HealthResponse(BaseModel):
    """État de santé de l'API et du modèle chargé."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "ok", "model_loaded": True}}
    )

    status: str
    model_loaded: bool

"""Routes HTTP de l'API de scoring."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Depends

from app.core.security import require_api_key
from app.schemas.prediction import PREDICTION_RESPONSE_EXAMPLE, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()


def build_predict_handler(request_model: type, prediction_service: PredictionService):
    """Construit le handler /predict avec un schema Pydantic dynamique.

    La classe `request_model` est creee au demarrage depuis la signature
    MLflow. Les annotations sont donc fixees explicitement pour que FastAPI
    voie le vrai type et genere une documentation OpenAPI correcte.
    """

    async def handler(payload):
        return prediction_service.predict(payload)

    handler.__annotations__ = {"payload": request_model, "return": PredictionResponse}
    return handler


def build_prediction_router(
    request_model: type,
    prediction_service: PredictionService,
) -> APIRouter:
    """Construit le router metier une fois le modele MLflow charge."""

    prediction_router = APIRouter(dependencies=[Depends(require_api_key)])
    prediction_router.add_api_route(
        "/predict",
        build_predict_handler(request_model, prediction_service),
        methods=["POST"],
        response_model=PredictionResponse,
        summary="Calculer le score de defaut d'un client",
        description=(
            "Recoit un client deja transforme en features Home Credit et retourne "
            "la probabilite de defaut, le seuil metier applique et la decision."
        ),
        responses={
            200: {
                "description": "Prediction de scoring calculee avec succes.",
                "content": {"application/json": {"example": PREDICTION_RESPONSE_EXAMPLE}},
            },
            401: {"description": "Cle API absente ou invalide."},
            422: {"description": "Payload invalide ou champ obligatoire manquant."},
            500: {"description": "Erreur interne lors du scoring."},
        },
    )
    return prediction_router


def register_prediction_routes(
    app: FastAPI,
    *,
    request_model: type,
    prediction_service: PredictionService,
) -> None:
    """Ajoute les routes metier a l'application apres chargement du modele."""

    if getattr(app.state, "prediction_routes_registered", False):
        return

    app.include_router(build_prediction_router(request_model, prediction_service))
    app.state.prediction_routes_registered = True


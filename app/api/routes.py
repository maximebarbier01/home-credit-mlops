"""Routes HTTP de l'API de scoring."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, FastAPI, Depends, HTTPException, status

from app.core.security import require_api_key
from app.db.repository import get_monitoring_summary
from app.schemas.monitoring import MonitoringSummaryResponse
from app.schemas.prediction import PREDICTION_RESPONSE_EXAMPLE, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.get(
    "/monitoring/summary",
    response_model=MonitoringSummaryResponse,
    tags=["monitoring"],
    dependencies=[Depends(require_api_key)],
    summary="Consulter le résumé opérationnel de l'API",
    description=(
        "Retourne un résumé léger calculé depuis la base de logs : volume "
        "d'appels, taux d'erreur, latence, volume de prédictions et décisions."
    ),
    responses={
        200: {"description": "Résumé de monitoring calculé avec succès."},
        401: {"description": "Clé API absente ou invalide."},
        503: {"description": "Base de monitoring non disponible."},
    },
)
async def monitoring_summary() -> MonitoringSummaryResponse:
    try:
        return MonitoringSummaryResponse(**get_monitoring_summary())
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring database is not configured.",
        ) from exc


def build_predict_handler(request_model: type, prediction_service: PredictionService):
    """Construit le handler /predict avec un schema Pydantic dynamique.

    La classe `request_model` est creee au demarrage depuis la signature
    MLflow. Les annotations sont donc fixees explicitement pour que FastAPI
    voie le vrai type et genere une documentation OpenAPI correcte.
    """

    async def handler(payload, background_tasks: BackgroundTasks):
        result = prediction_service.score(payload)
        background_tasks.add_task(
            prediction_service.log_prediction,
            result.request_payload,
            result.response.model_dump(),
            result.latency_ms,
        )
        return result.response

    handler.__annotations__ = {
        "payload": request_model,
        "background_tasks": BackgroundTasks,
        "return": PredictionResponse,
    }
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

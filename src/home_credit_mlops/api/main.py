"""API FastAPI exposant le modele de scoring credit.

Le modele est charge une seule fois au demarrage (voir `lifespan`), jamais
par requete : c'est une exigence explicite de la consigne (temps de
reponse, memoire, scalabilite). La route /predict est enregistree apres le
chargement puisque son schema depend du modele reellement charge.

`create_app()` est une factory (pas un singleton module-level directement
instancie) afin que les tests puissent construire une application isolee
par test, avec un chargeur de modele injecte, sans faire fuiter des routes
enregistrees dynamiquement d'un test a l'autre.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from home_credit_mlops.api.model_loader import load_scoring_model, resolve_model_source
from home_credit_mlops.api.schemas import (
    PredictionResponse,
    build_request_model,
    business_rule_validators,
    coerce_frame_dtypes,
)
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.settings import ServingConfig, load_settings

configure_logging()
LOGGER = logging.getLogger(__name__)


def build_predict_handler(request_model: type, model: Any, input_schema: Any):
    """Construit le handler /predict, type dynamiquement sur request_model.

    `from __future__ import annotations` (actif dans ce module) transforme
    les annotations en chaines de caracteres a la definition de la fonction
    (PEP 563) : `payload: request_model` deviendrait la chaine litterale
    "request_model", que FastAPI ne pourrait pas resoudre vers la vraie
    classe dynamique (elle n'existe que dans cette fermeture, pas dans les
    globals du module). On fixe donc `__annotations__` explicitement avec
    les objets reels une fois la fonction definie, pour contourner ca.
    """

    async def handler(payload):
        input_frame = pd.DataFrame([payload.model_dump()])
        input_frame = coerce_frame_dtypes(input_frame, input_schema)
        result = model.predict(input_frame)
        return PredictionResponse(**result.iloc[0].to_dict())

    handler.__annotations__ = {"payload": request_model, "return": PredictionResponse}
    return handler


def create_app(
    *,
    resolve_model: Callable[[ServingConfig], Path] = resolve_model_source,
    load_model: Callable[[Path], Any] = load_scoring_model,
) -> FastAPI:
    """Construit une instance de l'API. `resolve_model`/`load_model` sont
    injectables pour permettre aux tests de fournir un modele factice sans
    reseau ni telechargement Hugging Face Hub."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = load_settings()
        local_dir = resolve_model(settings.serving)
        scoring_model = load_model(local_dir)
        input_schema = scoring_model.metadata.get_input_schema()
        request_model = build_request_model(input_schema, business_rule_validators())

        app.state.model = scoring_model
        app.state.request_model = request_model
        app.add_api_route(
            "/predict",
            build_predict_handler(request_model, scoring_model, input_schema),
            methods=["POST"],
            response_model=PredictionResponse,
            summary="Score a credit application",
            description=(
                "Recoit un client deja transforme en features (memes colonnes "
                "que le pipeline d'entrainement) et retourne la probabilite de "
                "defaut, le seuil metier applique et la decision de credit."
            ),
        )
        LOGGER.info("Scoring model loaded and /predict route registered.")

        yield

    app = FastAPI(
        title="Home Credit Scoring API",
        description="Expose le modele de scoring credit 'Pret a depenser'.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_request_timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        LOGGER.info(
            "%s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        del request
        LOGGER.exception("Unhandled error while processing request", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health")
    async def health() -> dict[str, Any]:
        model_loaded = getattr(app.state, "model", None) is not None
        return {"status": "ok", "model_loaded": model_loaded}

    return app


app = create_app()

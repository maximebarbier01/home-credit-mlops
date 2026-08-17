"""Point d'entree FastAPI de l'API Home Credit."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router, register_prediction_routes
from app.core.config import ApiConfig, load_api_config
from app.core.exceptions import ModelNotLoadedError
from app.db.database import init_db
from app.db.repository import save_prediction_log
from app.services.model_service import (
    ModelLoader,
    ModelResolver,
    ModelService,
    load_scoring_model,
    resolve_model_source,
)
from app.services.prediction_service import PredictionRepository, PredictionService
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.settings import Settings, load_settings

configure_logging()
LOGGER = logging.getLogger(__name__)

SettingsLoader = Callable[[], Settings]
ApiConfigLoader = Callable[[], ApiConfig]
DbInitializer = Callable[[str], Any]


def create_app(
    *,
    resolve_model: ModelResolver = resolve_model_source,
    load_model: ModelLoader = load_scoring_model,
    prediction_repository: PredictionRepository | None = save_prediction_log,
    init_prediction_storage: DbInitializer | None = init_db,
    settings_loader: SettingsLoader = load_settings,
    api_config_loader: ApiConfigLoader = load_api_config,
) -> FastAPI:
    """Construit une instance FastAPI injectable pour les tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = settings_loader()
        api_config = api_config_loader()

        repository = prediction_repository if api_config.prediction_logging_enabled else None
        if repository is not None and init_prediction_storage is not None:
            init_prediction_storage(api_config.prediction_db_url)

        model_service = ModelService(
            resolve_model=resolve_model,
            load_model=load_model,
        )
        model_service.load(settings.serving)
        if model_service.request_model is None:
            raise ModelNotLoadedError("Scoring model request schema is not available.")

        prediction_service = PredictionService(
            model_service,
            prediction_repository=repository,
        )

        app.state.model_service = model_service
        app.state.prediction_service = prediction_service
        app.state.request_model = model_service.request_model
        register_prediction_routes(
            app,
            request_model=model_service.request_model,
            prediction_service=prediction_service,
        )
        LOGGER.info("Scoring model loaded and /predict route registered.")

        yield

    app = FastAPI(
        title="Home Credit Scoring API",
        description="Expose le modele de scoring credit 'Pret a depenser'.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(router)

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

    @app.exception_handler(ModelNotLoadedError)
    async def handle_model_not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
        del request
        LOGGER.exception("Scoring model unavailable", exc_info=exc)
        return JSONResponse(status_code=503, content={"detail": "Scoring model unavailable."})

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        del request
        LOGGER.exception("Unhandled error while processing request", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/", tags=["technical"])
    async def root() -> dict[str, str]:
        return {"message": "Home Credit Scoring API"}

    @app.get("/health", tags=["technical"])
    async def health() -> dict[str, Any]:
        model_service = getattr(app.state, "model_service", None)
        model_loaded = bool(model_service and model_service.is_loaded)
        return {"status": "ok", "model_loaded": model_loaded}

    return app


app = create_app()

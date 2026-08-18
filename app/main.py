"""Point d'entrée FastAPI de l'API Home Credit."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router, register_prediction_routes
from app.core.config import ApiConfig, load_api_config
from app.core.exceptions import ModelNotLoadedError
from app.db.database import init_db
from app.db.repository import save_api_call_log, save_prediction_log
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

ApiCallRepository = Callable[..., int | None]
SettingsLoader = Callable[[], Settings]
ApiConfigLoader = Callable[[], ApiConfig]
DbInitializer = Callable[[str], Any]


def _json_payload_from_body(request: Request, body: bytes) -> dict[str, Any] | None:
    """Retourne le payload JSON si la requête en contient un exploitable."""

    content_type = request.headers.get("content-type", "")
    if not body or "application/json" not in content_type:
        return None

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    return payload if isinstance(payload, dict) else {"payload": payload}


def _restore_request_body(request: Request, body: bytes) -> None:
    """Replace le body lu par le middleware pour FastAPI/Pydantic."""

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive  # noqa: SLF001 - pattern Starlette courant pour middleware


def _http_reason_phrase(status_code: int) -> str:
    """Retourne un libellé standard pour un code HTTP."""

    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return f"HTTP {status_code}"


def create_app(
    *,
    resolve_model: ModelResolver = resolve_model_source,
    load_model: ModelLoader = load_scoring_model,
    prediction_repository: PredictionRepository | None = save_prediction_log,
    api_call_repository: ApiCallRepository | None = save_api_call_log,
    init_prediction_storage: DbInitializer | None = init_db,
    settings_loader: SettingsLoader = load_settings,
    api_config_loader: ApiConfigLoader = load_api_config,
) -> FastAPI:
    """Construit une instance FastAPI injectable pour les tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = settings_loader()
        api_config = api_config_loader()

        app.state.api_config = api_config
        app.state.api_call_repository = (
            api_call_repository if api_config.api_call_logging_enabled else None
        )

        repository = prediction_repository if api_config.prediction_logging_enabled else None
        if repository is not None and init_prediction_storage is not None:
            init_prediction_storage(api_config.prediction_db_url)
        elif app.state.api_call_repository is not None and init_prediction_storage is not None:
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
        body = await request.body()
        _restore_request_body(request, body)
        request_payload = _json_payload_from_body(request, body)

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

        repository = getattr(request.app.state, "api_call_repository", None)
        if repository is not None:
            error_type = f"http_{response.status_code}" if response.status_code >= 400 else None
            try:
                repository(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    latency_ms=duration_ms,
                    request_payload=request_payload,
                    error_type=error_type,
                    error_message=_http_reason_phrase(response.status_code) if error_type else None,
                    client_host=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                )
            except Exception:
                LOGGER.exception("API call logging failed; returning HTTP response anyway.")

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

"""Chargement du modele MLflow et inference unitaire."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import mlflow
import mlflow.pyfunc
import pandas as pd
from huggingface_hub import snapshot_download

from app.core.exceptions import ModelNotLoadedError
from app.schemas.prediction import (
    PredictionResponse,
    build_request_model,
    business_rule_validators,
    coerce_frame_dtypes,
    plausible_range_validators,
)
from home_credit_mlops.settings import ServingConfig

LOGGER = logging.getLogger(__name__)

ModelResolver = Callable[[ServingConfig], Path]
ModelLoader = Callable[[Path], mlflow.pyfunc.PyFuncModel]


def resolve_model_source(serving: ServingConfig) -> Path:
    """Telecharge le dossier MLflow du modele depuis Hugging Face Hub."""

    LOGGER.info(
        "Downloading scoring model from Hugging Face Hub: %s (revision=%s)",
        serving.model_repo_id,
        serving.revision,
    )
    local_dir = snapshot_download(
        repo_id=serving.model_repo_id,
        repo_type="model",
        revision=serving.revision,
        local_dir=serving.local_cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(local_dir)


def load_scoring_model(local_dir: Path) -> mlflow.pyfunc.PyFuncModel:
    """Charge le modele MLflow depuis un dossier local sans toucher au tracking projet."""

    LOGGER.info("Loading scoring model from %s", local_dir)
    mlflow.set_tracking_uri(f"sqlite:///{tempfile.mkdtemp()}/mlflow.db")
    return mlflow.pyfunc.load_model(local_dir.as_posix())


class ModelService:
    """Service responsable du modele : chargement, schema et prediction."""

    def __init__(
        self,
        *,
        resolve_model: ModelResolver = resolve_model_source,
        load_model: ModelLoader = load_scoring_model,
    ) -> None:
        self._resolve_model = resolve_model
        self._load_model = load_model
        self.model: mlflow.pyfunc.PyFuncModel | Any | None = None
        self.input_schema: Any | None = None
        self.request_model: type | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, serving: ServingConfig) -> None:
        """Charge le modele une seule fois et prepare son schema de requete."""

        local_dir = self._resolve_model(serving)
        self.model = self._load_model(local_dir)
        self.input_schema = self.model.metadata.get_input_schema()
        self.request_model = build_request_model(
            self.input_schema,
            business_rule_validators() + plausible_range_validators(),
        )

    def predict(self, payload: dict[str, Any]) -> PredictionResponse:
        """Execute une prediction et retourne une reponse metier typee."""

        if self.model is None or self.input_schema is None:
            raise ModelNotLoadedError("Scoring model is not loaded.")

        input_frame = pd.DataFrame([payload])
        input_frame = coerce_frame_dtypes(input_frame, self.input_schema)
        result = self.model.predict(input_frame)
        return PredictionResponse(**result.iloc[0].to_dict())


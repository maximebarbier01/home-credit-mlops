"""Telechargement et chargement du modele de scoring pour l'API.

Le modele (~130 Mo) n'est jamais embarque dans l'image Docker : il est
telecharge une seule fois, au demarrage du conteneur, depuis un depot
Hugging Face Hub (voir scripts/export_model_for_serving.py pour la
publication). Cela permet de promouvoir un nouveau champion sans
reconstruire l'image.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow.pyfunc
from huggingface_hub import snapshot_download

from home_credit_mlops.settings import ServingConfig

LOGGER = logging.getLogger(__name__)


def resolve_model_source(serving: ServingConfig) -> Path:
    """Telecharge le modele depuis Hugging Face Hub et retourne son chemin local.

    Le modele MLflow est un dossier complet (MLmodel, pickle du pipeline,
    fichiers d'environnement...) : on utilise donc snapshot_download
    (dossier entier) plutot que hf_hub_download (fichier unique).
    """
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
    """Charge le modele MLflow (wrapper CreditScoringModel) depuis un dossier local."""
    LOGGER.info("Loading scoring model from %s", local_dir)
    return mlflow.pyfunc.load_model(local_dir.as_posix())

"""Utilitaires MLflow pour le tracking, le registry, l'UI et le serving local."""

from __future__ import annotations

import argparse
import subprocess

import mlflow
from mlflow import MlflowClient

from home_credit_mlops.settings import Settings, load_settings


def configure_mlflow(settings: Settings) -> None:
    """Configure le tracking MLflow et initialise l'expérience."""

    # Création des dossiers nécessaires au stockage MLflow
    settings.mlflow.backend_store_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    settings.mlflow.artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Configuration du serveur de tracking et du Model Registry
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_registry_uri(settings.mlflow.tracking_uri)

    # Recherche de l'expérience MLflow
    experiment = mlflow.get_experiment_by_name(settings.mlflow.experiment_name)

    # Création de l'expérience si elle n'existe pas encore
    if experiment is None:
        mlflow.create_experiment(
            name=settings.mlflow.experiment_name,
            artifact_location=settings.mlflow.artifact_root.as_uri(),
        )

    # Définition de l'expérience active
    mlflow.set_experiment(settings.mlflow.experiment_name)


def register_logged_model(model_uri: str, model_name: str) -> str:
    """Enregistre un modele deja logge dans le Model Registry MLflow."""
    """Enregistre un modèle loggé dans le Model Registry."""

    client = MlflowClient()

    # Création du modèle enregistré s'il n'existe pas encore
    try:
        client.create_registered_model(model_name)
    except Exception:
        # Le modèle existe probablement déjà dans le registre
        pass

    # Création d'une nouvelle version du modèle
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    return str(result.version)


def ui_main() -> None:
    """Lance l'interface web MLflow."""

    parser = argparse.ArgumentParser(description="Start the MLflow UI.")

    # Arguments disponibles en ligne de commande
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--config",
        default="configs/default.toml",
    )

    args = parser.parse_args()

    # Chargement de la configuration du projet
    settings = load_settings(args.config)

    # Construction de la commande MLflow
    command = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        settings.mlflow.tracking_uri,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    # Exécution de l'interface MLflow
    subprocess.run(command, check=True)


def serve_main() -> None:
    """Expose localement un modèle MLflow via une API REST."""

    parser = argparse.ArgumentParser(description="Serve an MLflow model locally.")

    # URI du modèle à servir, par exemple une version du Model Registry
    parser.add_argument(
        "--model-uri",
        required=True,
        help="Example: models:/home-credit-scoring/1",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    args = parser.parse_args()

    # Construction de la commande de déploiement MLflow
    command = [
        "mlflow",
        "models",
        "serve",
        "--model-uri",
        args.model_uri,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--env-manager",
        "local",
    ]

    # Démarrage du serveur local de prédiction
    subprocess.run(command, check=True)

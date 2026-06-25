from __future__ import annotations

import argparse
import subprocess

import mlflow
from mlflow import MlflowClient

from home_credit_mlops.settings import Settings, load_settings


def configure_mlflow(settings: Settings) -> None:
    settings.mlflow.backend_store_path.parent.mkdir(parents=True, exist_ok=True)
    settings.mlflow.artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_registry_uri(settings.mlflow.tracking_uri)

    experiment = mlflow.get_experiment_by_name(settings.mlflow.experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            name=settings.mlflow.experiment_name,
            artifact_location=settings.mlflow.artifact_root.as_uri(),
        )

    mlflow.set_experiment(settings.mlflow.experiment_name)


def register_logged_model(model_uri: str, model_name: str) -> str:
    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    return str(result.version)


def ui_main() -> None:
    parser = argparse.ArgumentParser(description="Start the MLflow UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--config", default="configs/default.toml")
    args = parser.parse_args()

    settings = load_settings(args.config)
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
    subprocess.run(command, check=True)


def serve_main() -> None:
    parser = argparse.ArgumentParser(description="Serve an MLflow model locally.")
    parser.add_argument("--model-uri", required=True, help="Example: models:/home-credit-scoring/1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

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
    subprocess.run(command, check=True)

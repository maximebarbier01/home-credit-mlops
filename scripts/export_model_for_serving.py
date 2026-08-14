"""Publie un modele deja enregistre dans MLflow vers un depot Hugging Face Hub.

Etape manuelle et deliberee, separee de `register_champion_model.py` :
promouvoir un champion vers le registry local MLflow et le publier
publiquement sur Hugging Face Hub (pour que l'API le telecharge au
demarrage, voir `api/model_loader.py`) sont deux decisions distinctes.
"""

from __future__ import annotations

import argparse
import logging
import os

import mlflow.artifacts
from huggingface_hub import HfApi

from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.mlflow_utils import configure_mlflow
from home_credit_mlops.settings import load_settings

LOGGER = logging.getLogger(__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a registered MLflow model and publish it to a Hugging Face Hub model repo."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument(
        "--model-uri",
        required=True,
        help="Example: models:/home-credit-scoring/3",
    )
    parser.add_argument(
        "--hf-repo-id",
        required=True,
        help="Example: your-hf-username/home-credit-scoring",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch/revision to push to on the Hugging Face Hub repo.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face Hub repo as private (requires HF_TOKEN for downloads too).",
    )
    return parser


def export_model(*, model_uri: str, hf_repo_id: str, revision: str, private: bool) -> str:
    """Telecharge `model_uri` localement puis le pousse sur `hf_repo_id`. Retourne l'URL du depot."""
    token = os.environ.get("HF_TOKEN")

    LOGGER.info("Downloading artifacts for %s", model_uri)
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    api = HfApi(token=token)
    LOGGER.info("Creating (or reusing) Hugging Face Hub repo %s", hf_repo_id)
    repo_url = api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    LOGGER.info("Uploading %s to %s (revision=%s)", local_path, hf_repo_id, revision)
    api.upload_folder(
        repo_id=hf_repo_id,
        folder_path=local_path,
        repo_type="model",
        revision=revision,
        commit_message=f"Publish {model_uri}",
    )

    return str(repo_url)


def main() -> None:
    configure_logging()
    parser = _build_argument_parser()
    args = parser.parse_args()

    settings = load_settings(args.config)
    configure_mlflow(settings)

    repo_url = export_model(
        model_uri=args.model_uri,
        hf_repo_id=args.hf_repo_id,
        revision=args.revision,
        private=args.private,
    )

    print(f"Model published: {repo_url}")
    print(f"Set [serving].model_repo_id = \"{args.hf_repo_id}\" in configs/default.toml")
    print(f"Set [serving].revision = \"{args.revision}\" in configs/default.toml")


if __name__ == "__main__":
    main()

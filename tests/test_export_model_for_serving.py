from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_model_for_serving.py"
SPEC = importlib.util.spec_from_file_location("export_model_for_serving", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
export_model_for_serving = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_model_for_serving)


def test_export_model_downloads_then_uploads_without_network(monkeypatch) -> None:
    monkeypatch.setattr(
        export_model_for_serving.mlflow.artifacts,
        "download_artifacts",
        lambda artifact_uri: "/tmp/fake-local-model",
    )

    mock_api_instance = MagicMock()
    mock_api_instance.create_repo.return_value = "https://huggingface.co/some-user/home-credit-scoring"
    monkeypatch.setattr(
        export_model_for_serving,
        "HfApi",
        lambda token=None: mock_api_instance,
    )

    repo_url = export_model_for_serving.export_model(
        model_uri="models:/home-credit-scoring/3",
        hf_repo_id="some-user/home-credit-scoring",
        revision="main",
        private=False,
    )

    assert repo_url == "https://huggingface.co/some-user/home-credit-scoring"
    mock_api_instance.create_repo.assert_called_once_with(
        repo_id="some-user/home-credit-scoring",
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    mock_api_instance.upload_folder.assert_called_once_with(
        repo_id="some-user/home-credit-scoring",
        folder_path="/tmp/fake-local-model",
        repo_type="model",
        revision="main",
        commit_message="Publish models:/home-credit-scoring/3",
    )


def test_build_argument_parser_requires_model_uri_and_repo_id() -> None:
    parser = export_model_for_serving._build_argument_parser()

    args = parser.parse_args(
        ["--model-uri", "models:/home-credit-scoring/3", "--hf-repo-id", "user/repo"]
    )

    assert args.model_uri == "models:/home-credit-scoring/3"
    assert args.hf_repo_id == "user/repo"
    assert args.revision == "main"
    assert args.private is False

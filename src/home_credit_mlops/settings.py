from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    default_train_path: Path
    target_column: str
    id_column: str
    test_size: float
    random_state: int


@dataclass(frozen=True)
class BusinessConfig:
    fn_cost: float
    fp_cost: float
    threshold_grid_size: int


@dataclass(frozen=True)
class TrainingConfig:
    cv_folds: int
    n_jobs: int


@dataclass(frozen=True)
class MlflowConfig:
    experiment_name: str
    backend_store_path: Path
    artifact_root: Path

    @property
    def tracking_uri(self) -> str:
        return f"sqlite:///{self.backend_store_path.as_posix()}"


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig
    dataset: DatasetConfig
    business: BusinessConfig
    training: TrainingConfig
    mlflow: MlflowConfig


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_settings(config_path: str | Path = "configs/default.toml") -> Settings:
    config_file = _resolve_path(str(config_path))
    with config_file.open("rb") as stream:
        data = tomllib.load(stream)

    paths = data["paths"]
    dataset = data["dataset"]
    business = data["business"]
    training = data["training"]
    mlflow = data["mlflow"]

    return Settings(
        paths=PathsConfig(
            raw_dir=_resolve_path(paths["raw_dir"]),
            interim_dir=_resolve_path(paths["interim_dir"]),
            processed_dir=_resolve_path(paths["processed_dir"]),
            reports_dir=_resolve_path(paths["reports_dir"]),
            artifacts_dir=_resolve_path(paths["artifacts_dir"]),
        ),
        dataset=DatasetConfig(
            default_train_path=_resolve_path(dataset["default_train_path"]),
            target_column=dataset["target_column"],
            id_column=dataset["id_column"],
            test_size=float(dataset["test_size"]),
            random_state=int(dataset["random_state"]),
        ),
        business=BusinessConfig(
            fn_cost=float(business["fn_cost"]),
            fp_cost=float(business["fp_cost"]),
            threshold_grid_size=int(business["threshold_grid_size"]),
        ),
        training=TrainingConfig(
            cv_folds=int(training["cv_folds"]),
            n_jobs=int(training["n_jobs"]),
        ),
        mlflow=MlflowConfig(
            experiment_name=mlflow["experiment_name"],
            backend_store_path=_resolve_path(mlflow["backend_store_path"]),
            artifact_root=_resolve_path(mlflow["artifact_root"]),
        ),
    )

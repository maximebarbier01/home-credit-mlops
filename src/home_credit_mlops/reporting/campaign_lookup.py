"""Recherche du champion retenu dans les artefacts d'une campagne de benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def find_latest_campaign_metadata(reports_root: Path, campaign_name: str) -> Path | None:
    """Retrouve le campaign_metadata.json le plus recent pour cette campagne.

    Chaque run de campagne (`run_home_credit_experiment.py`) ecrit un
    `campaign_metadata.json` contenant le champion retenu (seuil metier et
    meilleurs hyperparametres). Les scripts qui ont besoin de ce champion
    (`register_champion_model.py`, `analyze_fairness.py`) le relisent ici
    plutot que de dupliquer ces valeurs en dur, ce qui les desynchroniserait
    silencieusement d'une prochaine campagne.
    """
    candidates: list[tuple[str, Path]] = []
    for metadata_path in reports_root.glob("*/*/campaign_metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("campaign_name") != campaign_name or "best_model" not in metadata:
            continue
        candidates.append((str(metadata.get("created_at", "")), metadata_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def load_champion_from_campaign(metadata_path: Path) -> dict[str, Any]:
    """Extrait le champion (modele, seuil, hyperparametres) d'un campaign_metadata.json."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    best_model = metadata["best_model"]
    return {
        "model_name": best_model["base_model_name"],
        # Cle du candidat telle qu'utilisee dans les noms de fichiers
        # d'artefacts (ex. "lightgbm__smote"), distincte de "model_name"
        # (nom du modele de base seul, ex. "lightgbm").
        "candidate_key": best_model["model_name"],
        "sampling_strategy": best_model["sampling_strategy"],
        "threshold": float(best_model["threshold"]),
        "params": dict(best_model["best_params"]),
        "campaign_name": metadata.get("campaign_name"),
        "source_path": metadata_path.as_posix(),
    }

"""Exceptions métier de la couche API."""

from __future__ import annotations


class ModelNotLoadedError(RuntimeError):
    """Le modèle de scoring n'est pas disponible pour l'inférence."""

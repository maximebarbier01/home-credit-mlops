"""Exceptions metier de la couche API."""

from __future__ import annotations


class ModelNotLoadedError(RuntimeError):
    """Le modele de scoring n'est pas disponible pour l'inference."""


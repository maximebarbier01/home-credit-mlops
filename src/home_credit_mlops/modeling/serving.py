"""Modele MLflow exposant une reponse directement exploitable par le metier."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd


class CreditScoringModel(mlflow.pyfunc.PythonModel):
    """Encapsule le pipeline et applique son seuil metier versionne."""

    def __init__(self, pipeline: Any, business_threshold: float) -> None:
        if not 0.0 <= business_threshold <= 1.0:
            raise ValueError("`business_threshold` must be between 0 and 1.")

        self.pipeline = pipeline
        self.business_threshold = float(business_threshold)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Retourne la probabilite et la decision calculee au seuil metier."""

        del context, params
        probabilities = np.asarray(self.pipeline.predict_proba(model_input))[:, 1]
        predicted_defaults = (probabilities >= self.business_threshold).astype(int)

        return pd.DataFrame(
            {
                "default_probability": probabilities,
                "business_threshold": np.full(len(model_input), self.business_threshold),
                "predicted_default": predicted_defaults,
                "credit_decision": np.where(
                    predicted_defaults == 1,
                    "refused",
                    "approved",
                ),
            },
            index=model_input.index,
        )

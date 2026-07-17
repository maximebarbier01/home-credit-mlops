from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit_mlops.modeling.serving import CreditScoringModel


class ProbabilityPipelineStub:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = self.probabilities[: len(features)]
        return np.column_stack([1.0 - probabilities, probabilities])


def test_credit_scoring_model_applies_business_threshold() -> None:
    features = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})
    model = CreditScoringModel(
        pipeline=ProbabilityPipelineStub([0.10, 0.22, 0.80]),
        business_threshold=0.22,
    )

    predictions = model.predict(context=None, model_input=features)

    assert predictions["default_probability"].tolist() == [0.10, 0.22, 0.80]
    assert predictions["business_threshold"].tolist() == [0.22, 0.22, 0.22]
    assert predictions["predicted_default"].tolist() == [0, 1, 1]
    assert predictions["credit_decision"].tolist() == ["approved", "refused", "refused"]


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_credit_scoring_model_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        CreditScoringModel(
            pipeline=ProbabilityPipelineStub([0.5]),
            business_threshold=threshold,
        )

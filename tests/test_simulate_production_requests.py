from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.simulate_production_requests import _build_payloads


def test_build_payloads_drops_identifiers_and_can_add_invalid_requests(tmp_path: Path) -> None:
    data_path = tmp_path / "test_features.parquet"
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "AGE_YEARS": [35.0, 42.0],
            "AMT_INCOME_TOTAL": [50_000.0, 60_000.0],
            "AMT_CREDIT": [200_000.0, 250_000.0],
        }
    ).to_parquet(data_path, index=False)

    payloads = _build_payloads(
        data_path,
        sample_size=2,
        random_state=42,
        invalid_requests=1,
    )

    assert len(payloads) == 3
    assert "SK_ID_CURR" not in payloads[0]
    assert "TARGET" not in payloads[0]
    assert "AMT_INCOME_TOTAL" not in payloads[-1]


def test_build_payloads_sanitizes_unintended_invalid_values(tmp_path: Path) -> None:
    data_path = tmp_path / "test_features.parquet"
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "AGE_YEARS": [np.nan, -5.0],
            "AMT_INCOME_TOTAL": [np.nan, 0.0],
            "AMT_CREDIT": [np.inf, -1.0],
            "CNT_CHILDREN": [-1, np.nan],
            "CNT_FAM_MEMBERS": [np.nan, -2.0],
            "EXT_SOURCE_1": [np.nan, 2.0],
            "FLAG_MOBIL": [np.nan, 2.0],
            "CODE_GENDER": [None, "F"],
        }
    ).to_parquet(data_path, index=False)

    payloads = _build_payloads(
        data_path,
        sample_size=2,
        random_state=42,
        invalid_requests=0,
    )

    assert len(payloads) == 2
    for payload in payloads:
        assert all(value is not None for value in payload.values())
        assert all(
            not (isinstance(value, float) and (np.isnan(value) or np.isinf(value)))
            for value in payload.values()
        )
        assert 18 <= payload["AGE_YEARS"] < 100
        assert payload["AMT_INCOME_TOTAL"] > 0
        assert payload["AMT_CREDIT"] > 0
        assert payload["CNT_CHILDREN"] >= 0
        assert payload["CNT_FAM_MEMBERS"] >= 0
        assert 0.0 <= payload["EXT_SOURCE_1"] <= 1.0
        assert payload["FLAG_MOBIL"] in (0, 1)

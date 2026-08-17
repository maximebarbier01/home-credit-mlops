from __future__ import annotations

from pathlib import Path

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

"""Profile le temps de réponse observé de l'API de scoring."""

from __future__ import annotations

import argparse
import cProfile
from datetime import datetime
import logging
from pathlib import Path
import pstats

import pandas as pd

from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.settings import load_settings
from scripts.simulate_production_requests import _build_payloads, _post_json

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    default_data_path = settings.paths.processed_dir / "test_features.parquet"
    default_output_dir = (
        settings.paths.reports_dir
        / f"{datetime.now():%Y%m%d}_home_credit_performance"
        / f"{datetime.now():%Y%m%d_%H%M%S}_api_profile"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Envoie des requêtes HTTP à l'API, mesure les latences client et "
            "sauvegarde un profil cProfile du client de test."
        )
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--data", default=default_data_path.as_posix())
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output-dir", default=default_output_dir.as_posix())
    parser.add_argument("--profile-lines", type=int, default=40)
    return parser.parse_args()


def _send_requests(
    *,
    api_url: str,
    payloads: list[dict],
    timeout: float,
    api_key: str | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for index, payload in enumerate(payloads, start=1):
        result = _post_json(api_url, payload, timeout=timeout, api_key=api_key)
        status_code = result["status_code"]
        rows.append(
            {
                "request_index": index,
                "status_code": status_code,
                "success": bool(status_code is not None and 200 <= int(status_code) < 400),
                "client_latency_ms": result["latency_ms"],
                "error": result["error"],
            }
        )
        LOGGER.info(
            "Profile request %s/%s -> status=%s latency=%.1fms",
            index,
            len(payloads),
            status_code,
            result["latency_ms"],
        )
    return pd.DataFrame(rows)


def _summarize_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    latencies = pd.to_numeric(measurements["client_latency_ms"], errors="coerce").dropna()
    return pd.DataFrame(
        [
            {
                "request_count": len(measurements),
                "success_count": int(measurements["success"].sum()),
                "error_count": int((~measurements["success"]).sum()),
                "client_latency_mean_ms": float(latencies.mean()) if not latencies.empty else 0.0,
                "client_latency_p50_ms": float(latencies.quantile(0.50))
                if not latencies.empty
                else 0.0,
                "client_latency_p95_ms": float(latencies.quantile(0.95))
                if not latencies.empty
                else 0.0,
                "client_latency_p99_ms": float(latencies.quantile(0.99))
                if not latencies.empty
                else 0.0,
                "client_latency_max_ms": float(latencies.max()) if not latencies.empty else 0.0,
            }
        ]
    )


def _write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            safe_name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.book[safe_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions


def main() -> None:
    configure_logging()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = _build_payloads(
        Path(args.data),
        sample_size=args.sample_size + args.warmup_requests,
        random_state=args.random_state,
        invalid_requests=0,
    )
    warmup_payloads = payloads[: args.warmup_requests]
    measured_payloads = payloads[args.warmup_requests :]

    if warmup_payloads:
        LOGGER.info("Running %s warmup requests.", len(warmup_payloads))
        _send_requests(
            api_url=args.api_url,
            payloads=warmup_payloads,
            timeout=args.timeout,
            api_key=args.api_key,
        )

    profiler = cProfile.Profile()
    profiler.enable()
    measurements = _send_requests(
        api_url=args.api_url,
        payloads=measured_payloads,
        timeout=args.timeout,
        api_key=args.api_key,
    )
    profiler.disable()

    profile_path = output_dir / "cprofile_top.txt"
    with profile_path.open("w", encoding="utf-8") as stream:
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(args.profile_lines)

    summary = _summarize_measurements(measurements)
    workbook_path = output_dir / "api_profile_summary.xlsx"
    _write_workbook(
        workbook_path,
        {
            "summary": summary,
            "measurements": measurements,
        },
    )

    LOGGER.info("API profile workbook written to %s", workbook_path)
    LOGGER.info("cProfile top functions written to %s", profile_path)


if __name__ == "__main__":
    main()

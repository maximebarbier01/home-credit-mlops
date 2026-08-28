"""Teste la conversion ONNX Runtime du pipeline champion et la compare au
pipeline natif (imblearn/scikit-learn/LightGBM) sur precision et latence.

Piste explicitement listee par la consigne de l'etape 4 ("quantification,
optimisation de code, hardware") et jusqu'ici seulement documentee comme
"ecartee a ce stade" dans home_credit_mlops.performance.report sans test
empirique. Ce script fait le test reel et ecrit un rapport avec les
resultats mesures (precision ET latence), pour decider en connaissance de
cause plutot que par hypothese.

Necessite le groupe Poetry optionnel `onnx-benchmark` :
    poetry install --with onnx-benchmark
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from sklearn.pipeline import Pipeline as SkPipeline

from app.services.model_service import load_scoring_model, resolve_model_source
from home_credit_mlops.logging_utils import configure_logging
from home_credit_mlops.settings import load_settings

LOGGER = logging.getLogger(__name__)

_TARGET_OPSET = {"": 17, "ai.onnx.ml": 3}


def _default_output_dir(reports_dir: str | Path) -> Path:
    now = time.strftime("%Y%m%d_%H%M%S")
    return Path(reports_dir) / f"{now[:8]}_home_credit_performance" / f"{now}_onnx_benchmark"


@dataclass(frozen=True)
class OnnxBenchmarkResult:
    n_rows_correctness: int
    n_rows_latency: int
    max_abs_proba_diff: float
    mean_abs_proba_diff: float
    business_threshold: float
    decisions_flipped: int
    native_latency_ms: dict[str, float]
    onnx_latency_ms: dict[str, float]
    speedup_factor: float
    report_path: Path


def _register_lightgbm_converter() -> None:
    update_registered_converter(
        LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes,
        convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )


def _fill_categorical(series: pd.Series, fill_value: object) -> pd.Series:
    # Quelques colonnes (OCCUPATION_TYPE, FONDKAPREMONT_MODE, HOUSETYPE_MODE,
    # WALLSMATERIAL_MODE) ont tellement de valeurs manquantes que None est
    # lui-meme la statistique "most_frequent" apprise par le SimpleImputer :
    # aucune imputation reelle n'a lieu pour elles, et le OneHotEncoder
    # entraine a appris None comme categorie explicite. On laisse NaN tel
    # quel pour que .astype(str) produise la meme chaine "nan" que celle vue
    # a l'entrainement.
    if fill_value is None:
        return series
    return series.fillna(fill_value)


def convert_champion_to_onnx(
    pipeline, *, numeric_cols: list[str], categorical_cols: list[str], cat_fill_map: dict
) -> bytes:
    """Convertit le pipeline (preprocessing + LightGBM) champion en ONNX.

    L'etape SMOTE du pipeline imblearn est un no-op en predict (resampling
    uniquement au fit) : seuls le preprocesseur et le modele sont convertis.
    Le SimpleImputer categoriel n'est pas convertible tel quel (l'operateur
    ONNX Imputer ne supporte pas un sentinel "manquant" sur un tenseur
    string) : son effet est deterministe et reproduit en pandas avant
    l'inference ONNX (voir _fill_categorical), l'encodeur one-hot deja
    entraine restant seul dans le graphe ONNX pour la branche categorielle.
    """
    _register_lightgbm_converter()

    preprocessor = copy.deepcopy(pipeline.named_steps["preprocessor"])
    for name, trans, _cols in preprocessor.transformers_:
        if name == "categorical":
            trans.steps = [("encoder", trans.named_steps["encoder"])]

    sk_pipe = SkPipeline(
        steps=[("preprocessor", preprocessor), ("model", pipeline.named_steps["model"])]
    )

    initial_types = [(col, FloatTensorType([None, 1])) for col in numeric_cols]
    initial_types += [(col, StringTensorType([None, 1])) for col in categorical_cols]

    onnx_model = convert_sklearn(sk_pipe, initial_types=initial_types, target_opset=_TARGET_OPSET)
    return onnx_model.SerializeToString()


def _prepare_onnx_frame(
    frame: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cat_fill_map: dict,
) -> pd.DataFrame:
    out = frame[numeric_cols + categorical_cols].copy()
    for col in numeric_cols:
        out[col] = out[col].astype(np.float32)
    for col in categorical_cols:
        out[col] = _fill_categorical(out[col], cat_fill_map[col]).astype(str)
    return out


def _percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p))


def _latency_stats(values_ms: list[float]) -> dict[str, float]:
    arr = np.array(values_ms)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": _percentile(arr, 50),
        "p95_ms": _percentile(arr, 95),
        "p99_ms": _percentile(arr, 99),
    }


def _write_report(
    path: Path,
    *,
    n_rows_correctness: int,
    n_rows_latency: int,
    max_abs_diff: float,
    mean_abs_diff: float,
    threshold: float,
    flipped: int,
    native_stats: dict[str, float],
    onnx_stats: dict[str, float],
    speedup: float,
) -> None:
    flipped_rate = flipped / n_rows_correctness if n_rows_correctness else 0.0
    lines = [
        "# Test d'optimisation : conversion ONNX Runtime du pipeline champion",
        "",
        "Test empirique de la piste documentee comme \"ecartee a ce stade\" dans",
        "`home_credit_mlops.performance.report.build_optimization_decisions` : conversion",
        "du pipeline champion (preprocessing scikit-learn + LightGBM) en ONNX et comparaison",
        "avec le pipeline natif, sur des lignes reelles issues de "
        "`data/processed/train_features.parquet`.",
        "",
        "## Precision (regression fonctionnelle)",
        "",
        f"- Lignes comparees : {n_rows_correctness}.",
        f"- Ecart absolu maximum sur `default_probability` : {max_abs_diff:.4f}.",
        f"- Ecart absolu moyen sur `default_probability` : {mean_abs_diff:.4f}.",
        f"- Seuil metier : {threshold:.4f}.",
        (
            f"- Decisions credit (`credit_decision`) qui changent entre natif et ONNX : "
            f"{flipped} / {n_rows_correctness} ({flipped_rate:.2%})."
        ),
        "",
        "## Latence (inference a une ligne, comme /predict en production)",
        "",
        f"- Lignes mesurees : {n_rows_latency} (apres warmup).",
        (
            f"- Natif : moyenne={native_stats['mean_ms']:.3f} ms, "
            f"p50={native_stats['p50_ms']:.3f} ms, p95={native_stats['p95_ms']:.3f} ms, "
            f"p99={native_stats['p99_ms']:.3f} ms."
        ),
        (
            f"- ONNX Runtime : moyenne={onnx_stats['mean_ms']:.3f} ms, "
            f"p50={onnx_stats['p50_ms']:.3f} ms, p95={onnx_stats['p95_ms']:.3f} ms, "
            f"p99={onnx_stats['p99_ms']:.3f} ms."
        ),
        f"- Facteur d'acceleration ONNX (moyenne) : {speedup:.2f}x.",
        "",
        "## Decision",
        "",
    ]
    if flipped > 0:
        lines.append(
            "**ONNX Runtime n'est pas retenu.** La conversion introduit un ecart numerique "
            f"non nul sur `default_probability` qui fait basculer {flipped} decision(s) credit "
            "sur l'echantillon teste, proches du seuil metier. Le pipeline natif reste la "
            "source de verite pour la decision deja validee (voir l'analyse fairness/business "
            "cost de la partie modelisation) ; un gain de latence ne justifie pas un risque de "
            "regression sur des decisions de credit individuelles."
        )
    else:
        lines.append(
            "Aucune decision credit ne change sur l'echantillon teste. Le gain de latence "
            "mesure ci-dessus reste a mettre en balance avec la complexite operationnelle "
            "ajoutee (pipeline de conversion, maintenance du graphe ONNX en cas de "
            "reentrainement) avant adoption en production."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    *,
    reference_data_path: str | Path,
    output_dir: str | Path,
    n_rows_correctness: int,
    n_rows_latency: int,
    random_state: int,
) -> OnnxBenchmarkResult:
    settings = load_settings()
    local_dir = resolve_model_source(settings.serving)
    pyfunc_model = load_scoring_model(local_dir)
    inner = pyfunc_model.unwrap_python_model()
    pipeline = inner.pipeline
    threshold = inner.business_threshold

    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    cat_fill_map: dict[str, object] = {}
    for name, trans, cols in preprocessor.transformers_:
        if name == "numeric":
            numeric_cols = list(cols)
        elif name == "categorical":
            categorical_cols = list(cols)
            cat_imputer = trans.named_steps["imputer"]
            cat_fill_map = dict(zip(cols, cat_imputer.statistics_))

    LOGGER.info("Converting champion pipeline to ONNX (%d + %d columns)...", len(numeric_cols), len(categorical_cols))
    onnx_bytes = convert_champion_to_onnx(
        pipeline,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_fill_map=cat_fill_map,
    )

    import onnxruntime as rt

    session = rt.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    ordered_cols = numeric_cols + categorical_cols

    frame = pd.read_parquet(reference_data_path)
    drop_cols = [c for c in ("SK_ID_CURR", "TARGET") if c in frame.columns]
    features = frame.drop(columns=drop_cols)

    sample = features.sample(n=n_rows_correctness, random_state=random_state)
    onnx_sample = _prepare_onnx_frame(
        sample,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_fill_map=cat_fill_map,
    )

    native_proba = np.asarray(pipeline.predict_proba(sample))[:, 1]
    onnx_inputs = {
        onnx_input.name: onnx_sample[[col]].to_numpy().reshape(-1, 1)
        for onnx_input, col in zip(session.get_inputs(), ordered_cols)
    }
    onnx_out = session.run(None, onnx_inputs)
    proba_output = np.array([row[1] for row in onnx_out[1]])

    diff = np.abs(native_proba - proba_output)
    native_decision = (native_proba >= threshold).astype(int)
    onnx_decision = (proba_output >= threshold).astype(int)
    flipped = int((native_decision != onnx_decision).sum())

    latency_sample = features.sample(n=n_rows_latency + 10, random_state=random_state + 1)
    latency_onnx_frame = _prepare_onnx_frame(
        latency_sample,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_fill_map=cat_fill_map,
    )
    native_rows = [latency_sample.iloc[[i]] for i in range(len(latency_sample))]
    onnx_rows = [
        {
            onnx_input.name: latency_onnx_frame.iloc[[i]][[col]].to_numpy().reshape(-1, 1)
            for onnx_input, col in zip(session.get_inputs(), ordered_cols)
        }
        for i in range(len(latency_sample))
    ]

    for row in native_rows[:10]:
        pipeline.predict_proba(row)
    for row in onnx_rows[:10]:
        session.run(None, row)

    native_times: list[float] = []
    for row in native_rows[10:]:
        t0 = time.perf_counter()
        pipeline.predict_proba(row)
        native_times.append((time.perf_counter() - t0) * 1000)

    onnx_times: list[float] = []
    for row in onnx_rows[10:]:
        t0 = time.perf_counter()
        session.run(None, row)
        onnx_times.append((time.perf_counter() - t0) * 1000)

    native_stats = _latency_stats(native_times)
    onnx_stats = _latency_stats(onnx_times)
    speedup = native_stats["mean_ms"] / onnx_stats["mean_ms"]

    output = Path(output_dir)
    report_path = output / "onnx_benchmark_report.md"
    _write_report(
        report_path,
        n_rows_correctness=n_rows_correctness,
        n_rows_latency=len(native_times),
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        threshold=threshold,
        flipped=flipped,
        native_stats=native_stats,
        onnx_stats=onnx_stats,
        speedup=speedup,
    )

    onnx_path = output / "champion_pipeline.onnx"
    onnx_path.write_bytes(onnx_bytes)

    return OnnxBenchmarkResult(
        n_rows_correctness=n_rows_correctness,
        n_rows_latency=len(native_times),
        max_abs_proba_diff=float(diff.max()),
        mean_abs_proba_diff=float(diff.mean()),
        business_threshold=threshold,
        decisions_flipped=flipped,
        native_latency_ms=native_stats,
        onnx_latency_ms=onnx_stats,
        speedup_factor=speedup,
        report_path=report_path,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convertit le pipeline champion en ONNX Runtime et compare precision/latence "
            "au pipeline natif, sur des lignes reelles."
        )
    )
    parser.add_argument("--reference-data", default=None, help="Defaut : train_features.parquet")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-rows-correctness", type=int, default=2000)
    parser.add_argument("--n-rows-latency", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    configure_logging()
    parser = _build_argument_parser()
    args = parser.parse_args()

    settings = load_settings()
    reference_data = args.reference_data or settings.dataset.default_train_path
    output_dir = (
        Path(args.output_dir) if args.output_dir else _default_output_dir(settings.paths.reports_dir)
    )

    result = run_benchmark(
        reference_data_path=reference_data,
        output_dir=output_dir,
        n_rows_correctness=args.n_rows_correctness,
        n_rows_latency=args.n_rows_latency,
        random_state=args.random_state,
    )

    print(f"Report written to {result.report_path}")
    print(f"max_abs_proba_diff={result.max_abs_proba_diff:.4f} decisions_flipped={result.decisions_flipped}")
    print(f"native mean_ms={result.native_latency_ms['mean_ms']:.3f} onnx mean_ms={result.onnx_latency_ms['mean_ms']:.3f} speedup={result.speedup_factor:.2f}x")


if __name__ == "__main__":
    main()

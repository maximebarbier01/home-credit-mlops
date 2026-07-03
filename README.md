# Home Credit MLOps

Base project for the OpenClassrooms "Initiez-vous au MLOps" credit scoring exercise.

## Main entrypoints

- `scripts/build_home_credit_dataset.py`: data preparation, feature engineering, and final dataset EDA from the raw Home Credit tables
- `scripts/run_home_credit_experiment.py`: model training, benchmark, threshold optimization, SHAP, Excel exports, and MLflow tracking
- `scripts/mlflow_ui.py`: start the MLflow UI during the MLOps phase
- detailed French guide: `docs/mode_emploi_pipeline_ml.md`

## Project layout

```text
home-credit-mlops/
|-- configs/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- docs/
|-- scripts/
|-- src/home_credit_mlops/
|   |-- data/
|   |-- eda/
|   |-- features/
|   |-- modeling/
|   `-- reporting/
`-- tests/
```

## Build schema

The project now follows one main ML path from end to end:

1. `data/home_credit.py`
   prepares, cleans, joins, enriches, and documents the final modeling dataset.
2. `eda/diagnostics.py`
   produces the final dataset EDA and data-quality reports during the data-preparation step.
3. `features/preprocessing.py`
   defines the preprocessing used by the models.
4. `modeling/benchmark.py`
   trains and compares candidate models with cross-validation, evaluates them,
   optimizes the business decision threshold, exports diagnostics, and optionally logs everything in MLflow.
4. `modeling/interpretability.py`
   exports global feature importance and local and global SHAP explanations.
5. `reporting/excel.py`
   bundles the experiment outputs into Excel workbooks.

So the pattern stays simple:

- `scripts/` contains a few executable entrypoints
- `src/home_credit_mlops/` contains the reusable, testable, importable logic

## Important note about the environment

Poetry is installed in WSL on this machine, not in Windows PowerShell.
Run the project commands from WSL, or through `wsl bash -lc ...`.

The project is configured for Python `>=3.11,<3.13`, which matches the current WSL Python setup.

## Quick start

```bash
cd /home/maxime/projects/home-credit-mlops
poetry install
```

## Suggested workflow

1. Put the Kaggle files in `data/raw/`

2. Build the cleaned, feature-engineered dataset and the full data-preparation EDA package:

```bash
poetry run python scripts/build_home_credit_dataset.py
```

3. Run the unified experiment pipeline:

```bash
poetry run python scripts/run_home_credit_experiment.py --campaign-name dev_lightgbm_5k_cv3 --model lightgbm --sample-size 5000 --cv-folds 3 --n-jobs 1
```

For WSL / VS Code stability, keep development runs focused on one model with a sample and `--n-jobs 1`. Reserve full multi-model, full-dataset benchmarks for the final phase.

4. Open the MLflow UI when you want to inspect the tracked runs:

```bash
poetry run python scripts/mlflow_ui.py
```

5. If needed, disable tracking for a quick local dry run:

```bash
poetry run python scripts/run_home_credit_experiment.py --skip-mlflow --sample-size 3000 --cv-folds 3 --n-jobs 1
```

## What the unified experiment exports

Each run under `reports/YYYYMMDD_home_credit_experiments/<timestamp>_<campaign_name>/` includes:

- a root `summary.xlsx` workbook with cross-model comparison sheets such as `campaign_overview`, `model_performance_summary`, `cv_summary`, `holdout_summary`, `decision_threshold_summary`, and `mlflow_runs`
- per-folder Excel workbooks for `interpretability`, `diagnostics`, `predictions`, and `cv_results`
- OOF and holdout prediction parquet files
- ROC, PR, and confusion-matrix diagnostics
- grouped feature importance
- SHAP global and local explanations
- campaign-level metadata and MLflow run mapping
- decision-threshold metadata
- packaged reports where CSV exports are converted to Excel tabs and then removed

## What this scaffold already covers

- reusable package structure
- feature aggregation from the main Home Credit tables
- cross-validation and hyperparameter search
- business cost with heavier false-negative penalty
- threshold optimization on out-of-fold probabilities
- SHAP-based interpretability exports
- Excel bundling of experiment artifacts
- MLflow experiment tracking and local model registry support

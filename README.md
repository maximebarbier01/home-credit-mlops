# Home Credit MLOps

Base project for the OpenClassrooms "Initiez-vous au MLOps" credit scoring exercise.

## Main entrypoints

- `scripts/build_home_credit_dataset.py`: step-1 dataset build from the raw Home Credit tables
- `scripts/run_home_credit_experiment.py`: unified ML build with EDA, preprocessing, benchmark, threshold optimization, SHAP, Excel exports, and MLflow tracking
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
   prepares, cleans, joins, and enriches the raw Home Credit tables.
2. `features/preprocessing.py`
   defines the preprocessing used by the models.
3. `modeling/benchmark.py`
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

2. Build the cleaned and aggregated feature dataset:

```bash
poetry run python scripts/build_home_credit_dataset.py
```

3. Run the unified experiment pipeline:

```bash
poetry run python scripts/run_home_credit_experiment.py --model lightgbm --sample-size 5000 --cv-folds 3
```

4. Open the MLflow UI when you want to inspect the tracked runs:

```bash
poetry run python scripts/mlflow_ui.py
```

5. If needed, disable tracking for a quick local dry run:

```bash
poetry run python scripts/run_home_credit_experiment.py --skip-mlflow --sample-size 3000 --cv-folds 3
```

## What the unified experiment exports

Each run under `reports/home_credit_experiments/<timestamp>/` includes:

- benchmark tables and metadata
- OOF and holdout predictions
- ROC, PR, and confusion-matrix diagnostics
- grouped feature importance
- SHAP global and local explanations
- decision-threshold metadata
- Excel workbooks for `eda`, `interpretability`, `diagnostics`, `predictions`, `cv_results`, and the root summary

## What this scaffold already covers

- reusable package structure
- feature aggregation from the main Home Credit tables
- cross-validation and hyperparameter search
- business cost with heavier false-negative penalty
- threshold optimization on out-of-fold probabilities
- SHAP-based interpretability exports
- Excel bundling of experiment artifacts
- MLflow experiment tracking and local model registry support

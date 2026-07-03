# Home Credit MLOps

Base project for the OpenClassrooms "Initiez-vous au MLOps" credit scoring exercise.

## Main entrypoints

- `scripts/build_home_credit_dataset.py`: step-1 dataset build from the raw Home Credit tables
- `scripts/run_home_credit_experiment.py`: consolidated experiment pipeline with EDA, model benchmark, threshold optimization, SHAP, and Excel exports
- `scripts/mlflow_ui.py`: start the MLflow UI during the MLOps phase

## Project layout

```text
home-credit-mlops/
|-- configs/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- scripts/
|-- src/home_credit_mlops/
|   |-- data/
|   |-- eda/
|   |-- features/
|   |-- modeling/
|   `-- reporting/
`-- tests/
```

## Pattern used

This is the compromise that keeps the project readable without turning `scripts/` into a dumping ground:

- `scripts/` contains a few executable entrypoints
- `src/home_credit_mlops/` contains the reusable, testable, importable logic

So yes, your intuition was right: the issue was not the existence of scripts, but the fact that we had too many tiny wrappers.

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

3. Run the consolidated experiment pipeline:

```bash
poetry run python scripts/run_home_credit_experiment.py --model lightgbm --sample-size 5000 --cv-folds 3
```

4. Open the MLflow UI when you work on the dedicated MLOps tracking step:

```bash
poetry run python scripts/mlflow_ui.py
```

## What the consolidated experiment exports

Each run under `reports/home_credit_experiments/<timestamp>/` includes:

- benchmark tables and metadata
- OOF and holdout predictions
- ROC, PR, and confusion-matrix diagnostics
- grouped feature importance
- SHAP global and local explanations
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

# Home Credit MLOps

Base project for the OpenClassrooms "Initiez-vous au MLOps" credit scoring exercise.

## Main commands

- `poetry run home-credit-build-dataset`: step-1 dataset build from the raw Home Credit tables
- `poetry run home-credit-run-experiment`: consolidated experiment pipeline with EDA, model benchmark, threshold optimization, SHAP, and Excel exports
- `poetry run home-credit-compare-models`: MLflow-oriented model comparison utility kept for the MLOps phase
- `poetry run home-credit-train-model`: MLflow-oriented single-model training utility
- `poetry run home-credit-profile-raw`: optional raw-table profiling helper
- `poetry run home-credit-mlflow-ui`: start the MLflow UI
- `poetry run home-credit-serve-model`: test MLflow model serving locally

## Project layout

```text
home-credit-mlops/
|-- configs/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- src/home_credit_mlops/
|   |-- data/
|   |-- eda/
|   |-- features/
|   |-- modeling/
|   `-- reporting/
`-- tests/
```

## Why the code stays in `src/`

Keeping the full implementation in `src/home_credit_mlops/` is still the cleanest option.
The CLI commands are only entrypoints.

This gives you both:

- simple terminal usage with `poetry run home-credit-...`
- reusable, testable, importable code in the package

So your intuition was right that we had too much dispersion, but the fix was to reduce wrappers, not to move all the real logic into a single script file.

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
poetry run home-credit-build-dataset
```

3. Run the consolidated experiment pipeline:

```bash
poetry run home-credit-run-experiment --model lightgbm --sample-size 5000 --cv-folds 3
```

4. Open the MLflow UI when you work on the dedicated MLOps tracking step:

```bash
poetry run home-credit-mlflow-ui
```

5. Serve a registered model locally:

```bash
poetry run home-credit-serve-model --model-uri models:/home-credit-scoring/1
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

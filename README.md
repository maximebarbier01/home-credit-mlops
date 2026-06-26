# Home Credit MLOps

Base project for the OpenClassrooms "Initiez-vous au MLOps" credit scoring exercise.

This repository is organized around Python scripts instead of notebooks:

- `scripts/profile_home_credit_raw.py` for a raw-table overview and missing-value profiling
- `scripts/build_home_credit_dataset.py` for the full step-1 cleaning, aggregation, and merge pipeline
- `scripts/eda.py` for exploratory analysis on a single exported table
- `scripts/prepare_dataset.py` for a generic first cleaned dataset export
- `scripts/compare_models.py` for model comparison with MLflow tracking
- `scripts/train_model.py` for single-model training
- `scripts/mlflow_ui.py` to start the MLflow UI
- `scripts/serve_model.py` to test MLflow model serving

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
|   `-- modeling/
`-- tests/
```

## Important note about the environment

Poetry is installed in WSL on this machine, not in Windows PowerShell.
Run the project commands from WSL, or through `wsl bash -lc ...`.

The project is configured for Python `>=3.11,<3.13`, which matches the current WSL Python 3.12 setup.

## Quick start

```bash
cd /home/maxime/projects/home-credit-mlops
poetry install
```

## Suggested workflow

1. Put the Kaggle files in `data/raw/`
2. Profile the raw Home Credit tables:

```bash
poetry run python scripts/profile_home_credit_raw.py
```

3. Build the cleaned and aggregated feature dataset:

```bash
poetry run python scripts/build_home_credit_dataset.py
```

4. Compare candidate models:

```bash
poetry run python scripts/compare_models.py   --data data/processed/train_features.parquet   --target TARGET   --id-column SK_ID_CURR   --register-model-name home-credit-scoring
```

5. Open the MLflow UI:

```bash
poetry run python scripts/mlflow_ui.py
```

6. Serve a registered model:

```bash
poetry run python scripts/serve_model.py --model-uri models:/home-credit-scoring/1
```

## What this scaffold already covers

- script-first project structure
- reusable `src/` package
- cross-validation and hyperparameter search
- business cost with heavier false-negative penalty
- threshold optimization on out-of-fold probabilities
- MLflow experiment tracking and local model registry support

## What you will likely customize next

- feature engineering based on all Home Credit source tables
- richer candidate models such as LightGBM
- more advanced imbalance handling
- deployment packaging for pre-production

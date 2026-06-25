# Home Credit MLOps

Base project for the OpenClassrooms "Initiez-vous au MLOps" credit scoring exercise.

This repository is organized around Python scripts instead of notebooks:

- `scripts/eda.py` for exploratory analysis
- `scripts/prepare_dataset.py` for a first cleaned dataset export
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
2. Generate a first report:

```bash
poetry run python scripts/eda.py --input data/raw/application_train.csv --target TARGET
```

3. Export a first processed table:

```bash
poetry run python scripts/prepare_dataset.py   --input data/raw/application_train.csv   --output data/processed/train_features.parquet   --target TARGET   --id-column SK_ID_CURR
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
